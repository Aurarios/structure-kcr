"""Khmer text-line recognizer: CNN stem + transformer encoder (SVTR-flavored) + CTC head +
autoregressive attention decoder.

The attention decoder handles Khmer visual->logical reordering (pre-posed vowels ើ េ ែ, stacked
coeng) that pure CTC fights; the CTC head is an auxiliary alignment regularizer that speeds encoder
convergence. Final transcription is decoded from the attention head.

Input  : (B, 3, 48, W)  line crop, fixed height 48, variable width (right-padded in batch).
Encoder: stem downsamples H->1, W->W/8  =>  sequence (B, W/8, d).
Heads  : CTC linear (B, W/8, V) ; Transformer decoder cross-attending to the encoder sequence.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn


# --------------------------------------------------------------------------- CNN stem

def _conv_bn(cin, cout, k=3, s=1, p=1):
    return nn.Sequential(nn.Conv2d(cin, cout, k, s, p, bias=False),
                         nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class Stem(nn.Module):
    """48xW -> (d, 1, W/8). Three stride-2 stages downsample H 48->6, plus a final H-pool to 1."""

    def __init__(self, d: int):
        super().__init__()
        self.net = nn.Sequential(
            _conv_bn(3, d // 4),
            _conv_bn(d // 4, d // 4), nn.MaxPool2d((2, 2)),       # 48->24, W->W/2
            _conv_bn(d // 4, d // 2),
            _conv_bn(d // 2, d // 2), nn.MaxPool2d((2, 2)),       # 24->12, W->W/4
            _conv_bn(d // 2, d),
            _conv_bn(d, d), nn.MaxPool2d((2, 2)),                 # 12->6,  W->W/8
        )

    def forward(self, x):
        x = self.net(x)                # (B, d, 6, W/8)
        # collapse H by averaging the full height band. Mathematically identical to the previous
        # AdaptiveAvgPool2d((1, None)) but ONNX-exportable (adaptive pooling needs a constant size).
        x = x.mean(dim=2, keepdim=True)   # (B, d, 1, W/8)
        return x.squeeze(2).transpose(1, 2)    # (B, W/8, d)


# --------------------------------------------------------------------------- positional enc

class PositionalEncoding(nn.Module):
    def __init__(self, d, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d, 2).float() * (-math.log(10000.0) / d))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


# --------------------------------------------------------------------------- recognizer

@dataclass
class RecProfile:
    d_model: int
    enc_layers: int
    dec_layers: int
    heads: int
    ffn: int
    max_w_tokens: int     # max encoder sequence length (W/8)
    max_dec_len: int      # max target token length


PROFILES = {
    # max_dec_len must exceed the longest line's token count (a full article line can be
    # 60-120 tokens); set generously and clamp anomalies in the dataset.
    "single": RecProfile(d_model=256, enc_layers=6, dec_layers=2, heads=8, ffn=1024,
                         max_w_tokens=200, max_dec_len=256),
    "parallel": RecProfile(d_model=384, enc_layers=9, dec_layers=3, heads=8, ffn=1536,
                           max_w_tokens=256, max_dec_len=320),
}


class Recognizer(nn.Module):
    def __init__(self, profile: str, vocab_size: int, pad_id: int):
        super().__init__()
        p = PROFILES[profile]
        self.p = p
        self.pad_id = pad_id
        d = p.d_model
        self.stem = Stem(d)
        self.enc_pos = PositionalEncoding(d, p.max_w_tokens + 8)
        enc_layer = nn.TransformerEncoderLayer(d, p.heads, p.ffn, batch_first=True,
                                               activation="gelu", norm_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, p.enc_layers)
        self.ctc_head = nn.Linear(d, vocab_size)

        self.tok_emb = nn.Embedding(vocab_size, d, padding_idx=pad_id)
        self.dec_pos = PositionalEncoding(d, p.max_dec_len + 8)
        dec_layer = nn.TransformerDecoderLayer(d, p.heads, p.ffn, batch_first=True,
                                               activation="gelu", norm_first=True)
        self.decoder = nn.TransformerDecoder(dec_layer, p.dec_layers)
        self.out = nn.Linear(d, vocab_size)
        self.out.weight = self.tok_emb.weight   # tie

    def encode(self, images):
        feat = self.stem(images)                 # (B, T, d)
        feat = self.enc_pos(feat)
        return self.encoder(feat)                # (B, T, d)

    def forward(self, images, dec_inp):
        """Training forward. Returns (ctc_logits [B,T,V], attn_logits [B,L,V])."""
        memory = self.encode(images)
        ctc_logits = self.ctc_head(memory)
        L = dec_inp.size(1)
        tgt = self.dec_pos(self.tok_emb(dec_inp))
        causal = torch.triu(torch.full((L, L), float("-inf"), device=dec_inp.device), diagonal=1)
        pad_mask = dec_inp.eq(self.pad_id)
        h = self.decoder(tgt, memory, tgt_mask=causal, tgt_key_padding_mask=pad_mask)
        return ctc_logits, self.out(h)

    @torch.no_grad()
    def ctc_greedy(self, images):
        """Single-shot CTC decode from the auxiliary CTC head — NO autoregressive loop.

        Returns the per-timestep argmax ids (B, T) as a python list-of-lists; collapse repeats +
        drop the blank id with ``TextEncoder.ctc_collapse``. ~10-40x cheaper than ``greedy`` on CPU
        because it is one encoder forward + one linear, with no sequential decoder steps. Quality is
        lower than the attention head on Khmer reordering, so it is offered as a CPU "fast" path.
        """
        ctc_logits = self.ctc_head(self.encode(images))   # (B, T, V)
        return ctc_logits.argmax(-1).tolist()

    @torch.no_grad()
    def greedy(self, images, bos, eos, max_len=None, no_repeat_ngram: int = 3):
        """Autoregressive greedy decode from the attention head. Returns list[list[int]].

        ``no_repeat_ngram`` blocks the decoder from emitting a token that would complete an
        n-gram already generated in the same sequence — kills the repetition loops that an
        under-trained eos predictor falls into (transcribes the line, then repeats phrases).
        """
        max_len = max_len or self.p.max_dec_len
        memory = self.encode(images)
        B = images.size(0)
        ys = torch.full((B, 1), bos, dtype=torch.long, device=images.device)
        done = torch.zeros(B, dtype=torch.bool, device=images.device)
        for _ in range(max_len):
            tgt = self.dec_pos(self.tok_emb(ys))
            L = ys.size(1)
            causal = torch.triu(torch.full((L, L), float("-inf"), device=ys.device), diagonal=1)
            h = self.decoder(tgt, memory, tgt_mask=causal)
            logits = self.out(h[:, -1])
            if no_repeat_ngram and L >= no_repeat_ngram:
                n = no_repeat_ngram
                seqs = ys.tolist()
                for b in range(B):
                    prefix = tuple(seqs[b][-(n - 1):])
                    banned = set()
                    s = seqs[b]
                    for i in range(len(s) - n + 1):
                        if tuple(s[i:i + n - 1]) == prefix:
                            banned.add(s[i + n - 1])
                    for tok in banned:
                        logits[b, tok] = float("-inf")
            nxt = logits.argmax(-1)
            nxt = nxt.masked_fill(done, eos)
            ys = torch.cat([ys, nxt.unsqueeze(1)], dim=1)
            done = done | nxt.eq(eos)
            if bool(done.all()):
                break
        return [row[1:].tolist() for row in ys]   # drop bos


def build_recognizer(profile: str, vocab_size: int, pad_id: int) -> Recognizer:
    return Recognizer(profile, vocab_size, pad_id)


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())
