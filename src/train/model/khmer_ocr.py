"""Build a random-init Swin + BART encoder-decoder sized per profile.

Two profiles:
  single   ~50M params, 768x768 input  -> fits comfortably on a 4070 Ti 12GB
  parallel ~250M params, 1280x1280 input -> targets 3x A5000 24GB w/ DDP

We use HuggingFace `transformers` configs + `from_config()` so the architecture is standard and
exportable, but the weights are randomly initialized — this is training from scratch in the sense
that matters (no upstream pretraining), while reusing well-tested layer implementations.
"""
from __future__ import annotations

from dataclasses import dataclass

from transformers import (
    BartConfig,
    BartForCausalLM,
    SwinConfig,
    SwinModel,
    VisionEncoderDecoderConfig,
    VisionEncoderDecoderModel,
)


@dataclass
class ModelProfile:
    name: str                       # "single" | "parallel"
    image_size: tuple[int, int]     # (H, W) — portrait to match document aspect ratio
    enc_embed_dim: int
    enc_depths: tuple[int, ...]
    enc_num_heads: tuple[int, ...]
    enc_window_size: int
    dec_d_model: int
    dec_layers: int
    dec_heads: int
    dec_ffn_dim: int
    max_position_embeddings: int


PROFILES = {
    # ~50M total: Swin-Tiny-ish (~22M) + small BART decoder (~28M)
    "single": ModelProfile(
        name="single",
        image_size=(1024, 768),
        enc_embed_dim=96,
        enc_depths=(2, 2, 6, 2),
        enc_num_heads=(3, 6, 12, 24),
        enc_window_size=8,
        dec_d_model=512,
        dec_layers=4,
        dec_heads=8,
        dec_ffn_dim=2048,
        max_position_embeddings=1536,
    ),
    # ~250M total: Swin-Base (~88M) + BART-base-ish decoder (~160M with larger ffn)
    "parallel": ModelProfile(
        name="parallel",
        image_size=(1536, 1024),
        enc_embed_dim=128,
        enc_depths=(2, 2, 18, 2),
        enc_num_heads=(4, 8, 16, 32),
        enc_window_size=8,
        dec_d_model=1024,
        dec_layers=6,
        dec_heads=16,
        dec_ffn_dim=4096,
        max_position_embeddings=2048,
    ),
}


def build_model(profile_name: str, vocab_size: int, pad_token_id: int,
                bos_token_id: int, eos_token_id: int) -> VisionEncoderDecoderModel:
    if profile_name not in PROFILES:
        raise ValueError(f"unknown profile {profile_name!r}; expected one of {list(PROFILES)}")
    p = PROFILES[profile_name]

    enc_cfg = SwinConfig(
        image_size=list(p.image_size),
        patch_size=4,
        num_channels=3,
        embed_dim=p.enc_embed_dim,
        depths=list(p.enc_depths),
        num_heads=list(p.enc_num_heads),
        window_size=p.enc_window_size,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_path_rate=0.1,
        use_absolute_embeddings=False,
    )
    dec_cfg = BartConfig(
        vocab_size=vocab_size,
        d_model=p.dec_d_model,
        decoder_layers=p.dec_layers,
        decoder_attention_heads=p.dec_heads,
        decoder_ffn_dim=p.dec_ffn_dim,
        # encoder side of BART is unused (we feed Swin features via cross-attn), but the config
        # requires it; mirror the decoder dims to keep cross-attn projections compatible
        encoder_layers=p.dec_layers,
        encoder_attention_heads=p.dec_heads,
        encoder_ffn_dim=p.dec_ffn_dim,
        max_position_embeddings=p.max_position_embeddings,
        pad_token_id=pad_token_id,
        bos_token_id=bos_token_id,
        eos_token_id=eos_token_id,
        decoder_start_token_id=bos_token_id,
        is_decoder=True,
        add_cross_attention=True,
        tie_word_embeddings=True,
        scale_embedding=True,
    )

    encoder = SwinModel(enc_cfg)
    decoder = BartForCausalLM(dec_cfg)

    cfg = VisionEncoderDecoderConfig.from_encoder_decoder_configs(enc_cfg, dec_cfg)
    cfg.pad_token_id = pad_token_id
    cfg.decoder_start_token_id = bos_token_id
    cfg.bos_token_id = bos_token_id
    cfg.eos_token_id = eos_token_id
    cfg.tie_word_embeddings = True

    model = VisionEncoderDecoderModel(encoder=encoder, decoder=decoder, config=cfg)
    return model


def count_params(model) -> int:
    return sum(p.numel() for p in model.parameters())
