"""DBNet++ text detector — pure-PyTorch ResNet backbone + FPN neck + Differentiable-Binarization head.

Trained from scratch (no pretrained weights), so we implement ResNet here rather than depend on
torchvision. Profiles: 'resnet18' (~15M total, --single) and 'resnet50' (~26M backbone, --parallel).

Reference: Liao et al. AAAI 2020 (DBNet) and "Real-Time Scene Text Detection with Differentiable
Binarization and Adaptive Scale Fusion" (DBNet++).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.detect.data.build_det_targets import NUM_CLASSES


# --------------------------------------------------------------------------- ResNet backbone

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inp, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(inp, planes, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        idt = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            idt = self.downsample(x)
        return self.relu(out + idt)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inp, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(inp, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.downsample = downsample
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        idt = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            idt = self.downsample(x)
        return self.relu(out + idt)


class ResNet(nn.Module):
    """Returns C2,C3,C4,C5 feature maps (strides 4,8,16,32)."""

    def __init__(self, block, layers):
        super().__init__()
        self.inp = 64
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 7, 2, 3, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 1),
        )
        self.layer1 = self._make(block, 64, layers[0])
        self.layer2 = self._make(block, 128, layers[1], stride=2)
        self.layer3 = self._make(block, 256, layers[2], stride=2)
        self.layer4 = self._make(block, 512, layers[3], stride=2)
        self.out_channels = [64 * block.expansion, 128 * block.expansion,
                             256 * block.expansion, 512 * block.expansion]

    def _make(self, block, planes, n, stride=1):
        downsample = None
        if stride != 1 or self.inp != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inp, planes * block.expansion, 1, stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = [block(self.inp, planes, stride, downsample)]
        self.inp = planes * block.expansion
        for _ in range(1, n):
            layers.append(block(self.inp, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        c2 = self.layer1(x)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        return c2, c3, c4, c5


def _resnet(name):
    if name == "resnet18":
        return ResNet(BasicBlock, [2, 2, 2, 2])
    if name == "resnet50":
        return ResNet(Bottleneck, [3, 4, 6, 3])
    raise ValueError(f"unknown backbone {name}")


# --------------------------------------------------------------------------- FPN neck + DB head

class DBNet(nn.Module):
    def __init__(self, backbone: str = "resnet18", inner: int = 256, k: int = 50):
        super().__init__()
        self.backbone = _resnet(backbone)
        ch = self.backbone.out_channels
        self.k = k
        # lateral 1x1 to `inner`
        self.lat = nn.ModuleList([nn.Conv2d(c, inner, 1, bias=False) for c in ch])
        # smooth 3x3 producing inner//4 each, then concat -> inner
        self.smooth = nn.ModuleList([nn.Conv2d(inner, inner // 4, 3, 1, 1, bias=False) for _ in ch])

        def head(out_ch=1):
            return nn.Sequential(
                nn.Conv2d(inner, inner // 4, 3, 1, 1, bias=False),
                nn.BatchNorm2d(inner // 4), nn.ReLU(inplace=True),
                nn.ConvTranspose2d(inner // 4, inner // 4, 2, 2),
                nn.BatchNorm2d(inner // 4), nn.ReLU(inplace=True),
                nn.ConvTranspose2d(inner // 4, out_ch, 2, 2),
            )
        self.prob_head = head()
        self.thresh_head = head()
        self.class_head = head(NUM_CLASSES)   # per-pixel region-type logits (0=background)

    def _neck(self, feats):
        c2, c3, c4, c5 = feats
        p5 = self.lat[3](c5)
        p4 = self.lat[2](c4) + F.interpolate(p5, size=c4.shape[-2:], mode="nearest")
        p3 = self.lat[1](c3) + F.interpolate(p4, size=c3.shape[-2:], mode="nearest")
        p2 = self.lat[0](c2) + F.interpolate(p3, size=c2.shape[-2:], mode="nearest")
        outs = [self.smooth[0](p2), self.smooth[1](p3), self.smooth[2](p4), self.smooth[3](p5)]
        # upsample all to p2 (1/4) resolution and concat
        size = p2.shape[-2:]
        outs = [outs[0]] + [F.interpolate(o, size=size, mode="nearest") for o in outs[1:]]
        return torch.cat(outs, dim=1)   # inner channels

    def forward(self, x):
        fuse = self._neck(self.backbone(x))
        prob = self.prob_head(fuse)        # logits, full input res
        class_logits = self.class_head(fuse)
        if not self.training:
            return {"prob": torch.sigmoid(prob), "class_logits": class_logits}
        thresh = torch.sigmoid(self.thresh_head(fuse))
        prob_s = torch.sigmoid(prob)
        binary = 1.0 / (1.0 + torch.exp(-self.k * (prob_s - thresh)))
        return {"prob_logits": prob, "prob": prob_s, "thresh": thresh, "binary": binary,
                "class_logits": class_logits}


# --------------------------------------------------------------------------- loss

def _ohem_bce(logits, gt, mask, neg_ratio=3.0):
    """Balanced BCE with online hard-negative mining."""
    gt = gt.unsqueeze(1) if gt.dim() == 3 else gt
    mask = mask.unsqueeze(1) if mask.dim() == 3 else mask
    loss = F.binary_cross_entropy_with_logits(logits, gt, reduction="none")
    pos = (gt * mask).bool()
    neg = ((1 - gt) * mask).bool()
    n_pos = int(pos.sum().item())
    n_neg = min(int(neg.sum().item()), int(max(n_pos, 1) * neg_ratio))
    pos_loss = loss[pos].sum()
    neg_loss = loss[neg]
    if n_neg > 0 and neg_loss.numel() > 0:
        neg_loss, _ = neg_loss.topk(min(n_neg, neg_loss.numel()))
        neg_loss = neg_loss.sum()
    else:
        neg_loss = torch.tensor(0.0, device=logits.device)
    return (pos_loss + neg_loss) / (n_pos + n_neg + 1e-6)


def _dice(pred, gt, mask):
    gt = gt.unsqueeze(1) if gt.dim() == 3 else gt
    mask = mask.unsqueeze(1) if mask.dim() == 3 else mask
    pred = pred * mask
    gt = gt * mask
    inter = (pred * gt).sum()
    union = pred.sum() + gt.sum() + 1e-6
    return 1.0 - 2.0 * inter / union


def db_loss(out: dict, batch: dict, alpha: float = 1.0, beta: float = 10.0,
            gamma: float = 0.5) -> dict:
    ls = _ohem_bce(out["prob_logits"], batch["prob"], batch["prob_mask"])
    lb = _dice(out["binary"], batch["prob"], batch["prob_mask"])
    thr = batch["thresh"].unsqueeze(1)
    tmask = batch["thresh_mask"].unsqueeze(1)
    denom = tmask.sum() + 1e-6
    lt = (torch.abs(out["thresh"] - thr) * tmask).sum() / denom
    # class CE only on text pixels (gt prob > 0); background pixels are ignored to focus the head
    cls_t = batch["class_map"]                       # (B,H,W) long
    text = batch["prob"] > 0.5                        # (B,H,W) bool
    ignore = cls_t.clone()
    ignore[~text] = -100
    lc = F.cross_entropy(out["class_logits"], ignore, ignore_index=-100)
    total = ls + alpha * lb + beta * lt + gamma * lc
    return {"loss": total, "ls": ls, "lb": lb, "lt": lt, "lc": lc}


PROFILES = {
    "single": {"backbone": "resnet18", "inner": 256},
    "parallel": {"backbone": "resnet50", "inner": 256},
}


def build_detector(profile: str = "single") -> DBNet:
    p = PROFILES[profile]
    return DBNet(backbone=p["backbone"], inner=p["inner"])


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())
