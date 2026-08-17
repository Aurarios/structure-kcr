"""V4 layout detector — anchor-free FCOS-style object detector (from scratch).

Region-level box-regression detector (vs DBNet's per-pixel text segmentation), so it handles
overlapping/nested classes natively (a `table` region enclosing its cells, `chart` vs `image`,
`signature`). Reuses the project's own ResNet backbone from `dbnet.py` (no pretrained weights).

Pipeline: ResNet C3/C4/C5 -> FPN P3..P7 -> per-level Controllable Receptive Module (GL-CRM-inspired
multi-dilation fusion for the one-line-title -> full-page-table scale range) -> shared FCOS head
(focal classification + GIoU box regression as l,t,r,b distances + centerness). Decode in
`infer_obj.py`.

Reference: Tian et al., "FCOS: Fully Convolutional One-Stage Object Detection" (ICCV 2019), plus the
GL-CRM idea from DocLayout-YOLO (arXiv:2410.12628).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.detect.model.dbnet import _resnet                     # reuse the from-scratch ResNet
from src.detect.data.build_obj_targets import NUM_CLASSES_V4

STRIDES = [8, 16, 32, 64, 128]                                  # P3..P7
# Documents have extreme-aspect-ratio regions (a 900x40 title). Standard FCOS assigns a box to a
# level by its MAX side (half-width here -> a coarse level whose grid is too sparse to land any
# location inside the 40px-tall band -> the box is never learned). We instead assign by box HEIGHT,
# the discriminative scale for document text/blocks, so thin-wide titles land on fine P3.
HEIGHT_BOUNDS = [40.0, 80.0, 160.0, 360.0]                      # -> 5 levels by box height (px)
INF = 1e9


# ----------------------------------------------------------------- CRM (multi-receptive fusion)

class CRM(nn.Module):
    """Controllable Receptive Module: parallel dilated 3x3 convs (rates 1/2/3) fused -> multi-scale
    context per FPN level (cheap GL-CRM-inspired block)."""

    def __init__(self, ch: int):
        super().__init__()
        self.branches = nn.ModuleList([
            nn.Conv2d(ch, ch, 3, padding=d, dilation=d, bias=False) for d in (1, 2, 3)])
        self.bn = nn.GroupNorm(32, ch)
        self.fuse = nn.Conv2d(ch, ch, 1, bias=False)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        y = sum(b(x) for b in self.branches)
        return x + self.act(self.bn(self.fuse(y)))               # residual


# ----------------------------------------------------------------- FPN P3..P7

class FPN(nn.Module):
    def __init__(self, in_chs: list[int], inner: int = 256):
        super().__init__()
        # in_chs are C3,C4,C5 channels
        self.lat = nn.ModuleList([nn.Conv2d(c, inner, 1) for c in in_chs])
        self.out = nn.ModuleList([nn.Conv2d(inner, inner, 3, padding=1) for _ in in_chs])
        self.p6 = nn.Conv2d(inner, inner, 3, stride=2, padding=1)
        self.p7 = nn.Conv2d(inner, inner, 3, stride=2, padding=1)
        self.crm = nn.ModuleList([CRM(inner) for _ in range(5)])

    def forward(self, c3, c4, c5):
        p5 = self.lat[2](c5)
        p4 = self.lat[1](c4) + F.interpolate(p5, size=c4.shape[-2:], mode="nearest")
        p3 = self.lat[0](c3) + F.interpolate(p4, size=c3.shape[-2:], mode="nearest")
        p3, p4, p5 = self.out[0](p3), self.out[1](p4), self.out[2](p5)
        p6 = self.p6(p5)
        p7 = self.p7(F.relu(p6))
        feats = [p3, p4, p5, p6, p7]
        return [self.crm[i](f) for i, f in enumerate(feats)]


# ----------------------------------------------------------------- FCOS head

class Scale(nn.Module):
    def __init__(self, v=1.0):
        super().__init__()
        self.s = nn.Parameter(torch.tensor(float(v)))

    def forward(self, x):
        return x * self.s


class FCOSHead(nn.Module):
    def __init__(self, num_classes: int, inner: int = 256, n_levels: int = 5, n_conv: int = 4):
        super().__init__()
        def tower():
            layers = []
            for _ in range(n_conv):
                layers += [nn.Conv2d(inner, inner, 3, padding=1, bias=False),
                           nn.GroupNorm(32, inner), nn.ReLU(inplace=True)]
            return nn.Sequential(*layers)
        self.cls_tower = tower()
        self.reg_tower = tower()
        self.cls_logits = nn.Conv2d(inner, num_classes, 3, padding=1)
        self.reg = nn.Conv2d(inner, 4, 3, padding=1)
        self.ctr = nn.Conv2d(inner, 1, 3, padding=1)
        self.scales = nn.ModuleList([Scale(1.0) for _ in range(n_levels)])
        # init: cls bias so initial prob ~0.01 (focal-loss stability)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.constant_(self.cls_logits.bias, -math.log((1 - 0.01) / 0.01))

    def forward(self, feats):
        cls_out, reg_out, ctr_out = [], [], []
        for i, f in enumerate(feats):
            ct = self.cls_tower(f)
            rt = self.reg_tower(f)
            cls_out.append(self.cls_logits(ct))
            ctr_out.append(self.ctr(ct))
            reg_out.append(torch.exp(self.scales[i](self.reg(rt))))   # positive distances (stride units)
        return cls_out, reg_out, ctr_out


class LayoutHead(nn.Module):
    """Page-level layout classifier over the COARSEST FPN level (widest receptive field).

    Auxiliary/multi-task, NOT a first stage: it shapes the shared backbone with page context while
    detection stays one forward pass with no cascade to propagate errors. A hard layout-first router
    was considered and rejected — the ambiguities that motivated it (word_bank vs list_item) live
    INSIDE a layout, so routing cannot resolve them, and per-layout detection F1 was already
    0.95-1.00. At inference this head is simply ignored.
    """

    def __init__(self, inner: int, num_layouts: int):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(inner, inner), nn.ReLU(inplace=True),
                                nn.Dropout(0.1), nn.Linear(inner, num_layouts))

    def forward(self, feats):
        return self.fc(feats[-1].mean(dim=(2, 3)))       # global average pool -> logits


class FCOS(nn.Module):
    def __init__(self, backbone: str = "resnet18", inner: int = 256,
                 num_classes: int = NUM_CLASSES_V4, num_layouts: int = 0):
        super().__init__()
        self.backbone = _resnet(backbone)
        self.num_classes = num_classes
        self.num_layouts = num_layouts
        c3, c4, c5 = self.backbone.out_channels[1:]      # out_channels = [C2,C3,C4,C5]
        self.fpn = FPN([c3, c4, c5], inner)
        self.head = FCOSHead(num_classes, inner)
        # only built when requested, so V4/V5 state_dicts still load with strict=True
        self.layout_head = LayoutHead(inner, num_layouts) if num_layouts else None

    def forward(self, x):
        _, c3, c4, c5 = self.backbone(x)
        feats = self.fpn(c3, c4, c5)
        cls_out, reg_out, ctr_out = self.head(feats)
        sizes = [f.shape[-2:] for f in feats]
        out = {"cls": cls_out, "reg": reg_out, "ctr": ctr_out, "sizes": sizes}
        if self.layout_head is not None:
            out["layout"] = self.layout_head(feats)
        return out


# ----------------------------------------------------------------- locations + target assignment

def locations_for(sizes, strides, device):
    """List of (H*W, 2) center coordinates (input-px) per level."""
    locs = []
    for (h, w), s in zip(sizes, strides):
        ys, xs = torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device),
                                indexing="ij")
        loc = torch.stack([(xs.reshape(-1) + 0.5) * s, (ys.reshape(-1) + 0.5) * s], dim=1)
        locs.append(loc)
    return locs


def _assign_image(locations, level_stride, level_idx, boxes, labels, num_classes, topk=9):
    """ATSS-style assignment for one image — robust to extreme document aspect ratios.

    For each GT, take the `topk` center-nearest locations PER FPN level as candidates (so thin-wide
    titles get positives on fine levels and big regions on coarse levels, with no hand-tuned size
    ranges), then keep candidates whose center is INSIDE the box. Overlaps tie-break by smallest area.

    locations:[L,2], level_stride:[L], level_idx:[L] (0..4), boxes:[N,4], labels:[N].
    Returns labels_t:[L], reg_t:[L,4] (stride-norm l,t,r,b), ctr_t:[L], pos_mask:[L].
    """
    L = locations.shape[0]
    device = locations.device
    if boxes.numel() == 0:
        return (torch.full((L,), num_classes, dtype=torch.long, device=device),
                torch.zeros(L, 4, device=device), torch.zeros(L, device=device),
                torch.zeros(L, dtype=torch.bool, device=device))
    N = boxes.shape[0]
    xs, ys = locations[:, 0:1], locations[:, 1:2]              # [L,1]
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]   # [N]
    l = xs - x1[None]; t = ys - y1[None]; r = x2[None] - xs; b = y2[None] - ys   # [L,N]
    reg = torch.stack([l, t, r, b], dim=-1)                    # [L,N,4]
    inside = reg.min(-1).values > 0                            # [L,N]
    cx = (x1 + x2) / 2; cy = (y1 + y2) / 2
    cdist = (xs - cx[None]) ** 2 + (ys - cy[None]) ** 2        # [L,N] center distance
    # ATSS candidates: topk nearest-center locations per level per GT
    cand = torch.zeros(L, N, dtype=torch.bool, device=device)
    for lv in range(int(level_idx.max().item()) + 1):
        idx = (level_idx == lv).nonzero(as_tuple=True)[0]
        if idx.numel() == 0:
            continue
        k = min(topk, idx.numel())
        sel = cdist[idx].topk(k, dim=0, largest=False).indices  # [k,N] -> rows into idx
        cand[idx[sel], torch.arange(N, device=device).expand(k, N)] = True
    valid = cand & inside                                     # [L,N]
    area = ((x2 - x1) * (y2 - y1))[None, :].expand(L, -1).clone()
    area[~valid] = INF
    min_area, gt_idx = area.min(1)                            # [L]
    pos = min_area < INF
    labels_t = torch.full((L,), num_classes, dtype=torch.long, device=device)
    labels_t[pos] = labels[gt_idx[pos]]
    reg_t = reg[torch.arange(L, device=device), gt_idx] / level_stride[:, None]
    lr = reg_t[:, [0, 2]]; tb = reg_t[:, [1, 3]]
    ctr_t = torch.sqrt((lr.min(1).values / lr.max(1).values.clamp(min=1e-6)) *
                       (tb.min(1).values / tb.max(1).values.clamp(min=1e-6)).clamp(min=0))
    ctr_t = torch.where(pos, ctr_t, torch.zeros_like(ctr_t))
    return labels_t, reg_t, ctr_t, pos


# ----------------------------------------------------------------- losses

def _giou_ltrb(pred, target):
    """GIoU loss on (l,t,r,b) distance boxes. pred,target: [P,4]."""
    pl, pt, pr, pb = pred.unbind(-1)
    tl, tt, tr, tb = target.unbind(-1)
    area_p = (pl + pr) * (pt + pb)
    area_t = (tl + tr) * (tt + tb)
    w_i = torch.min(pl, tl) + torch.min(pr, tr)
    h_i = torch.min(pt, tt) + torch.min(pb, tb)
    inter = w_i.clamp(min=0) * h_i.clamp(min=0)
    union = area_p + area_t - inter + 1e-7
    iou = inter / union
    w_c = torch.max(pl, tl) + torch.max(pr, tr)
    h_c = torch.max(pt, tt) + torch.max(pb, tb)
    area_c = w_c * h_c + 1e-7
    giou = iou - (area_c - union) / area_c
    return (1 - giou)


def fcos_loss(out: dict, targets: list[dict], num_classes: int = NUM_CLASSES_V4,
              focal_alpha: float = 0.25, focal_gamma: float = 2.0,
              layout_weight: float = 0.1) -> dict:
    """targets: list (len B) of {'boxes':[N,4] input-px, 'labels':[N], optional 'layout':int}.

    When the model carries an auxiliary layout head AND targets supply `layout`, a small-weight
    cross-entropy is added. The weight is deliberately low: the head exists to shape shared
    features, not to compete with detection for capacity.
    """
    device = out["cls"][0].device
    sizes = out["sizes"]
    locs = locations_for(sizes, STRIDES, device)
    stride_pl = torch.cat([torch.full((l.shape[0],), s, device=device, dtype=torch.float)
                           for l, s in zip(locs, STRIDES)])
    levelidx_pl = torch.cat([torch.full((locs[i].shape[0],), i, device=device, dtype=torch.long)
                             for i in range(len(locs))])
    all_locs = torch.cat(locs, 0)                              # [L,2]

    B = len(targets)
    # flatten predictions to [B, L, *]
    cls_f = torch.cat([o.permute(0, 2, 3, 1).reshape(B, -1, num_classes) for o in out["cls"]], 1)
    reg_f = torch.cat([o.permute(0, 2, 3, 1).reshape(B, -1, 4) for o in out["reg"]], 1)
    ctr_f = torch.cat([o.permute(0, 2, 3, 1).reshape(B, -1) for o in out["ctr"]], 1)

    cls_loss = reg_loss = ctr_loss = torch.tensor(0.0, device=device)
    total_pos = 0
    for bi in range(B):
        boxes = targets[bi]["boxes"].to(device).float()
        labels = targets[bi]["labels"].to(device).long()
        lab_t, reg_t, ctr_t, pos = _assign_image(all_locs, stride_pl, levelidx_pl, boxes, labels, num_classes)
        # focal classification (one-hot over C; background -> all zeros)
        oh = torch.zeros_like(cls_f[bi])
        p = pos.nonzero(as_tuple=True)[0]
        if p.numel():
            oh[p, lab_t[p]] = 1.0
        prob = cls_f[bi].sigmoid()
        ce = F.binary_cross_entropy_with_logits(cls_f[bi], oh, reduction="none")
        pt = prob * oh + (1 - prob) * (1 - oh)
        alpha = focal_alpha * oh + (1 - focal_alpha) * (1 - oh)
        cls_loss = cls_loss + (alpha * (1 - pt).pow(focal_gamma) * ce).sum()
        if p.numel():
            reg_loss = reg_loss + (_giou_ltrb(reg_f[bi][p], reg_t[p]) * ctr_t[p]).sum()
            ctr_loss = ctr_loss + F.binary_cross_entropy_with_logits(
                ctr_f[bi][p], ctr_t[p], reduction="sum")
        total_pos += int(p.numel())
    n = max(total_pos, 1)
    cls_loss = cls_loss / n
    reg_loss = reg_loss / n
    ctr_loss = ctr_loss / n
    total = cls_loss + reg_loss + ctr_loss
    lay_loss = torch.tensor(0.0, device=device)
    if "layout" in out and targets and targets[0].get("layout") is not None:
        lay_t = torch.tensor([int(t["layout"]) for t in targets], device=device, dtype=torch.long)
        lay_loss = F.cross_entropy(out["layout"], lay_t)
        total = total + layout_weight * lay_loss
    return {"loss": total, "lc": cls_loss, "lr": reg_loss, "lctr": ctr_loss,
            "llay": lay_loss, "n_pos": total_pos}


PROFILES = {"single": {"backbone": "resnet18", "inner": 256},
            "parallel": {"backbone": "resnet50", "inner": 256}}


def build_obj_detector(profile: str = "single", num_classes: int = NUM_CLASSES_V4,
                       num_layouts: int = 0) -> FCOS:
    p = PROFILES[profile]
    return FCOS(backbone=p["backbone"], inner=p["inner"], num_classes=num_classes,
                num_layouts=num_layouts)


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())
