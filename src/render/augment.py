"""Moderate scan-like augmentation (printed-doc scope, not real-world capture).

Operates in IMAGE-pixel space. Photometric/noise/blur/JPEG ops don't move boxes; the one geometric
op (small rotation) transforms every box's corners and recomputes its axis-aligned bbox so labels
stay aligned. box_sanity.py re-validates all boxes afterward.
"""
from __future__ import annotations

import random

import cv2
import numpy as np


def _png_to_bgr(png: bytes) -> np.ndarray:
    arr = np.frombuffer(png, np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def _bgr_to_png(img: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", img)
    return buf.tobytes()


def _transform_box(box, M):
    x1, y1, x2, y2 = box
    pts = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)
    ones = np.ones((4, 1), dtype=np.float32)
    out = (M @ np.hstack([pts, ones]).T).T  # 4x2
    nx1, ny1 = out[:, 0].min(), out[:, 1].min()
    nx2, ny2 = out[:, 0].max(), out[:, 1].max()
    return [float(nx1), float(ny1), float(nx2), float(ny2)]


def _clip_box(box, w, h):
    x1, y1, x2, y2 = box
    return [max(0.0, min(w, x1)), max(0.0, min(h, y1)),
            max(0.0, min(w, x2)), max(0.0, min(h, y2))]


def _rotate(img, leaves, deg):
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), deg, 1.0)
    img = cv2.warpAffine(img, M, (w, h), borderValue=(255, 255, 255), flags=cv2.INTER_CUBIC)
    for lf in leaves:
        lf["bbox"] = _clip_box(_transform_box(lf["bbox"], M), w, h)
        if lf.get("lines"):
            lf["lines"] = [_clip_box(_transform_box(b, M), w, h) for b in lf["lines"]]
    return img


def _shadow(img, rng):
    h, w = img.shape[:2]
    mask = np.ones((h, w), np.float32)
    side = rng.choice(["l", "r", "t", "b"])
    grad = np.linspace(0.75, 1.0, w if side in "lr" else h, dtype=np.float32)
    if side == "r":
        grad = grad[::-1]
    if side in "lr":
        mask = np.tile(grad, (h, 1))
    else:
        mask = np.tile(grad[:, None], (1, w))
    return (img.astype(np.float32) * mask[..., None]).clip(0, 255).astype(np.uint8)


def augment(png: bytes, leaves: list[dict], cfg: dict, rng: random.Random) -> tuple[bytes, list[dict]]:
    if rng.random() > cfg.get("apply_prob", 0.85):
        return png, leaves
    img = _png_to_bgr(png)

    g = cfg.get("geometric", {})
    if rng.random() < g.get("rotate_prob", 0.0):
        deg = rng.uniform(*g["rotate_deg"])
        img = _rotate(img, leaves, deg)

    p = cfg.get("photometric", {})
    if rng.random() < p.get("brightness_contrast_prob", 0.0):
        alpha = 1.0 + rng.uniform(*p["contrast"])           # contrast
        beta = rng.uniform(*p["brightness"]) * 255          # brightness
        img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
    if rng.random() < p.get("gamma_prob", 0.0):
        gamma = rng.uniform(*p["gamma"]) / 100.0
        lut = np.array([((i / 255.0) ** (1.0 / gamma)) * 255 for i in range(256)]).astype(np.uint8)
        img = cv2.LUT(img, lut)

    nz = cfg.get("noise", {})
    if rng.random() < nz.get("gauss_prob", 0.0):
        var = rng.uniform(*nz["gauss_var"])
        noise = rng.gauss  # noqa
        g_noise = np.random.normal(0, var ** 0.5, img.shape).astype(np.float32)
        img = (img.astype(np.float32) + g_noise).clip(0, 255).astype(np.uint8)

    bl = cfg.get("blur", {})
    if rng.random() < bl.get("gaussian_prob", 0.0):
        k = rng.choice([x for x in range(bl["gaussian_kernel"][0], bl["gaussian_kernel"][1] + 1) if x % 2])
        img = cv2.GaussianBlur(img, (k, k), 0)
    elif rng.random() < bl.get("motion_prob", 0.0):
        k = rng.choice([x for x in range(bl["motion_kernel"][0], bl["motion_kernel"][1] + 1) if x % 2])
        kernel = np.zeros((k, k), np.float32)
        kernel[k // 2, :] = 1.0 / k
        img = cv2.filter2D(img, -1, kernel)

    pa = cfg.get("paper", {})
    if rng.random() < pa.get("shadow_prob", 0.0):
        img = _shadow(img, rng)

    cm = cfg.get("compression", {})
    if rng.random() < cm.get("jpeg_prob", 0.0):
        q = rng.randint(*cm["jpeg_quality"])
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])
        if ok:
            img = cv2.imdecode(buf, cv2.IMREAD_COLOR)

    return _bgr_to_png(img), leaves
