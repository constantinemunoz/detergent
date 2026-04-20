from __future__ import annotations

import cv2
import numpy as np

from .config import EffectSettings


def build_displacement(matte: np.ndarray, edge: np.ndarray, t_sec: float, settings: EffectSettings) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build dx/dy displacement maps and a visualization image."""
    gy, gx = np.gradient(matte.astype(np.float32))

    h, w = matte.shape
    yy, xx = np.indices((h, w), dtype=np.float32)

    shimmer_phase = t_sec * settings.shimmer_speed
    noise = (
        np.sin(xx * 0.043 + shimmer_phase * 2.4)
        + np.cos(yy * 0.051 - shimmer_phase * 1.7)
        + np.sin((xx + yy) * 0.025 + shimmer_phase * 1.1)
    )
    noise *= settings.shimmer_amount * 0.5

    edge_weight = settings.edge_distortion_boost * edge
    inner_weight = settings.interior_distortion_amount * matte
    total_weight = np.clip(edge_weight + inner_weight, 0.0, 2.5)

    dx = (gx * settings.displacement_amount + noise * 0.35) * total_weight
    dy = (gy * settings.displacement_amount + noise * 0.35) * total_weight

    flow_vis = np.zeros((h, w, 3), dtype=np.uint8)
    flow_vis[..., 2] = np.clip((dx * 12.0) + 127, 0, 255).astype(np.uint8)
    flow_vis[..., 1] = np.clip((dy * 12.0) + 127, 0, 255).astype(np.uint8)
    flow_vis[..., 0] = (np.clip(total_weight, 0, 1) * 255).astype(np.uint8)

    return dx.astype(np.float32), dy.astype(np.float32), flow_vis


def remap_bgr(img_bgr: np.ndarray, dx: np.ndarray, dy: np.ndarray) -> np.ndarray:
    h, w = img_bgr.shape[:2]
    yy, xx = np.indices((h, w), dtype=np.float32)
    map_x = np.clip(xx + dx, 0, w - 1).astype(np.float32)
    map_y = np.clip(yy + dy, 0, h - 1).astype(np.float32)
    return cv2.remap(img_bgr, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def apply_chromatic_aberration(base: np.ndarray, dx: np.ndarray, dy: np.ndarray, amount: float, mask: np.ndarray) -> np.ndarray:
    if amount <= 0:
        return base

    shift = amount * 0.4
    r = remap_bgr(base, dx * shift, dy * shift)[..., 2]
    g = base[..., 1]
    b = remap_bgr(base, -dx * shift, -dy * shift)[..., 0]

    aberrated = cv2.merge([b, g, r])
    alpha = np.clip(mask[..., None], 0, 1)
    out = base.astype(np.float32) * (1 - alpha) + aberrated.astype(np.float32) * alpha
    return np.clip(out, 0, 255).astype(np.uint8)
