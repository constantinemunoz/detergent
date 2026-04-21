from __future__ import annotations

import cv2
import numpy as np

from .config import EffectSettings


def build_displacement(matte: np.ndarray, edge: np.ndarray, t_sec: float, settings: EffectSettings) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build dx/dy displacement maps and a visualization image in pixel units."""
    gy, gx = np.gradient(matte.astype(np.float32))

    h, w = matte.shape
    yy, xx = np.indices((h, w), dtype=np.float32)
    shimmer_phase = t_sec * settings.shimmer_speed

    # Continuous directional field for large cinematic warping.
    field_a = np.sin(xx * 0.017 + shimmer_phase * 1.8) + np.cos(yy * 0.021 - shimmer_phase * 1.3)
    field_b = np.cos((xx + yy) * 0.015 + shimmer_phase * 1.1) + np.sin((xx - yy) * 0.012 - shimmer_phase * 1.6)
    angle = field_a * 1.7 + field_b * 1.3
    dir_x = np.cos(angle)
    dir_y = np.sin(angle)

    # Additional high-frequency shimmer motion.
    shimmer_x = np.sin(xx * 0.061 + shimmer_phase * 2.2) * np.cos(yy * 0.033 - shimmer_phase * 1.7)
    shimmer_y = np.cos(xx * 0.047 - shimmer_phase * 1.9) * np.sin(yy * 0.058 + shimmer_phase * 2.1)

    edge_weight = settings.edge_distortion_boost * edge
    inner_weight = settings.interior_distortion_amount * matte
    total_weight = np.clip(edge_weight + inner_weight, 0.0, 4.0)

    # displacement_amount is interpreted directly as pixel displacement amplitude.
    base_amp = settings.displacement_amount
    grad_boost = 0.35 * base_amp
    shimmer_amp = settings.shimmer_amount * 0.2 * base_amp

    dx = ((dir_x * base_amp) + (gx * grad_boost) + (shimmer_x * shimmer_amp)) * total_weight
    dy = ((dir_y * base_amp) + (gy * grad_boost) + (shimmer_y * shimmer_amp)) * total_weight

    flow_vis = np.zeros((h, w, 3), dtype=np.uint8)
    scale = max(1.0, base_amp)
    flow_vis[..., 2] = np.clip((dx / scale) * 127.0 + 127.0, 0, 255).astype(np.uint8)
    flow_vis[..., 1] = np.clip((dy / scale) * 127.0 + 127.0, 0, 255).astype(np.uint8)
    flow_vis[..., 0] = (np.clip(total_weight / np.max(total_weight + 1e-6), 0, 1) * 255).astype(np.uint8)

    return dx.astype(np.float32), dy.astype(np.float32), flow_vis


def remap_bgr(img_bgr: np.ndarray, dx: np.ndarray, dy: np.ndarray, wrap: bool = True) -> np.ndarray:
    h, w = img_bgr.shape[:2]
    yy, xx = np.indices((h, w), dtype=np.float32)
    map_x = (xx + dx).astype(np.float32)
    map_y = (yy + dy).astype(np.float32)
    border_mode = cv2.BORDER_WRAP if wrap else cv2.BORDER_REFLECT
    return cv2.remap(img_bgr, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=border_mode)


def apply_chromatic_aberration(base: np.ndarray, dx: np.ndarray, dy: np.ndarray, amount: float, mask: np.ndarray) -> np.ndarray:
    if amount <= 0:
        return base

    shift = amount * 0.4
    r = remap_bgr(base, dx * shift, dy * shift, wrap=True)[..., 2]
    g = base[..., 1]
    b = remap_bgr(base, -dx * shift, -dy * shift, wrap=True)[..., 0]

    aberrated = cv2.merge([b, g, r])
    alpha = np.clip(mask[..., None], 0, 1)
    out = base.astype(np.float32) * (1 - alpha) + aberrated.astype(np.float32) * alpha
    return np.clip(out, 0, 255).astype(np.uint8)
