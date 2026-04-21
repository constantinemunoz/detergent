from __future__ import annotations

import cv2
import numpy as np

from .config import EffectSettings


def auto_sample_key_color(frame_bgr: np.ndarray) -> tuple[int, int, int]:
    """Estimate chroma backdrop color from border pixels while ignoring letterbox blacks."""
    h, w = frame_bgr.shape[:2]
    border = max(6, min(h, w) // 18)
    top = frame_bgr[:border, :, :]
    bottom = frame_bgr[h - border :, :, :]
    left = frame_bgr[:, :border, :]
    right = frame_bgr[:, w - border :, :]
    samples = np.concatenate([top.reshape(-1, 3), bottom.reshape(-1, 3), left.reshape(-1, 3), right.reshape(-1, 3)], axis=0)

    hsv = cv2.cvtColor(samples.reshape(-1, 1, 3), cv2.COLOR_BGR2HSV).reshape(-1, 3)
    sat_mask = hsv[:, 1] > 35
    val_mask = hsv[:, 2] > 25
    clean = samples[sat_mask & val_mask]
    if clean.shape[0] < 100:
        clean = samples

    # Use median for robustness to outliers from subject overlap at edges.
    median = np.median(clean, axis=0).astype(np.uint8)
    return int(median[0]), int(median[1]), int(median[2])


def _color_distance_ycrcb(frame_bgr: np.ndarray, key_bgr: tuple[int, int, int]) -> np.ndarray:
    frame_ycc = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2YCrCb).astype(np.float32)
    key_patch = np.full((1, 1, 3), key_bgr, dtype=np.uint8)
    key_ycc = cv2.cvtColor(key_patch, cv2.COLOR_BGR2YCrCb).astype(np.float32)[0, 0]
    chroma_delta = frame_ycc[..., 1:] - key_ycc[1:]
    ycc_dist = np.linalg.norm(chroma_delta, axis=2) / 181.0

    frame_hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    key_hsv = cv2.cvtColor(key_patch, cv2.COLOR_BGR2HSV).astype(np.float32)[0, 0]
    hue = frame_hsv[..., 0]
    hue_delta = np.abs(hue - key_hsv[0])
    hue_delta = np.minimum(hue_delta, 180.0 - hue_delta) / 90.0
    sat = frame_hsv[..., 1] / 255.0
    # hue signal is less useful in unsaturated regions, so gate by saturation.
    hue_weighted = hue_delta * np.clip(sat * 1.4, 0.0, 1.0)

    # Combined chroma distance in [~0,1+].
    return (0.72 * ycc_dist + 0.28 * hue_weighted).astype(np.float32)


def create_alpha_matte(frame_bgr: np.ndarray, settings: EffectSettings) -> np.ndarray:
    dist = _color_distance_ycrcb(frame_bgr, settings.key_color_bgr)
    threshold = max(1e-5, settings.key_threshold)
    softness = max(1e-5, settings.key_softness)
    low, high = threshold, threshold + softness
    matte = np.clip((dist - low) / (high - low), 0.0, 1.0)
    return matte.astype(np.float32)


def suppress_spill(frame_bgr: np.ndarray, matte: np.ndarray, settings: EffectSettings) -> np.ndarray:
    frame = frame_bgr.astype(np.float32)
    b, g, r = cv2.split(frame)
    spill = np.clip(g - np.maximum(r, b), 0, 255)
    edge = cv2.GaussianBlur((1.0 - matte).astype(np.float32), (0, 0), 1.1)
    attenuation = settings.spill_suppression * np.clip(edge, 0.0, 1.0)
    g = g - spill * attenuation
    fixed = cv2.merge([b, g, r])
    return np.clip(fixed, 0, 255).astype(np.uint8)


def keyed_foreground(frame_bgr: np.ndarray, matte: np.ndarray) -> np.ndarray:
    alpha3 = matte[..., None]
    out = frame_bgr.astype(np.float32) * alpha3
    return np.clip(out, 0, 255).astype(np.uint8)
