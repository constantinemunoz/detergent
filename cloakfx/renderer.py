from __future__ import annotations

from typing import Callable

import cv2
import numpy as np

from .chroma import create_alpha_matte, keyed_foreground, suppress_spill
from .config import EffectSettings
from .displacement import apply_chromatic_aberration, build_displacement, remap_bgr
from .matte import clean_matte, edge_mask
from .video_io import SyncedVideoPair, create_writer


def process_frame(bg: np.ndarray, fg: np.ndarray, t_sec: float, settings: EffectSettings) -> np.ndarray:
    raw_matte = create_alpha_matte(fg, settings)
    matte = clean_matte(raw_matte, settings)
    edge = edge_mask(matte, settings.edge_width)

    spill_fixed = suppress_spill(fg, matte, settings)
    keyed = keyed_foreground(spill_fixed, matte)

    # Two-layer refraction: subtle in interior, stronger on the edges.
    dx, dy, flow_vis = build_displacement(matte, edge, t_sec, settings)
    refract_strong = remap_bgr(bg, dx, dy)
    refract_soft = remap_bgr(bg, dx * 0.35, dy * 0.35)
    edge_mix = np.clip(edge[..., None], 0.0, 1.0)
    refracted = (refract_soft.astype(np.float32) * (1.0 - edge_mix) + refract_strong.astype(np.float32) * edge_mix).astype(np.uint8)

    if settings.blur_inside_matte > 0:
        blur_amount = int(max(1, round(settings.blur_inside_matte * 3)))
        k = blur_amount * 2 + 1
        blurred = cv2.GaussianBlur(refracted, (k, k), sigmaX=0)
        alpha = np.clip((matte * 0.8 + edge * 0.2)[..., None], 0, 1)
        refracted = (refracted.astype(np.float32) * (1 - alpha) + blurred.astype(np.float32) * alpha).astype(np.uint8)

    refracted = apply_chromatic_aberration(refracted, dx, dy, settings.chromatic_aberration_amount, np.clip(matte * 0.5 + edge * 0.5, 0, 1))

    # Only replace pixels inside the matte region, keeping scene untouched elsewhere.
    cloak_alpha = np.clip((matte * 0.65 + edge * 0.75)[..., None], 0.0, 1.0)
    cloaked_region = (bg.astype(np.float32) * (1.0 - cloak_alpha) + refracted.astype(np.float32) * cloak_alpha).astype(np.uint8)

    if settings.edge_highlight_amount > 0:
        highlight = np.zeros_like(bg, dtype=np.float32)
        highlight[..., 1] = 165
        highlight[..., 0] = 55
        highlight[..., 2] = 95
        cloaked_region = np.clip(
            cloaked_region.astype(np.float32) + highlight * (edge[..., None] * settings.edge_highlight_amount),
            0,
            255,
        ).astype(np.uint8)

    blend = float(np.clip(settings.blend_with_original, 0, 1))
    final = cv2.addWeighted(bg, 1.0 - blend, cloaked_region, blend, 0)

    dv = settings.debug_view
    if dv == "Foreground":
        return fg
    if dv == "Keyed Foreground":
        return keyed
    if dv == "Matte":
        matte_v = (matte * 255).astype(np.uint8)
        return cv2.cvtColor(matte_v, cv2.COLOR_GRAY2BGR)
    if dv == "Edge Mask":
        edge_v = (edge * 255).astype(np.uint8)
        return cv2.cvtColor(edge_v, cv2.COLOR_GRAY2BGR)
    if dv == "Displacement Field":
        return flow_vis
    return final


def render_video(
    bg_path: str,
    fg_path: str,
    out_path: str,
    settings: EffectSettings,
    progress_cb: Callable[[int, int], None] | None = None,
    preview_scale: float = 1.0,
) -> None:
    pair = SyncedVideoPair(bg_path, fg_path)
    output_size = (pair.output_w, pair.output_h)
    if preview_scale != 1.0:
        output_size = (max(8, int(output_size[0] * preview_scale)), max(8, int(output_size[1] * preview_scale)))

    writer = create_writer(out_path, pair.output_fps, output_size)
    try:
        for i in range(pair.total_frames):
            bg, fg, t = pair.get_frame_pair(i)
            frame = process_frame(bg, fg, t, settings)
            if preview_scale != 1.0:
                frame = cv2.resize(frame, output_size, interpolation=cv2.INTER_AREA)
            writer.write(frame)
            if progress_cb:
                progress_cb(i + 1, pair.total_frames)
    finally:
        writer.release()
        pair.release()


def preview_frame(bg_path: str, fg_path: str, settings: EffectSettings, frame_idx: int = 0) -> np.ndarray:
    pair = SyncedVideoPair(bg_path, fg_path)
    try:
        idx = max(0, min(frame_idx, pair.total_frames - 1))
        bg, fg, t = pair.get_frame_pair(idx)
        return process_frame(bg, fg, t, settings)
    finally:
        pair.release()


def auto_key_color_from_videos(bg_path: str, fg_path: str) -> tuple[int, int, int]:
    _ = bg_path
    from .chroma import auto_sample_key_color

    cap = cv2.VideoCapture(fg_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open foreground video: {fg_path}")
    try:
        ok, frame = cap.read()
        if not ok or frame is None:
            raise RuntimeError("Failed to read first foreground frame for key sampling.")
        return auto_sample_key_color(frame)
    finally:
        cap.release()
