from __future__ import annotations

from dataclasses import dataclass


@dataclass
class EffectSettings:
    key_color_bgr: tuple[int, int, int] = (0, 255, 0)
    key_threshold: float = 0.18
    key_softness: float = 0.12
    spill_suppression: float = 0.45
    matte_expand_contract: int = 0
    matte_blur: int = 3
    denoise_strength: int = 2
    displacement_amount: float = 13.0
    edge_distortion_boost: float = 1.4
    interior_distortion_amount: float = 0.45
    blur_inside_matte: float = 0.7
    edge_highlight_amount: float = 0.25
    edge_width: float = 3.0
    chromatic_aberration_amount: float = 1.2
    shimmer_amount: float = 1.8
    shimmer_speed: float = 1.1
    blend_with_original: float = 1.0
    debug_view: str = "Final Output"


DEBUG_VIEWS = [
    "Final Output",
    "Foreground",
    "Keyed Foreground",
    "Matte",
    "Edge Mask",
    "Displacement Field",
]
