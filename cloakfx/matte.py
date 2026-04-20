from __future__ import annotations

import cv2
import numpy as np

from .config import EffectSettings


def clean_matte(matte: np.ndarray, settings: EffectSettings) -> np.ndarray:
    m = np.clip(matte, 0.0, 1.0)

    if settings.denoise_strength > 0:
        k = max(1, settings.denoise_strength * 2 + 1)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, kernel)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel)

    if settings.matte_expand_contract != 0:
        k = abs(settings.matte_expand_contract) * 2 + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        if settings.matte_expand_contract > 0:
            m = cv2.dilate(m, kernel, iterations=1)
        else:
            m = cv2.erode(m, kernel, iterations=1)

    if settings.matte_blur > 0:
        blur_k = settings.matte_blur * 2 + 1
        m = cv2.GaussianBlur(m, (blur_k, blur_k), sigmaX=0, sigmaY=0)

    return np.clip(m, 0.0, 1.0).astype(np.float32)


def edge_mask(matte: np.ndarray, edge_width: float) -> np.ndarray:
    sigma = max(0.5, edge_width)
    grad_x = cv2.Sobel(matte, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(matte, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(grad_x, grad_y)
    mag = cv2.GaussianBlur(mag, (0, 0), sigmaX=sigma, sigmaY=sigma)
    if float(np.max(mag)) > 1e-6:
        mag = mag / float(np.max(mag))
    return np.clip(mag, 0.0, 1.0).astype(np.float32)
