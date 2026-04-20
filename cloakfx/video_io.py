from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class VideoMeta:
    width: int
    height: int
    fps: float
    frame_count: int
    duration: float


class SyncedVideoPair:
    def __init__(self, bg_path: str, fg_path: str):
        self.bg_cap = cv2.VideoCapture(bg_path)
        self.fg_cap = cv2.VideoCapture(fg_path)

        if not self.bg_cap.isOpened():
            raise ValueError(f"Could not open background video: {bg_path}")
        if not self.fg_cap.isOpened():
            raise ValueError(f"Could not open foreground video: {fg_path}")

        self.bg_meta = self._read_meta(self.bg_cap)
        self.fg_meta = self._read_meta(self.fg_cap)

        self.output_w = self.bg_meta.width
        self.output_h = self.bg_meta.height
        self.output_fps = self.bg_meta.fps if self.bg_meta.fps > 0 else (self.fg_meta.fps or 30.0)

        self.duration = min(self.bg_meta.duration, self.fg_meta.duration)
        self.total_frames = max(1, int(self.duration * self.output_fps))

    @staticmethod
    def _read_meta(cap: cv2.VideoCapture) -> VideoMeta:
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        if fps <= 1e-6:
            fps = 30.0
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        duration = (frame_count / fps) if frame_count > 0 else 0.0
        return VideoMeta(width=w, height=h, fps=fps, frame_count=frame_count, duration=duration)

    def _read_at_time(self, cap: cv2.VideoCapture, meta: VideoMeta, t_sec: float) -> np.ndarray:
        idx = int(round(t_sec * meta.fps))
        idx = min(max(0, idx), max(0, meta.frame_count - 1))
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok or frame is None:
            raise RuntimeError("Failed reading a video frame.")
        return frame

    def get_frame_pair(self, output_frame_idx: int) -> tuple[np.ndarray, np.ndarray, float]:
        t = output_frame_idx / self.output_fps
        bg = self._read_at_time(self.bg_cap, self.bg_meta, t)
        fg = self._read_at_time(self.fg_cap, self.fg_meta, t)
        fg = cv2.resize(fg, (self.output_w, self.output_h), interpolation=cv2.INTER_LINEAR)
        bg = cv2.resize(bg, (self.output_w, self.output_h), interpolation=cv2.INTER_LINEAR)
        return bg, fg, t

    def release(self) -> None:
        self.bg_cap.release()
        self.fg_cap.release()


def create_writer(path: str, fps: float, size: tuple[int, int]) -> cv2.VideoWriter:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, fps, size)
    if not writer.isOpened():
        raise ValueError(f"Failed to create output writer for: {path}")
    return writer
