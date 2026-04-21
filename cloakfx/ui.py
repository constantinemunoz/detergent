from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import cv2
from PySide6.QtCore import QThread, Qt, Signal
from PySide6.QtGui import QColor, QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QColorDialog,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QProgressBar,
    QScrollArea,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from .config import DEBUG_VIEWS, EffectSettings
from .renderer import auto_key_color_from_videos, preview_frame, render_video


class RenderWorker(QThread):
    progress = Signal(int, int)
    done = Signal(str)
    failed = Signal(str)

    def __init__(self, bg_path: str, fg_path: str, out_path: str, settings: EffectSettings):
        super().__init__()
        self.bg_path = bg_path
        self.fg_path = fg_path
        self.out_path = out_path
        self.settings = settings

    def run(self) -> None:
        try:
            render_video(self.bg_path, self.fg_path, self.out_path, self.settings, progress_cb=self.progress.emit)
            self.done.emit(self.out_path)
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(str(exc))


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("CloakFX - Invisible Refraction Renderer")
        self.resize(1400, 900)

        self.settings = EffectSettings()
        self.render_worker: RenderWorker | None = None

        self.bg_edit = QLineEdit()
        self.fg_edit = QLineEdit()
        self.out_edit = QLineEdit(str(Path.cwd() / "cloak_output.mp4"))

        self.preview_label = QLabel("Preview will appear here")
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setMinimumSize(960, 540)
        self.preview_label.setStyleSheet("background:#101010; color:#cccccc; border:1px solid #333;")

        self.progress = QProgressBar()

        root = QWidget()
        self.setCentralWidget(root)
        main = QHBoxLayout(root)

        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.addWidget(self._build_file_group())
        left_layout.addWidget(self._build_controls())
        left_layout.addStretch(1)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(left_panel)
        scroll.setMinimumWidth(380)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.addWidget(self.preview_label, stretch=1)
        right_layout.addWidget(self.progress)

        button_row = QHBoxLayout()
        self.preview_btn = QPushButton("Preview Frame")
        self.preview_btn.clicked.connect(self.on_preview)
        self.render_btn = QPushButton("Render Export")
        self.render_btn.clicked.connect(self.on_render)
        button_row.addWidget(self.preview_btn)
        button_row.addWidget(self.render_btn)
        right_layout.addLayout(button_row)

        main.addWidget(scroll)
        main.addWidget(right, stretch=1)

    def _build_file_group(self) -> QWidget:
        box = QGroupBox("Videos")
        form = QGridLayout(box)

        def add_row(row: int, label: str, edit: QLineEdit, handler):
            form.addWidget(QLabel(label), row, 0)
            form.addWidget(edit, row, 1)
            btn = QPushButton("Browse")
            btn.clicked.connect(handler)
            form.addWidget(btn, row, 2)

        add_row(0, "Background", self.bg_edit, self.pick_bg)
        add_row(1, "Foreground/Roto", self.fg_edit, self.pick_fg)
        add_row(2, "Output", self.out_edit, self.pick_out)

        color_btn = QPushButton("Pick Key Color")
        color_btn.clicked.connect(self.pick_key_color)
        auto_btn = QPushButton("Auto Sample Key Color")
        auto_btn.clicked.connect(self.auto_key_color)
        preset_btn = QPushButton("Invisible Preset")
        preset_btn.clicked.connect(self.apply_invisible_preset)
        form.addWidget(color_btn, 3, 1)
        form.addWidget(auto_btn, 3, 2)
        form.addWidget(preset_btn, 4, 1, 1, 2)

        return box

    def _slider(self, minimum: int, maximum: int, value: int, on_change) -> QSlider:
        s = QSlider(Qt.Horizontal)
        s.setRange(minimum, maximum)
        s.setValue(value)
        s.valueChanged.connect(on_change)
        return s

    def _build_controls(self) -> QWidget:
        box = QGroupBox("Effect Controls")
        form = QFormLayout(box)

        self.debug_combo = QComboBox()
        self.debug_combo.addItems(DEBUG_VIEWS)
        self.debug_combo.currentTextChanged.connect(lambda v: setattr(self.settings, "debug_view", v))
        form.addRow("Debug View", self.debug_combo)

        sliders = [
            ("key_threshold", "Key Threshold", 1, 100, int(self.settings.key_threshold * 100), 0.01),
            ("key_softness", "Key Softness", 1, 100, int(self.settings.key_softness * 100), 0.01),
            ("spill_suppression", "Spill Suppression", 0, 100, int(self.settings.spill_suppression * 100), 0.01),
            ("matte_expand_contract", "Matte Expand/Contract", -10, 10, self.settings.matte_expand_contract, 1.0),
            ("matte_blur", "Matte Blur/Feather", 0, 12, self.settings.matte_blur, 1.0),
            ("denoise_strength", "Denoise Cleanup", 0, 6, self.settings.denoise_strength, 1.0),
            ("displacement_amount", "Displacement Amount", 0, 600, int(self.settings.displacement_amount), 1.0),
            ("edge_distortion_boost", "Edge Distortion Boost", 0, 600, int(self.settings.edge_distortion_boost * 100), 0.01),
            ("interior_distortion_amount", "Interior Distortion", 0, 600, int(self.settings.interior_distortion_amount * 100), 0.01),
            ("blur_inside_matte", "Blur Inside Matte", 0, 300, int(self.settings.blur_inside_matte * 100), 0.01),
            ("edge_highlight_amount", "Edge Highlight Amount", 0, 100, int(self.settings.edge_highlight_amount * 100), 0.01),
            ("edge_width", "Edge Width", 1, 100, int(self.settings.edge_width * 10), 0.1),
            ("chromatic_aberration_amount", "Chromatic Aberration", 0, 300, int(self.settings.chromatic_aberration_amount * 100), 0.01),
            ("shimmer_amount", "Shimmer Amount", 0, 500, int(self.settings.shimmer_amount * 100), 0.01),
            ("shimmer_speed", "Shimmer Speed", 1, 600, int(self.settings.shimmer_speed * 100), 0.01),
            ("blend_with_original", "Blend With Original", 0, 100, int(self.settings.blend_with_original * 100), 0.01),
        ]

        for key, label, min_v, max_v, init_v, scale in sliders:
            value_label = QLabel(f"{getattr(self.settings, key):.2f}" if scale != 1.0 else str(getattr(self.settings, key)))

            def update(v: int, key=key, scale=scale, value_label=value_label):
                new_val = int(v) if scale == 1.0 else float(v * scale)
                setattr(self.settings, key, new_val)
                value_label.setText(f"{new_val:.2f}" if scale != 1.0 else str(new_val))

            row = QWidget()
            lay = QHBoxLayout(row)
            lay.setContentsMargins(0, 0, 0, 0)
            lay.addWidget(self._slider(min_v, max_v, init_v, update), stretch=1)
            lay.addWidget(value_label)
            form.addRow(label, row)

        return box

    def pick_bg(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select Background Video")
        if path:
            self.bg_edit.setText(path)

    def pick_fg(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select Foreground Video")
        if path:
            self.fg_edit.setText(path)

    def pick_out(self):
        path, _ = QFileDialog.getSaveFileName(self, "Select Output Video", self.out_edit.text(), "MP4 files (*.mp4)")
        if path:
            self.out_edit.setText(path)

    def pick_key_color(self):
        b, g, r = self.settings.key_color_bgr
        color = QColorDialog.getColor(QColor(r, g, b), self, "Choose key color")
        if color.isValid():
            self.settings.key_color_bgr = (color.blue(), color.green(), color.red())

    def auto_key_color(self):
        fg_path = self.fg_edit.text().strip()
        if not fg_path:
            QMessageBox.warning(self, "Missing foreground", "Please pick a foreground/roto video first.")
            return
        try:
            self.settings.key_color_bgr = auto_key_color_from_videos(self.bg_edit.text().strip(), fg_path)
            b, g, r = self.settings.key_color_bgr
            QMessageBox.information(self, "Sampled", f"Sampled key color BGR=({b}, {g}, {r})")
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Auto sample failed", str(exc))


    def apply_invisible_preset(self):
        self.settings.key_threshold = 0.14
        self.settings.key_softness = 0.20
        self.settings.spill_suppression = 0.55
        self.settings.matte_expand_contract = 0
        self.settings.matte_blur = 4
        self.settings.denoise_strength = 2
        self.settings.displacement_amount = 180
        self.settings.edge_distortion_boost = 2.5
        self.settings.interior_distortion_amount = 1.2
        self.settings.blur_inside_matte = 0.8
        self.settings.edge_highlight_amount = 0.04
        self.settings.edge_width = 4.2
        self.settings.chromatic_aberration_amount = 0.45
        self.settings.shimmer_amount = 0.9
        self.settings.shimmer_speed = 1.2
        self.settings.blend_with_original = 1.0
        QMessageBox.information(self, "Preset applied", "Applied recommended 'Invisible' starting values. Re-run preview, then tune Key Threshold/Softness until the Matte view cleanly isolates the subject.")

    def _validate_paths(self) -> tuple[str, str, str] | None:
        bg = self.bg_edit.text().strip()
        fg = self.fg_edit.text().strip()
        out = self.out_edit.text().strip()
        if not bg or not Path(bg).exists():
            QMessageBox.warning(self, "Missing background", "Please select a valid background video file.")
            return None
        if not fg or not Path(fg).exists():
            QMessageBox.warning(self, "Missing foreground", "Please select a valid foreground video file.")
            return None
        if not out:
            QMessageBox.warning(self, "Missing output", "Please choose an output file path.")
            return None
        return bg, fg, out

    def on_preview(self):
        paths = self._validate_paths()
        if not paths:
            return
        bg, fg, _ = paths
        try:
            frame = preview_frame(bg, fg, self.settings)
            self._show_preview(frame)
            self.progress.setValue(100)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Preview failed", str(exc))

    def on_render(self):
        paths = self._validate_paths()
        if not paths:
            return
        bg, fg, out = paths

        self.progress.setValue(0)
        self.render_btn.setEnabled(False)
        self.preview_btn.setEnabled(False)

        self.render_worker = RenderWorker(bg, fg, out, EffectSettings(**asdict(self.settings)))
        self.render_worker.progress.connect(self._on_render_progress)
        self.render_worker.done.connect(self._on_render_done)
        self.render_worker.failed.connect(self._on_render_failed)
        self.render_worker.start()

    def _on_render_progress(self, current: int, total: int):
        if total <= 0:
            self.progress.setValue(0)
            return
        self.progress.setValue(int((current / total) * 100))

    def _on_render_done(self, out_path: str):
        self.render_btn.setEnabled(True)
        self.preview_btn.setEnabled(True)
        QMessageBox.information(self, "Render complete", f"Exported:\n{out_path}")

    def _on_render_failed(self, message: str):
        self.render_btn.setEnabled(True)
        self.preview_btn.setEnabled(True)
        QMessageBox.critical(self, "Render failed", message)

    def _show_preview(self, frame_bgr):
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        h, w = frame_rgb.shape[:2]
        qimg = QImage(frame_rgb.data, w, h, frame_rgb.strides[0], QImage.Format_RGB888)
        pix = QPixmap.fromImage(qimg)
        scaled = pix.scaled(self.preview_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.preview_label.setPixmap(scaled)


def run_app() -> int:
    app = QApplication.instance() or QApplication([])
    win = MainWindow()
    win.show()
    return app.exec()
