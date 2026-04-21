# CloakFX (Standalone Desktop App)

CloakFX is a **self-contained Python desktop program** that creates a movie-style **invisible person / cloaked refraction** effect from:

1. A background plate video
2. A foreground/roto subject video shot against a solid-color backdrop

No After Effects/Premiere/OFX plugins are required.

## Features

- PySide6 desktop GUI
- Background/Foreground/Output file pickers
- One-click preview
- Offline render/export with progress bar
- Robust synchronization for videos with mismatched:
  - frame rate
  - resolution
  - duration (processing ends at shorter clip)
- Chroma key pipeline with:
  - color-distance keying in YCrCb chroma space
  - threshold + softness
  - spill suppression
  - matte expand/contract
  - denoise/open-close + feather blur
- Cloak/refraction pipeline:
  - edge mask from matte gradient
  - displacement field from matte gradients + animated shimmer noise
  - stronger distortion on edges and optional interior distortion (supports very large pixel warps)
  - optional blur inside matte
  - optional chromatic aberration/RGB split
  - optional edge highlight
- Debug views:
  - Final Output
  - Foreground
  - Keyed Foreground
  - Matte
  - Edge Mask
  - Displacement Field

---

## Project structure

- `main.py` - GUI entrypoint
- `cloakfx/ui.py` - PySide6 UI, controls, preview, render thread
- `cloakfx/config.py` - settings dataclass and debug modes
- `cloakfx/chroma.py` - key color sampling, keying, spill suppression
- `cloakfx/matte.py` - matte cleanup and edge mask
- `cloakfx/displacement.py` - displacement, remap, chromatic aberration
- `cloakfx/video_io.py` - synced frame loading and video writing
- `cloakfx/renderer.py` - full frame pipeline and export loop
- `cloakfx/cli.py` - optional command-line renderer
- `requirements.txt` - dependencies

---

## Install

### 1) Python
Use Python **3.11**.

### 2) FFmpeg (system)
Install FFmpeg and ensure `ffmpeg` is in PATH.

- macOS: `brew install ffmpeg`
- Ubuntu/Debian: `sudo apt-get install ffmpeg`
- Windows: install from ffmpeg.org and add to PATH

### 3) Python dependencies

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
```

---

## Run GUI

```bash
python main.py
```

### GUI workflow
1. Choose **Background Video**.
2. Choose **Foreground/Roto Video** (subject on solid backdrop).
3. Choose **Output** path.
4. Click **Auto Sample Key Color** (or manually pick key color).
5. Adjust key/matte/displacement controls.
6. Click **Preview Frame** for quick look.
7. Click **Render Export** for full output.

---

## CLI usage (optional)

```bash
python -m cloakfx.cli \
  --background path/to/background.mp4 \
  --foreground path/to/foreground_green.mp4 \
  --output path/to/out.mp4 \
  --displacement 14
```

---

## How the effect works

1. **Chroma keying:** computes chroma distance in YCrCb space between each foreground pixel and selected key color.
2. **Alpha matte:** applies threshold + softness ramp to produce semi-soft alpha.
3. **Matte cleanup:** denoise with morphological operations, optional expand/contract, Gaussian feather.
4. **Edge extraction:** Sobel gradient magnitude of matte gives edge-heavy mask.
5. **Displacement field:** combines matte gradients + animated shimmer noise, weighted higher at edges.
6. **Refraction:** remaps background UV sampling with displacement vectors, using wrap-around borders for extreme warps.
7. **Cinematic polish:** optional blur-in-matte, chromatic aberration, and edge highlight.
8. **Composite:** blend distorted cloak result with original background.

---


## Troubleshooting: subject is still visible

If the person still appears as a normal cutout, it is usually a matte/key issue, not the refraction stage.

1. Click **Invisible Preset** then **Auto Sample Key Color**.
2. Set **Debug View = Matte** and tune:
   - increase **Key Threshold** until the backdrop disappears
   - increase **Key Softness** for smooth transitions
   - adjust **Matte Blur/Feather** and **Denoise Cleanup** to remove speckle
3. Set **Debug View = Edge Mask** and ensure the strongest values are around silhouette edges.
4. Return to **Final Output** and keep:
   - lower **Edge Highlight Amount** (too high makes outline obvious)
   - moderate **Interior Distortion**
   - stronger **Edge Distortion Boost** than interior

Also verify your **background plate does not contain the subject**; otherwise no settings can make them disappear.


For extreme cloak looks, `Displacement Amount` now represents **pixel amplitude** directly and can be pushed into the hundreds; remapping uses wrap-around borders so pixels cycle instead of clamping at edges.

---

## Current limitations

- CPU-only implementation (can be slow at 4K).
- Preview currently processes a single frame on demand, not full realtime playback.
- Audio is not copied to the output yet.
- Auto key color assumes border area mostly contains backdrop.
- Fine hair/transparency handling is first-pass (not deep-learning matting).

---

## Suggested next improvements

1. Add timeline and continuous preview playback.
2. Add FFmpeg mux stage to preserve/copy audio.
3. Add temporal matte stabilization across frames.
4. Add optical-flow driven shimmer/advection for richer heat-haze behavior.
5. Add GPU acceleration path (OpenCV CUDA / Numba / Vulkan / C++ core).
6. Add preset system (Predator, Glass Cloak, Mirage).
