from __future__ import annotations

import argparse

from .config import EffectSettings
from .renderer import render_video


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="CloakFX video renderer")
    p.add_argument("--background", required=True, help="Background video path")
    p.add_argument("--foreground", required=True, help="Foreground/roto video path")
    p.add_argument("--output", required=True, help="Output video path")
    p.add_argument("--displacement", type=float, default=13.0)
    p.add_argument("--debug-view", default="Final Output")
    return p


def main() -> int:
    args = build_parser().parse_args()
    settings = EffectSettings(displacement_amount=args.displacement, debug_view=args.debug_view)

    def progress(cur: int, total: int):
        pct = (cur / total) * 100 if total else 0
        print(f"\rRendering {cur}/{total} ({pct:.1f}%)", end="", flush=True)

    render_video(args.background, args.foreground, args.output, settings, progress_cb=progress)
    print("\nDone")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
