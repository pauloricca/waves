#!/usr/bin/env python3
"""
Convert M4A files in the samples directory to WAV files.

The script finds .m4a files that do not already have a .wav file with the
same stem in the same directory, then converts them with ffmpeg.

Depends on ffmpeg, install it with:
    brew install ffmpeg

Usage:
    ./scripts/m4a2wav.py
    ./scripts/m4a2wav.py --dry-run
    ./scripts/m4a2wav.py --samples-dir samples/drums
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


DEFAULT_SAMPLE_RATE = 44100
COMMON_FFMPEG_PATHS = (
    "/opt/homebrew/bin/ffmpeg",
    "/usr/local/bin/ffmpeg",
    "/opt/local/bin/ffmpeg",
    "/usr/bin/ffmpeg",
    "/bin/ffmpeg",
)


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def find_ffmpeg(explicit_path: str | None = None) -> str:
    if explicit_path:
        path = Path(explicit_path).expanduser()
        if path.is_file() and os.access(path, os.X_OK):
            return str(path)
        raise FileNotFoundError(f"ffmpeg not found or not executable: {explicit_path}")

    from_path = shutil.which("ffmpeg")
    if from_path:
        return from_path

    for candidate in COMMON_FFMPEG_PATHS:
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate

    searched = ", ".join(COMMON_FFMPEG_PATHS)
    raise FileNotFoundError(
        "ffmpeg was not found. Install it or pass --ffmpeg /path/to/ffmpeg. "
        f"Checked PATH and: {searched}"
    )


def resolve_samples_dir(path: str) -> Path:
    samples_dir = Path(path).expanduser()
    if not samples_dir.is_absolute():
        samples_dir = project_root() / samples_dir
    return samples_dir.resolve()


def iter_pending_conversions(samples_dir: Path) -> list[tuple[Path, Path]]:
    pending: list[tuple[Path, Path]] = []
    for m4a_path in sorted(samples_dir.rglob("*")):
        if not m4a_path.is_file() or m4a_path.suffix.lower() != ".m4a":
            continue

        wav_path = m4a_path.with_suffix(".wav")
        if wav_path.exists():
            continue

        pending.append((m4a_path, wav_path))

    return pending


def convert_file(
    ffmpeg_path: str,
    input_path: Path,
    output_path: Path,
    sample_rate: int,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    command = [
        ffmpeg_path,
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(input_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-c:a",
        "pcm_s16le",
        "-n",
        str(output_path),
    ]
    subprocess.run(command, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert missing WAV counterparts for M4A files under samples/.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s
  %(prog)s --dry-run
  %(prog)s --samples-dir samples/drums
  %(prog)s --ffmpeg /opt/homebrew/bin/ffmpeg
        """,
    )
    parser.add_argument(
        "--samples-dir",
        default="samples",
        help="Directory to scan recursively (default: samples)",
    )
    parser.add_argument(
        "--ffmpeg",
        default=None,
        help="Path to ffmpeg. If omitted, PATH and common install locations are checked.",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=DEFAULT_SAMPLE_RATE,
        help=f"Output WAV sample rate in Hz (default: {DEFAULT_SAMPLE_RATE})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print files that would be converted without writing WAV files.",
    )
    args = parser.parse_args()

    if args.sample_rate <= 0:
        print("Error: --sample-rate must be positive", file=sys.stderr)
        return 1

    samples_dir = resolve_samples_dir(args.samples_dir)
    if not samples_dir.is_dir():
        print(f"Error: samples directory not found: {samples_dir}", file=sys.stderr)
        return 1

    pending = iter_pending_conversions(samples_dir)
    if not pending:
        print(f"No M4A files need conversion under {samples_dir}")
        return 0

    print(f"Found {len(pending)} M4A file(s) needing WAV conversion under {samples_dir}")
    for input_path, output_path in pending:
        print(f"  {input_path.relative_to(samples_dir)} -> {output_path.relative_to(samples_dir)}")

    if args.dry_run:
        return 0

    try:
        ffmpeg_path = find_ffmpeg(args.ffmpeg)
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"Using ffmpeg: {ffmpeg_path}")
    converted = 0
    failed = 0

    for input_path, output_path in pending:
        try:
            convert_file(ffmpeg_path, input_path, output_path, args.sample_rate)
        except subprocess.CalledProcessError as exc:
            failed += 1
            print(f"Error converting {input_path}: ffmpeg exited with {exc.returncode}", file=sys.stderr)
            continue

        converted += 1
        print(f"Converted {input_path.name} -> {output_path.name}")

    print(f"Done. Converted {converted} file(s), failed {failed}.")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
