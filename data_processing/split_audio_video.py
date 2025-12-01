"""Utilities for splitting audio (and optionally video) into matched clips."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Dict, List, Optional

from pydub import AudioSegment

try:
    from moviepy import VideoFileClip
except Exception:  # pragma: no cover - optional dependency
    VideoFileClip = None


def split_audio(
    file_path: str,
    n_parts: int,
    output_dir: Optional[str] = None,
    *,
    audio_format: Optional[str] = None,
) -> List[Dict]:
    """Split an audio file into ``n_parts`` roughly-equal clips."""

    audio = AudioSegment.from_file(file_path, format=audio_format)
    duration_ms = len(audio)

    if n_parts <= 0:
        raise ValueError("n_parts must be positive")

    part_length = math.ceil(duration_ms / n_parts)
    base_name = Path(file_path).stem
    output = Path(output_dir or Path(file_path).parent or ".")
    output.mkdir(parents=True, exist_ok=True)

    manifest: List[Dict] = []
    for i in range(n_parts):
        start = i * part_length
        end = min((i + 1) * part_length, duration_ms)
        chunk = audio[start:end]
        out_path = output / f"{base_name}_part{i+1:03d}.mp3"
        chunk.export(out_path, format="mp3", bitrate="192k")
        entry = {
            "index": i,
            "start_ms": start,
            "end_ms": end,
            "path": str(out_path),
        }
        manifest.append(entry)
        print(f"→ Audio {out_path} ({(end - start) / 1000:.2f}s)")

    return manifest


def split_video(video_path, segments, output_dir):
    """
    Splits a video file into 'segments' equal parts.
    Keeps the source file open during the loop to prevent FFMPEG pipe errors.
    """
    video_path = str(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    manifest_entries = []

    # Context manager ensures the file stays open while subclips are processed
    with VideoFileClip(video_path) as video:
        duration = video.duration
        segment_duration = duration / segments
        
        print(f"Processing Video: {video_path} ({duration:.2f}s total)")

        for i in range(segments):
            start = i * segment_duration
            # Force the last segment to go exactly to the end to avoid rounding gaps
            end = (i + 1) * segment_duration if i < segments - 1 else duration
            
            clip_filename = f"video_part_{i:03d}.mp4"
            out_path = output_dir / clip_filename
            
            # Create the subclip referencing the OPEN video object
            sub = video.subclipped(start, end)
            
            # Write immediately while parent is open
            sub.write_videofile(
                str(out_path),
                codec="libx264",
                audio_codec="aac",
                logger=None,   # Set to 'bar' if you want a progress bar
                threads=4      # Optional: speed up encoding
            )
            
            manifest_entries.append({
                "index": i,
                "file": str(out_path),
                "start": start,
                "end": end
            })
            print(f"✓ Saved {clip_filename}")

    return manifest_entries

def main() -> None:
    parser = argparse.ArgumentParser(description="Split audio (and optional video) into matched clips")
    
    # CHANGE 1: Remove required=True and add default=None
    parser.add_argument("--audio", default=None, help="Path to source audio file (mp3/wav/etc)")
    parser.add_argument("--segments", type=int, default=40, help="Number of clips to generate")
    parser.add_argument("--audio-out", default="architecture/audios", help="Directory to store audio clips")
    parser.add_argument("--video", default=None, help="Optional path to matching video file")
    parser.add_argument("--video-out", default="videos_pipeline/clips", help="Directory to store video clips")
    parser.add_argument("--manifest", default="output/split_manifest.json", help="Where to save the JSON manifest")
    args = parser.parse_args()

    # CHANGE 2: Ensure at least one input is provided
    if not args.audio and not args.video:
        parser.error("You must provide at least --audio or --video")

    # CHANGE 3: Only run split_audio if args.audio is present
    audio_manifest = None
    if args.audio:
        try:
            audio_manifest = split_audio(args.audio, args.segments, args.audio_out)
        except Exception as e:
            print(f"[ERROR] Audio split failed: {e}")

    video_manifest = None
    if args.video:
        try:
            # You might need to import split_video if it's not already imported
            video_manifest = split_video(args.video, args.segments, args.video_out)
        except ImportError as exc:
            print(f"[WARN] Video split skipped: {exc}")

    manifest = {
        "audio_source": args.audio,
        "video_source": args.video,
        "segments": args.segments,
        "audio_clips": audio_manifest,
        "video_clips": video_manifest,
    }

    manifest_path = Path(args.manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"✓ Manifest saved to {manifest_path}")

if __name__ == "__main__":
    main()