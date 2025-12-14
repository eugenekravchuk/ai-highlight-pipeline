from __future__ import annotations

import math
import os
import shutil
import subprocess
from pathlib import Path
from typing import List, Optional

from pydub import AudioSegment


def check_tool(name: str):
    if shutil.which(name) is None:
        raise RuntimeError(f"Required tool '{name}' not found in PATH. Install ffmpeg/ffprobe.")


def get_duration_seconds(file_path: Path) -> float:
    check_tool("ffprobe")
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(file_path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {proc.stderr.strip()}")
    out = proc.stdout.strip()
    try:
        return float(out)
    except ValueError:
        raise RuntimeError(f"Couldn't parse duration from ffprobe output: {out!r}")


# -------------------- split by N parts --------------------

def split_mp4_equal_parts(file_path: str, n_parts: int, output_dir: Optional[str] = None) -> List[str]:
    """
    Split MP4 into N equal parts (stream-copy). Parts may not be exact 20.000s due to keyframes.
    """
    if n_parts < 1:
        raise ValueError("n_parts must be >= 1")
    inp = Path(file_path)
    if not inp.exists():
        raise FileNotFoundError(inp)

    out_dir_p = Path(output_dir) if output_dir is not None else inp.parent
    out_dir_p.mkdir(parents=True, exist_ok=True)

    check_tool("ffmpeg")
    total_dur = get_duration_seconds(inp)
    if total_dur <= 0:
        raise RuntimeError(f"Invalid duration ({total_dur}) for file {file_path}")

    part_len = total_dur / n_parts
    outputs: List[str] = []

    for i in range(n_parts):
        start = part_len * i
        t = (total_dur - start) if (i == n_parts - 1) else part_len

        out_path = out_dir_p / f"{inp.stem}_part{i+1}{inp.suffix}"
        cmd = [
            "ffmpeg", "-y",
            "-ss", f"{start:.6f}",
            "-i", str(inp),
            "-t", f"{t:.6f}",
            "-c", "copy",
            "-avoid_negative_ts", "1",
            str(out_path),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(f"ffmpeg failed while creating part {i+1}: {proc.stderr.strip()}")
        outputs.append(str(out_path))

    return outputs


def split_audio_equal_parts_pydub(
    file_path: str,
    n_parts: int,
    output_dir: Optional[str] = None,
    audio_format: str = "mp3",
    bitrate: str = "192k",
) -> List[str]:
    """
    Split an audio file into N equal parts using pydub.
    """
    if n_parts < 1:
        raise ValueError("n_parts must be >= 1")
    inp = Path(file_path)
    if not inp.exists():
        raise FileNotFoundError(inp)

    out_dir_p = Path(output_dir) if output_dir is not None else inp.parent
    out_dir_p.mkdir(parents=True, exist_ok=True)

    audio = AudioSegment.from_file(str(inp), format=inp.suffix.lstrip("."))
    duration_ms = len(audio)
    part_length = math.ceil(duration_ms / n_parts)

    outputs: List[str] = []
    for i in range(n_parts):
        start = i * part_length
        end = min((i + 1) * part_length, duration_ms)
        if start >= end:
            break

        chunk = audio[start:end]
        out_path = out_dir_p / f"{inp.stem}_part{i+1}.{audio_format}"

        export_kwargs = {"format": audio_format}
        if audio_format == "mp3":
            export_kwargs["bitrate"] = bitrate

        chunk.export(str(out_path), **export_kwargs)
        outputs.append(str(out_path))

    return outputs


# -------------------- split by fixed segment length --------------------

def split_audio_fixed_segments_ffmpeg(
    audio_path: str,
    out_dir: str | Path,
    seg_s: int = 20,
    bitrate: str = "192k",
) -> List[Path]:
    """
    Split audio into fixed seg_s chunks using ffmpeg.
    Outputs: audio_00000.mp3, audio_00001.mp3, ...
    """
    check_tool("ffmpeg")

    inp = Path(audio_path)
    if not inp.exists():
        raise FileNotFoundError(inp)

    out_dir_p = Path(out_dir)
    out_dir_p.mkdir(parents=True, exist_ok=True)
    pattern = str(out_dir_p / "audio_%05d.mp3")

    cmd = [
        "ffmpeg", "-y",
        "-i", str(inp),
        "-c:a", "libmp3lame",
        "-b:a", str(bitrate),
        "-f", "segment",
        "-segment_time", str(int(seg_s)),
        "-reset_timestamps", "1",
        pattern,
    ]
    subprocess.run(cmd, check=True)
    return sorted(out_dir_p.glob("audio_*.mp3"))


def split_match_to_fixed_segments(
    video_path: str,
    out_root: str | Path = "chunks_out",
    seg_s: int = 20,
    audio_bitrate: str = "192k",
) -> tuple[Path, Path]:
    """
    Create two folders:
      - video_chunks: seg_s MP4 clips (NO audio)
      - audio_chunks: seg_s MP3 clips (audio only)
    """
    check_tool("ffmpeg")

    inp = Path(video_path)
    if not inp.exists():
        raise FileNotFoundError(inp)

    out_root_p = Path(out_root)
    vdir = out_root_p / "video_chunks"
    adir = out_root_p / "audio_chunks"
    vdir.mkdir(parents=True, exist_ok=True)
    adir.mkdir(parents=True, exist_ok=True)

    # Video-only segments (re-encode for exact boundaries)
    v_pattern = str(vdir / "video_%05d.mp4")
    cmd_v = [
        "ffmpeg", "-y",
        "-i", str(inp),
        "-map", "0:v:0",
        "-an",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-force_key_frames", f"expr:gte(t,n_forced*{seg_s})",
        "-f", "segment",
        "-segment_time", str(int(seg_s)),
        "-reset_timestamps", "1",
        v_pattern,
    ]
    subprocess.run(cmd_v, check=True)

    # Audio-only segments
    a_pattern = str(adir / "audio_%05d.mp3")
    cmd_a = [
        "ffmpeg", "-y",
        "-i", str(inp),
        "-map", "0:a:0",
        "-vn",
        "-c:a", "libmp3lame",
        "-b:a", str(audio_bitrate),
        "-ar", "48000",
        "-ac", "2",
        "-f", "segment",
        "-segment_time", str(int(seg_s)),
        "-reset_timestamps", "1",
        a_pattern,
    ]
    subprocess.run(cmd_a, check=True)

    return vdir, adir
