from pydub import AudioSegment
import sys, os, math

import subprocess
from pathlib import Path
from typing import List, Optional
import shutil

def _check_tool(name: str):
    if shutil.which(name) is None:
        raise RuntimeError(f"Required tool '{name}' not found in PATH. Please install it (ffmpeg/ffprobe).")

def _get_duration_seconds(file_path: Path) -> float:
    _check_tool("ffprobe")
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(file_path)
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {proc.stderr.strip()}")
    out = proc.stdout.strip()
    try:
        return float(out)
    except ValueError:
        raise RuntimeError(f"Couldn't parse duration from ffprobe output: {out!r}")

def split_mp4(file_path: str, n_parts: int, output_dir: Optional[str] = None) -> List[str]:
    if n_parts < 1:
        raise ValueError("n_parts must be >= 1")
    inp = Path(file_path)
    if not inp.exists():
        raise ValueError(f"Input file does not exist: {file_path}")
    out_dir = Path(output_dir) if output_dir is not None else inp.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    _check_tool("ffmpeg")
    _check_tool("ffprobe")
    total_dur = _get_duration_seconds(inp)
    if total_dur <= 0:
        raise RuntimeError(f"Invalid duration ({total_dur}) for file {file_path}")
    part_len = total_dur / n_parts
    outputs: List[str] = []
    for i in range(n_parts):
        start = part_len * i
        if i == n_parts - 1:
            t = total_dur - start
        else:
            t = part_len
        out_name = f"{inp.stem}_part{i+1}{inp.suffix}"
        out_path = out_dir / out_name
        cmd = [
            "ffmpeg", "-y",
            "-ss", f"{start:.6f}",
            "-i", str(inp),
            "-t", f"{t:.6f}",
            "-c", "copy",
            "-avoid_negative_ts", "1",
            str(out_path)
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(f"ffmpeg failed while creating part {i+1}: {proc.stderr.strip()}")
        outputs.append(str(out_path))
    return outputs

def split_mp4_audio(
    file_path: str,
    n_parts: int,
    output_dir: Optional[str] = None,
    audio_format: str = "mp3",
):
    if n_parts < 1:
        raise ValueError("n_parts must be >= 1")

    inp = Path(file_path)
    if not inp.exists():
        raise ValueError(f"Input file does not exist: {file_path}")

    audio = AudioSegment.from_file(str(inp), format=inp.suffix.lstrip("."))
    duration_ms = len(audio)

    part_length = math.ceil(duration_ms / n_parts)
    base_name = inp.stem
    out_dir = Path(output_dir) if output_dir is not None else inp.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    outputs = []
    for i in range(n_parts):
        start = i * part_length
        end   = min((i + 1) * part_length, duration_ms)
        if start >= end:
            break

        chunk = audio[start:end]
        out_name = f"{base_name}_part{i+1}.{audio_format}"
        out_path = out_dir / out_name
        export_kwargs = {"format": audio_format}
        if audio_format == "mp3":
            export_kwargs["bitrate"] = "192k"

        chunk.export(str(out_path), **export_kwargs)
        print(f"→ {out_path} ({(end - start) / 1000:.2f} sec)")
        outputs.append(str(out_path))

    return outputs

def split_mp3(file_path: str, n_parts: int, output_dir: str | None = None):

    audio = AudioSegment.from_file(file_path, format="mp3")
    duration_ms = len(audio)

    part_length = math.ceil(duration_ms / n_parts)
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    output_dir = output_dir or os.path.dirname(file_path) or "."

    os.makedirs(output_dir, exist_ok=True)

    for i in range(n_parts):
        start = i * part_length
        end = min((i + 1) * part_length, duration_ms)
        chunk = audio[start:end]
        out_path = os.path.join(output_dir, f"{base_name}_part{i+1}.mp3")
        chunk.export(out_path, format="mp3", bitrate="192k")
        print(f"→ {out_path} ({(end - start) / 1000:.2f} сек)")
