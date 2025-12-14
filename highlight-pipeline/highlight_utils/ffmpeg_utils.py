from pathlib import Path
import subprocess, json

def write_concat_list(paths, txt_path: Path):
    txt_path.parent.mkdir(parents=True, exist_ok=True)
    with txt_path.open("w", encoding="utf-8") as f:
        for p in paths:
            ap = str(Path(p).resolve())
            f.write(f"file '{ap}'\n")

def ffmpeg_concat_reencode(list_file: Path, out_file: Path, fps: int | None = None):
    out_file.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y",
        "-f", "concat", "-safe", "0",
        "-i", str(list_file),
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-ar", "48000",
        "-ac", "2",
        "-b:a", "192k",
        "-af", "aresample=async=1",
        "-movflags", "+faststart",
        str(out_file),
    ]
    if fps is not None:
        cmd.insert(-1, "-r")
        cmd.insert(-1, str(int(fps)))
    subprocess.run(cmd, check=True)

def ffmpeg_cut_mux_av(in_video: Path, in_audio: Path, out_file: Path,
                      start_s: float, end_s: float, fps: int | None = None):
    out_file.parent.mkdir(parents=True, exist_ok=True)
    start_s = max(0.0, float(start_s))
    end_s = max(start_s, float(end_s))

    cmd = [
        "ffmpeg", "-y",
        "-ss", f"{start_s:.3f}", "-to", f"{end_s:.3f}", "-i", str(in_video),
        "-ss", f"{start_s:.3f}", "-to", f"{end_s:.3f}", "-i", str(in_audio),
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-ar", "48000",
        "-ac", "2",
        "-b:a", "192k",
        "-af", "aresample=async=1",
        "-shortest",
        "-movflags", "+faststart",
        str(out_file),
    ]
    if fps is not None:
        cmd.insert(-1, "-r")
        cmd.insert(-1, str(int(fps)))
    subprocess.run(cmd, check=True)

def has_audio_stream(path: Path) -> bool:
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "a",
        "-show_entries", "stream=codec_name,channels,sample_rate",
        "-of", "json",
        str(path),
    ]
    out = subprocess.check_output(cmd).decode("utf-8", errors="ignore")
    j = json.loads(out)
    return bool(j.get("streams"))
