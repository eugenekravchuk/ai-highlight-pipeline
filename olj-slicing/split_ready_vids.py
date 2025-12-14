import subprocess
from pathlib import Path

INPUT_DIR = Path("../data/full_match")
OUT_ROOT  = Path("../data/full_match_processed/")

OUT_AUDIO = OUT_ROOT / "audios"
OUT_VIDEO = OUT_ROOT / "videos_only"

OUT_AUDIO.mkdir(parents=True, exist_ok=True)
OUT_VIDEO.mkdir(parents=True, exist_ok=True)

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}

def run(cmd):
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(p.stderr.strip())

videos = sorted([p for p in INPUT_DIR.iterdir() if p.suffix.lower() in VIDEO_EXTS])
print("Found:", len(videos))

for vp in videos:
    stem = vp.stem

    audio_out = OUT_AUDIO / f"{stem}.mp3"
    video_out = OUT_VIDEO / f"{stem}{vp.suffix}"

    run([
        "ffmpeg", "-y", "-i", str(vp),
        "-vn",
        "-ar", "44100", "-ac", "2", "-b:a", "192k",
        str(audio_out)
    ])

    run([
        "ffmpeg", "-y", "-i", str(vp),
        "-an",
        "-c:v", "copy",
        str(video_out)
    ])

    print("OK:", vp.name)

print("Done.")
