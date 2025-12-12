from pathlib import Path
from moviepy import VideoFileClip, TextClip, CompositeVideoClip, ColorClip
# v2 exposes concatenate_videoclips via the compositing module; also often available at top-level.
try:
    from moviepy import concatenate_videoclips  # works on many v2 builds
except Exception:
    from moviepy.video.compositing.CompositeVideoClip import concatenate_videoclips

# v2 Effects are classes, applied via clip.with_effects([...])
from moviepy.video.fx import CrossFadeIn
from moviepy.audio.fx import AudioFadeIn, AudioFadeOut

import textwrap

# ---------------------------
# Config
# ---------------------------
DOWNLOADS = Path("downloads")      # folder where yt-dlp put the mp4 files
OUT_PATH = "basketball_highlights.mp4"
FRAME_W, FRAME_H = 1920, 1080
PER_CLIP_MAX = 25.0
CROSSFADE = 0.8
FPS = 30

# OPTIONAL: set a font file path for TextClip (recommended on Windows)
# Leave as None to let Pillow choose a default.
FONT_PATH = None  # e.g., r"C:\Windows\Fonts\arial.ttf"

# ---------------------------
# Helpers
# ---------------------------
def fit_letterbox(clip, w=FRAME_W, h=FRAME_H, bg_color=(0, 0, 0)):
    # v2: use .resized() and .with_position() / .with_duration()
    if clip.w / clip.h >= w / h:
        clip = clip.resized(height=h)
    else:
        clip = clip.resized(width=w)

    bg = ColorClip(size=(w, h), color=bg_color).with_duration(clip.duration)
    return CompositeVideoClip([bg, clip.with_position("center")]).with_duration(clip.duration)

def caption_clip(text, dur=3.5, fontsize=48):
    if not text:
        return None
    # In v2, TextClip accepts font path and uses Pillow (no ImageMagick).
    # method='caption' still works; when using caption, supply size=(width, height or None)
    wrapped = "\n".join(textwrap.wrap(text, width=50))
    return (TextClip(
                text=wrapped,
                font=FONT_PATH,           # None = Pillow default font
                font_size=fontsize,
                color="white",
                stroke_color="black",
                stroke_width=2,
                method="caption",
                size=(int(FRAME_W * 0.9), None),
                transparent=True,
                duration=dur
            )
            .with_position(("center", int(FRAME_H * 0.86))))

def make_per_clip(clip_path: Path, title: str) -> CompositeVideoClip:
    raw = VideoFileClip(str(clip_path))
    base = raw.subclipped(0, PER_CLIP_MAX) if (PER_CLIP_MAX and raw.duration and raw.duration > PER_CLIP_MAX) else raw
    base = fit_letterbox(base, FRAME_W, FRAME_H)
    cap = caption_clip(title, dur=3.5, fontsize=50)

    overlays = [base]
    if cap:
        overlays.append(cap)  # cap already has duration & position

    clip = CompositeVideoClip(overlays).with_duration(base.duration)
    # v2 audio fades are effects:
    clip = clip.with_effects([AudioFadeIn(0.4), AudioFadeOut(0.4)])
    return clip

# ---------------------------
# Main
# ---------------------------
def build_highlight_reel(folder=DOWNLOADS, out_path=OUT_PATH):
    folder.mkdir(parents=True, exist_ok=True)

    clips = []
    for p in sorted(folder.glob("*.mp4")):
        if not p.is_file():
            continue
        title = p.stem  # filename without extension
        clips.append(make_per_clip(p, title))

    if not clips:
        raise RuntimeError(f"No .mp4 files found in {folder.resolve()}")

    # Apply crossfade effects to all but the first
    for i in range(1, len(clips)):
        clips[i] = clips[i].with_effects([CrossFadeIn(CROSSFADE), AudioFadeIn(CROSSFADE)])

    # Use negative padding to overlap clips by CROSSFADE seconds
    final = concatenate_videoclips(clips, method="compose", padding=-CROSSFADE).with_fps(FPS)
    final.write_videofile(out_path, codec="libx264", audio_codec="aac", fps=FPS)
    return Path(out_path)

if __name__ == "__main__":
    result = build_highlight_reel()
    print("Saved stitched video:", result)
 