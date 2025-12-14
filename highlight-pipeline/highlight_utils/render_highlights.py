from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import shutil
import numpy as np
from moviepy import VideoFileClip

from .ranges import build_highlight_ranges_from_scores
from .ffmpeg_utils import (
    ffmpeg_cut_mux_av,
    ffmpeg_concat_reencode,
    write_concat_list,
)

def _safe_relpath(in_path: Path) -> Path:
    try:
        return in_path.resolve().relative_to(Path.cwd().resolve())
    except Exception:
        return Path(in_path.name)

def _overlaps(a, b, c, d, gap=0.0):
    return not (b <= c - gap or d <= a - gap)

def _range_score(probs, seg_dur, a, b):
    i0 = int(max(0, np.floor(a / seg_dur)))
    i1 = int(min(len(probs), np.ceil(b / seg_dur)))
    if i1 <= i0:
        return -1e9
    return float(np.max(probs[i0:i1]))

@dataclass
class RenderConfig:
    global_budget_s: float = 240.0
    per_video_candidate_budget_s_frac: float = 0.5
    per_video_candidate_budget_s_cap: float = 900.0
    pre_roll_s: float = 0.0

    smooth_win: int = 7
    min_sep_s: float = 35.0
    pre_s: float = 5.0
    post_s: float = 5.0
    merge_gap_s: float = 2.0
    score_floor_q: float = 0.90
    exclude_start_s: float = 0.0
    exclude_end_s: float = 0.0

    min_gap_within_video_s: float = 2.0

    final_name: str = "all_highlights_4min.mp4"
    force_fps_final: int = 30

def build_candidates(video_paths, pred_probs_per_video):
    candidates = []
    video_meta = []

    for vid, (vpath, probs) in enumerate(zip(video_paths, pred_probs_per_video)):
        vpath = Path(vpath)
        probs = np.asarray(probs, dtype=np.float32).reshape(-1)
        T = len(probs)

        if T == 0:
            video_meta.append((0.0, 30, 1.0))
            continue

        with VideoFileClip(str(vpath)) as clip:
            clip_dur = float(clip.duration or 0.0)
            fps = int(getattr(clip, "fps", None) or 30)

        if clip_dur <= 0:
            video_meta.append((0.0, fps, 1.0))
            continue

        seg_dur = clip_dur / T
        video_meta.append((clip_dur, fps, seg_dur))

    return candidates, video_meta

def render_global_highlights(
    video_paths,
    audio_paths,
    pred_probs_per_video,
    out_root: str | Path = "highlights_out",
    tmp_root: str | Path | None = None,
    cfg: RenderConfig = RenderConfig(),
):
    out_root = Path(out_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    if tmp_root is None:
        tmp_root = out_root / "_tmp_segments"
    tmp_root = Path(tmp_root).resolve()
    tmp_root.mkdir(parents=True, exist_ok=True)

    candidates = []
    video_meta = []

    for vid, (vpath, probs) in enumerate(zip(video_paths, pred_probs_per_video)):
        vpath = Path(vpath)
        probs = np.asarray(probs, dtype=np.float32).reshape(-1)
        T = len(probs)
        if T == 0:
            video_meta.append((0.0, 30, 1.0))
            continue

        with VideoFileClip(str(vpath)) as clip:
            clip_dur = float(clip.duration or 0.0)
            fps = int(getattr(clip, "fps", None) or 30)

        if clip_dur <= 0:
            video_meta.append((0.0, fps, 1.0))
            continue

        seg_dur = clip_dur / T
        video_meta.append((clip_dur, fps, seg_dur))

        big_budget = min(cfg.per_video_candidate_budget_s_cap, cfg.per_video_candidate_budget_s_frac * clip_dur)

        ranges = build_highlight_ranges_from_scores(
            scores=probs,
            seg_dur=seg_dur,
            clip_dur=clip_dur,
            budget_s=big_budget,
            smooth_win=cfg.smooth_win,
            min_sep_s=cfg.min_sep_s,
            pre_s=cfg.pre_s,
            post_s=cfg.post_s,
            merge_gap_s=cfg.merge_gap_s,
            exclude_start_s=cfg.exclude_start_s,
            exclude_end_s=cfg.exclude_end_s,
            score_floor_q=cfg.score_floor_q,
        )

        for (a, b) in ranges:
            dur = float(b - a)
            if dur <= 0:
                continue
            sc = _range_score(probs, seg_dur, a, b)
            candidates.append(
                {"vid": vid, "a": float(a), "b": float(b), "dur": dur, "score": sc, "vpath": vpath}
            )

    remaining = float(cfg.global_budget_s)
    candidates.sort(key=lambda x: x["score"], reverse=True)

    selected = []
    selected_by_vid = {}

    for cand in candidates:
        if remaining <= 0:
            break

        vid = cand["vid"]
        a, b = cand["a"], cand["b"]
        dur = cand["dur"]


        ok = True
        for (aa, bb) in selected_by_vid.get(vid, []):
            if _overlaps(a, b, aa, bb, gap=cfg.min_gap_within_video_s):
                ok = False
                break
        if not ok:
            continue

        take_dur = min(dur, remaining)
        if take_dur <= 1.0:
            continue

        b_take = a + take_dur
        selected.append({**cand, "b": b_take, "dur": take_dur})
        selected_by_vid.setdefault(vid, []).append((a, b_take))
        remaining -= take_dur

    per_video_outputs = []
    selected = sorted(selected, key=lambda x: (x["vid"], x["a"]))

    sel_group = {}
    for s in selected:
        sel_group.setdefault(s["vid"], []).append((s["a"], s["b"]))

    for vid in sorted(sel_group.keys()):
        ranges = sorted(sel_group[vid], key=lambda x: x[0])

        vpath = Path(video_paths[vid])
        apath = Path(audio_paths[vid])
        clip_dur, fps, _seg_dur = video_meta[vid]
        if clip_dur <= 0:
            continue

        rel = _safe_relpath(vpath)
        out_dir = (out_root / rel.parent).resolve()
        out_dir.mkdir(parents=True, exist_ok=True)

        base_name = vpath.stem
        per_video_out = out_dir / f"{base_name}_picked.mp4"
        per_video_tmp_dir = tmp_root / rel.parent / base_name
        per_video_tmp_dir.mkdir(parents=True, exist_ok=True)

        segment_files = []
        for k, (start_t, end_t) in enumerate(ranges):
            pre_start = max(0.0, float(start_t) - cfg.pre_roll_s)
            end_t = min(float(clip_dur), float(end_t))
            if end_t - pre_start < 1.0:
                continue

            seg_out = per_video_tmp_dir / f"seg_{k:04d}.mp4"
            ffmpeg_cut_mux_av(
                in_video=vpath,
                in_audio=apath,
                out_file=seg_out,
                start_s=pre_start,
                end_s=end_t,
                fps=fps,
            )
            segment_files.append(seg_out)

        if not segment_files:
            shutil.rmtree(per_video_tmp_dir, ignore_errors=True)
            continue

        list_file = per_video_tmp_dir / "concat.txt"
        write_concat_list(segment_files, list_file)
        ffmpeg_concat_reencode(list_file, per_video_out, fps=fps)

        per_video_outputs.append(per_video_out)
        shutil.rmtree(per_video_tmp_dir, ignore_errors=True)

    if per_video_outputs:
        final_out = out_root / cfg.final_name
        final_list = tmp_root / "all_concat.txt"
        write_concat_list(per_video_outputs, final_list)
        ffmpeg_concat_reencode(final_list, final_out, fps=cfg.force_fps_final)
        return final_out, per_video_outputs

    return None, []
