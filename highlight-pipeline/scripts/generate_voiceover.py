from __future__ import annotations

import os
os.environ["TRANSFORMERS_NO_TORCHVISION"] = "1"

import json
import re
import time
import shutil
import subprocess
from pathlib import Path
from typing import Dict, Any, Tuple, List

import numpy as np
import requests
from faster_whisper import WhisperModel


# ------------------ ENV / CONSTANTS ------------------

OPENAI_API_KEY   = os.getenv("OPENAI_API_KEY")
ELEVEN_API_KEY   = os.getenv("ELEVEN_API_KEY")
FOOTBALL_API_KEY = os.getenv("FOOTBALL_API_KEY")

ELEVEN_VOICE_ID_OR_NAME = os.getenv("ELEVEN_VOICE", "Bella")

API_BASE_URL  = "https://v3.football.api-sports.io"

WHISPER_MODEL = "medium"
WHISPER_BEAM  = 2
LANG_HINT     = "auto"
WPM           = 145
SCRIPT_STYLE  = "analytical"


# ------------------ UTILS ------------------

def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p

def _check_tool(name: str):
    if shutil.which(name) is None:
        raise RuntimeError(
            f"Required tool '{name}' not found in PATH. "
            f"Install it and ensure it is accessible from the current environment."
        )

def seconds_to_hhmmss(sec: float) -> str:
    m, s = divmod(int(sec), 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"

def safe_get(d, *keys, default=None):
    cur = d
    for k in keys:
        if isinstance(cur, dict) and k in cur:
            cur = cur[k]
        else:
            return default
    return cur


# ------------------ AUDIO EXTRACT + STT ------------------

def extract_audio(in_video: str, out_wav: str, sr: int = 16000) -> None:
    """
    Extract mono WAV using ffmpeg CLI (ffmpeg must be in PATH).
    No ffmpeg-python dependency.
    """
    _check_tool("ffmpeg")
    _check_tool("ffprobe")

    out_path = Path(out_wav)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg", "-y",
        "-i", str(in_video),
        "-vn",
        "-ac", "1",
        "-ar", str(int(sr)),
        "-f", "wav",
        str(out_wav),
    ]
    subprocess.run(cmd, check=True)

def stt_whisper(
    audio_wav: str,
    model_name: str = WHISPER_MODEL,
    beam_size: int = WHISPER_BEAM,
    lang: str = LANG_HINT,
) -> Tuple[List[dict], str]:
    """
    Speech-to-text using faster-whisper.
    """

    model = WhisperModel(model_name, device="cpu")
    opts = {"beam_size": beam_size}
    if lang and lang != "auto":
        opts["language"] = lang

    segments, _ = model.transcribe(audio_wav, vad_filter=True, **opts)
    segs = [{"start": float(s.start), "end": float(s.end), "text": s.text.strip()} for s in segments]
    full_text = " ".join([s["text"] for s in segs])
    return segs, full_text

import subprocess
import numpy as np

def _ffmpeg_load_mono_f32(wav_path: str, sr: int = 16000) -> np.ndarray:
    cmd = [
        "ffmpeg", "-v", "error",
        "-i", str(wav_path),
        "-f", "f32le",
        "-ac", "1",
        "-ar", str(int(sr)),
        "pipe:1",
    ]
    raw = subprocess.check_output(cmd)
    y = np.frombuffer(raw, dtype=np.float32)
    return y

def audio_excitement(wav_path: str, sr: int = 16000):
    y = _ffmpeg_load_mono_f32(wav_path, sr=sr)
    if y.size == 0:
        return {"z": np.array([], dtype=np.float32), "t": np.array([], dtype=np.float32), "z_peak": 0.0, "t_peak": 0.0}

    frame = int(0.05 * sr)   # 50ms
    hop   = int(0.02 * sr)   # 20ms
    frame = max(frame, 1)
    hop = max(hop, 1)

    # pad for safe framing
    if y.size < frame:
        y = np.pad(y, (0, frame - y.size))

    n = 1 + (y.size - frame) // hop
    rms = np.empty(n, dtype=np.float32)
    for i in range(n):
        s = y[i * hop : i * hop + frame]
        rms[i] = np.sqrt(np.mean(s * s) + 1e-12)

    z = (rms - rms.mean()) / (rms.std() + 1e-8)
    t = (np.arange(n, dtype=np.float32) * hop) / float(sr)

    peak_i = int(np.argmax(z)) if z.size else 0
    return {
        "z": z,
        "t": t,
        "z_peak": float(z[peak_i]) if z.size else 0.0,
        "t_peak": float(t[peak_i]) if t.size else 0.0,
    }



# ------------------ API-FOOTBALL HELPERS ------------------

def api_headers():
    return {"x-apisports-key": FOOTBALL_API_KEY}

def api_get(url, params):
    if not FOOTBALL_API_KEY or FOOTBALL_API_KEY.startswith("<<<"):
        return {}
    try:
        r = requests.get(url, headers=api_headers(), params=params, timeout=60)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print("[API-Football] Request error:", e)
        return {}

def api_find_team_id(name: str):
    if not name:
        return None
    data = api_get(f"{API_BASE_URL}/teams", {"search": name})
    for item in data.get("response", []):
        team = safe_get(item, "team", default={})
        if team and name.lower() in (team.get("name", "").lower()):
            return team.get("id")
    return None

def api_find_fixture_id(home: str, away: str, season: int, date_str: str):
    data = api_get(f"{API_BASE_URL}/fixtures", {"date": date_str, "season": season})
    for it in data.get("response", []):
        h = safe_get(it, "teams", "home", "name", default="").lower()
        a = safe_get(it, "teams", "away", "name", default="").lower()
        if (not home or home.lower() in h) and (not away or away.lower() in a):
            return safe_get(it, "fixture", "id")

    if home and away:
        h2h = api_get(f"{API_BASE_URL}/fixtures/headtohead", {"h2h": f"{home}-{away}", "last": 1})
        for it in h2h.get("response", []):
            fid = safe_get(it, "fixture", "id")
            if fid:
                return fid

    return None

def api_team_last_fixtures(team_id: int, last: int = 5):
    if not team_id:
        return {}
    return api_get(f"{API_BASE_URL}/fixtures", {"team": team_id, "last": last})

def api_players_stats(team_id: int, season: int, page: int = 1):
    if not team_id:
        return {}
    return api_get(f"{API_BASE_URL}/players", {"team": team_id, "season": season, "page": page})

def summarize_form(fixtures_json):
    try:
        resp = fixtures_json.get("response", [])
        W = D = L = GF = GA = 0
        for f in resp:
            gh = safe_get(f, "goals", "home", default=0) or 0
            ga = safe_get(f, "goals", "away", default=0) or 0
            home_win = safe_get(f, "teams", "home", "winner", default=None)
            away_win = safe_get(f, "teams", "away", "winner", default=None)

            if home_win is True or away_win is True:
                W += 1
            elif gh == ga:
                D += 1
            else:
                L += 1

            GF += max(gh, ga)
            GA += min(gh, ga)

        return {"W": W, "D": D, "L": L, "GF": GF, "GA": GA, "count": len(resp)}
    except Exception:
        return {"W": 0, "D": 0, "L": 0, "GF": 0, "GA": 0, "count": 0}

def summarize_players(players_json, top_n=3):
    try:
        resp = players_json.get("response", [])
        table = []
        for item in resp:
            player = safe_get(item, "player", default={})
            stats0 = safe_get(item, "statistics", 0, default={})

            goals   = safe_get(stats0, "goals", "total", default=0) or 0
            assists = safe_get(stats0, "goals", "assists", default=0) or 0
            shots   = safe_get(stats0, "shots", "total", default=0) or 0

            name = player.get("name", "Unknown")
            pos  = safe_get(stats0, "games", "position", default="")

            table.append({"name": name, "position": pos, "goals": goals, "assists": assists, "shots": shots})

        table.sort(key=lambda r: (r["goals"], r["assists"], r["shots"]), reverse=True)
        return table[:top_n]
    except Exception:
        return []

def build_match_analytics(home_team, away_team, season, date_str, excitement_peak):
    if not FOOTBALL_API_KEY or (not home_team and not away_team):
        return {"note": "Official match data skipped.", "excitement_peak": excitement_peak}

    home_id = api_find_team_id(home_team) if home_team else None
    away_id = api_find_team_id(away_team) if away_team else None

    fixture_id = api_find_fixture_id(home_team, away_team, season, date_str)

    home_form = summarize_form(api_team_last_fixtures(home_id, last=5)) if home_id else {}
    away_form = summarize_form(api_team_last_fixtures(away_id, last=5)) if away_id else {}

    home_players = summarize_players(api_players_stats(home_id, season)) if home_id else []
    away_players = summarize_players(api_players_stats(away_id, season)) if away_id else []

    h2h = {}
    if home_team and away_team:
        h2h = api_get(f"{API_BASE_URL}/fixtures/headtohead", {"h2h": f"{home_team}-{away_team}", "last": 5})

    return {
        "fixture": {"id": fixture_id, "date": date_str, "home": home_team, "away": away_team},
        "excitement_peak": excitement_peak,
        "home_form": home_form,
        "away_form": away_form,
        "home_key_players": home_players,
        "away_key_players": away_players,
        "h2h_count": len(h2h.get("response", [])) if h2h else 0,
    }


# ------------------ SCRIPT GENERATION ------------------

def fallback_script(transcript_text: str, analytics_json: dict, target_minutes: float, style: str) -> str:
    target_words = int(target_minutes * WPM)
    body = " ".join(transcript_text.split()[:max(120, target_words - 140)])
    return (
        f"# Script (~{target_minutes} min)\n\n"
        f"Intro: quick analytical recap of the match using commentary and basic stats.\n\n"
        f"{body}\n\n"
        f"Outro: momentum swings and key moments shaped the result."
    )

def gpt_generate_script(transcript_text: str, analytics: dict, target_minutes: float, style: str) -> str:
    if not OPENAI_API_KEY or OPENAI_API_KEY.startswith("<<<"):
        return fallback_script(transcript_text, analytics, target_minutes, style)

    try:
        from openai import OpenAI

        client = OpenAI(api_key=OPENAI_API_KEY)
        prompt = f"""
You are a football analyst and scriptwriter.
Your style must be direct, analytical, and concise.
Zero fluff. No filler phrases.

Task:
Create an English voiceover script of about {target_minutes}+2 minutes (≈ {WPM} words per minute).

Internal structure guidance (DO NOT output this structure):
- Begin with 1–2 sentences introducing the match and its significance.
- Briefly analyze pre-match context using ONLY the analytics data.
- Select and analyze 3–5 decisive match moments from the commentary.
- Conclude with 1–2 sharp sentences explaining why the result happened.

IMPORTANT:
- Output ONLY the spoken narration text.
- DO NOT include section titles, labels, headings, or words like "Intro", "Outro", or any numbering.
- The script must be clean, continuous narration suitable for direct voiceover.
- No markdown. No lists. No explanations.

Write for energetic broadcast delivery:
- Prefer short punchy sentences, varied rhythm, and occasional emphatic pauses using dashes (—).
- Avoid monotone listing; highlight momentum swings with sharper phrasing

Use:
- [ANALYTICS JSON] for factual context
- [COMMENTARY TRANSCRIPT] to identify key events

[ANALYTICS JSON]
{json.dumps(analytics, ensure_ascii=False, indent=2)}

[COMMENTARY TRANSCRIPT (trimmed)]
{transcript_text[:3500]}
"""
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a sharp, direct football analyst and scriptwriter. No fluff."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.7,
            max_tokens=1200,
        )
        return resp.choices[0].message.content.strip()

    except Exception as e:
        print("[OpenAI] Fallback (reason:", e, ")")
        return fallback_script(transcript_text, analytics, target_minutes, style)


# ------------------ ELEVENLABS TTS ------------------

_UUID_RE = re.compile(r"^[A-Za-z0-9]{20,}$")

def eleven_resolve_voice_id(candidate: str) -> str:
    if _UUID_RE.match(candidate or ""):
        print(f"[TTS] Using ElevenLabs voice_id: {candidate}")
        return candidate

    if not ELEVEN_API_KEY or ELEVEN_API_KEY.startswith("<<<"):
        print("[TTS] ELEVEN_API_KEY missing; cannot resolve voice name.")
        return candidate

    try:
        r = requests.get(
            "https://api.elevenlabs.io/v1/voices",
            headers={"xi-api-key": ELEVEN_API_KEY},
            timeout=60,
        )
        r.raise_for_status()
        data = r.json()
        voices = data.get("voices", [])
        chosen = None

        for v in voices:
            if v.get("name", "").lower() == (candidate or "").lower():
                chosen = v.get("voice_id")
                break

        if not chosen and voices:
            chosen = voices[0].get("voice_id")
            print(f"[TTS] Voice '{candidate}' not found; using first available voice: {voices[0].get('name')} ({chosen})")
        else:
            print(f"[TTS] Resolved voice '{candidate}' -> {chosen}")

        return chosen or candidate

    except Exception as e:
        print("[TTS] Could not resolve ElevenLabs voice name:", e)
        return candidate

def tts_elevenlabs(text: str, out_mp3: str, voice_id_or_name: str = ELEVEN_VOICE_ID_OR_NAME) -> None:
    if not ELEVEN_API_KEY or ELEVEN_API_KEY.startswith("<<<"):
        raise RuntimeError("ELEVEN_API_KEY is missing.")

    voice_id = eleven_resolve_voice_id(voice_id_or_name)
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"

    r = requests.post(
        url,
        headers={
            "xi-api-key": ELEVEN_API_KEY,
            "accept": "audio/mpeg",
            "Content-Type": "application/json",
        },
        json={
            "text": text,
            "model_id": "eleven_multilingual_v2",
            "voice_settings": {
                "stability": 0.15,
                "similarity_boost": 0.85,
                "style": 0.65,
                "use_speaker_boost": True
            },
        },
        timeout=180,
    )
    r.raise_for_status()
    Path(out_mp3).write_bytes(r.content)


# ------------------ NOTEBOOK PIPELINE ENTRYPOINT ------------------

def create_voiceover_for_match(
    video_path: str,
    home_team: str = "",
    away_team: str = "",
    season: int = 2024,
    date_str: str = "2024-10-26",
    target_min: float = 3.0,
    work_dir: str = "work",
    sr: int = 16000,
    whisper_model: str = WHISPER_MODEL,
    whisper_beam: int = WHISPER_BEAM,
    lang_hint: str = LANG_HINT,
    script_style: str = SCRIPT_STYLE,
) -> Dict[str, Any]:
    """
    Notebook-friendly wrapper:
    - extracts audio
    - STT via faster-whisper
    - computes excitement
    - fetches API-Football analytics (if key exists)
    - generates script (OpenAI if key exists, else fallback)
    - synthesizes TTS via ElevenLabs (requires key)

    Returns dict with paths + artifacts.
    """
    work = ensure_dir(work_dir)

    audio_wav       = str(work / "audio.wav")
    transcript_json = str(work / "transcript.json")
    script_md       = str(work / "script.md")
    voice_mp3       = str(work / "voiceover.mp3")

    t0 = time.perf_counter()
    print("[1/6] Extracting audio…")
    extract_audio(video_path, audio_wav, sr=sr)
    print(f"    Done in {time.perf_counter() - t0:.2f}s")

    t1 = time.perf_counter()
    print("[2/6] Running Whisper STT…")
    segments, transcript_text = stt_whisper(audio_wav, model_name=whisper_model, beam_size=whisper_beam, lang=lang_hint)
    Path(transcript_json).write_text(json.dumps(segments, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"    Segments: {len(segments)} | words: {len(transcript_text.split())} | {time.perf_counter() - t1:.2f}s")

    t2 = time.perf_counter()
    print("[3/6] Computing audio excitement…")
    exc = audio_excitement(audio_wav, sr=sr); print(f"    Excitement z-peak: {exc['z_peak']:.2f} | {time.perf_counter() - t2:.2f}s")

    t3 = time.perf_counter()
    print("[4/6] Fetching match/team analytics (API-Football)…")
    exc_small = {"z_peak": exc["z_peak"], "t_peak": exc["t_peak"]}
    analytics = build_match_analytics(home_team, away_team, season, date_str, excitement_peak=exc_small)
    print(f"    Analytics keys: {list(analytics.keys())} | {time.perf_counter() - t3:.2f}s")

    t4 = time.perf_counter()
    print("[5/6] Generating script (English)…")
    script_text = gpt_generate_script(transcript_text, analytics, target_min, script_style)
    Path(script_md).write_text(script_text, encoding="utf-8")
    print(f"    Script saved -> {script_md} | {time.perf_counter() - t4:.2f}s")

    t5 = time.perf_counter()
    print("[6/6] Synthesizing voiceover (ElevenLabs, MP3)…")
    tts_elevenlabs(script_text, voice_mp3, ELEVEN_VOICE_ID_OR_NAME)
    print(f"    Voiceover saved -> {voice_mp3} | {time.perf_counter() - t5:.2f}s")

    return {
        "audio_wav": audio_wav,
        "transcript_segments": segments,
        "transcript_text": transcript_text,
        "transcript_json": transcript_json,
        "analytics": analytics,
        "excitement_peak": exc,
        "script_text": script_text,
        "script_md": script_md,
        "voice_mp3": voice_mp3,
        "work_dir": str(work),
    }