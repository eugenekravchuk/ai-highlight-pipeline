# run.py
import os
os.environ["TRANSFORMERS_NO_TORCHVISION"] = "1"

import sys
import json
import time
import shutil
from pathlib import Path
import subprocess
import re
import ffmpeg
import librosa
import numpy as np
import requests

OPENAI_API_KEY   = os.getenv("OPENAI_API_KEY")
ELEVEN_API_KEY   = os.getenv("ELEVEN_API_KEY")
FOOTBALL_API_KEY = os.getenv("FOOTBALL_API_KEY")

ELEVEN_VOICE_ID_OR_NAME = os.getenv("ELEVEN_VOICE", "Bella")

API_BASE_URL     = "https://v3.football.api-sports.io"

WHISPER_MODEL    = "medium"
WHISPER_BEAM     = 2
LANG_HINT        = "auto"
WPM              = 145
SCRIPT_STYLE     = "analytical"

FFMPEG_CMD = shutil.which("ffmpeg") or r"C:\Users\user\AppData\Local\Microsoft\WinGet\Links\ffmpeg.exe"

def ensure_work():
    Path("work").mkdir(exist_ok=True)

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

def extract_audio(in_video: str, out_wav: str, sr: int = 16000, ffmpeg_cmd: str = FFMPEG_CMD):
    exe_ok = (ffmpeg_cmd and Path(ffmpeg_cmd).exists()) or (shutil.which("ffmpeg") is not None)
    if not exe_ok:
        raise FileNotFoundError("ffmpeg not found. Install ffmpeg or set FFMPEG_CMD to its full path.")
    (
        ffmpeg
        .input(in_video)
        .output(out_wav, ac=1, ar=sr, vn=None, loglevel="error")
        .overwrite_output()
        .run(cmd=ffmpeg_cmd)
    )

def stt_whisper(audio_wav: str, model_name: str = WHISPER_MODEL, beam_size: int = WHISPER_BEAM, lang: str = LANG_HINT):
    from faster_whisper import WhisperModel
    model = WhisperModel(model_name, device="cpu")
    opts = {"beam_size": beam_size}
    if lang and lang != "auto":
        opts["language"] = lang
    segments, _ = model.transcribe(audio_wav, vad_filter=True, **opts)
    segs = [{"start": float(s.start), "end": float(s.end), "text": s.text.strip()} for s in segments]
    full_text = " ".join([s["text"] for s in segs])
    return segs, full_text

def audio_excitement(wav_path: str, sr: int = 16000) -> float:
    y, _ = librosa.load(wav_path, sr=sr, mono=True)
    if y.size == 0:
        return 0.0
    hop = int(0.5 * sr); win = hop * 2
    rms = librosa.feature.rms(y=y, frame_length=win, hop_length=hop, center=True).flatten()
    if len(rms) == 0:
        return 0.0
    z = (rms - rms.mean()) / (rms.std() + 1e-8)
    return float(np.max(z))

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
        if team and name.lower() in (team.get("name","").lower()):
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
            GF += max(gh, ga); GA += min(gh, ga)
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
            name    = player.get("name", "Unknown")
            pos     = safe_get(stats0, "games", "position", default="")
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
        "h2h_count": len(h2h.get("response", [])) if h2h else 0
    }

def fallback_script(transcript_text: str, analytics_json: dict, target_minutes: float, style: str) -> str:
    target_words = int(target_minutes * WPM)
    body = " ".join(transcript_text.split()[:max(120, target_words-140)])
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
You are a football analyst and scriptwriter. Your style must be direct, analytical, and contain zero fluff.
Use concise, clear language. No filler words.

Task: Create an English voiceover script of about {target_minutes} minutes (target ~{WPM} wpm).
Use the [ANALYTICS JSON] for pre-match context and the [COMMENTARY TRANSCRIPT] to identify key match events.

Script Structure:

1.  **Intro (1-2 sentences):**
    * State the match, the final score (if clear from transcript), and its significance.

2.  **Pre-Match Analysis (Concise):**
    * Analyze data *only* from the [ANALYTICS JSON].
    * Compare the recent form (W/D/L, GF/GA) of both teams.
    * Identify the key players (top scorers/assisters) for each side.
    * Mention head-to-head context if available.

3.  **Key Highlight Analysis (The Body):**
    * Scan the [COMMENTARY TRANSCRIPT] and extract the *most important* match events.
    * Focus *only* on:
        * **Goals:** Who scored, when (if mentioned), and the impact.
        * **Red Cards:** Describe the incident and its consequences.
        * **Decisive Moments:** Major saves, penalty incidents, or clear turning points.
    * Do NOT narrate the entire match. Select 3-5 key highlights and analyze *why* they mattered.

4.  **Outro (1-2 sentences):**
    * A sharp conclusion on the decisive factors that led to the result.

[OFFICIAL/ANALYTICS JSON]
{json.dumps(analytics, ensure_ascii=False, indent=2)}

[COMMENTARY TRANSCRIPT (trimmed)]
{transcript_text[:3500]}
"""
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a sharp, direct football analyst and scriptwriter. No fluff."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=1200
        )
        return resp.choices[0].message.content.strip()

    except Exception as e:
        print("[OpenAI] Fallback (reason:", e, ")")
        return fallback_script(transcript_text, analytics, target_minutes, style)

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
            timeout=60
        )
        r.raise_for_status()
        data = r.json()
        voices = data.get("voices", [])
        chosen = None
        for v in voices:
            if v.get("name","").lower() == (candidate or "").lower():
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
            "Content-Type": "application/json"
        },
        json={
            "text": text,
            "model_id": "eleven_multilingual_v2",
            "voice_settings": {"stability": 0.3, "similarity_boost": 0.75}
        },
        timeout=180
    )
    r.raise_for_status()
    Path(out_mp3).write_bytes(r.content)

def main():
    if len(sys.argv) < 3:
        print("Usage:\n  python run.py video.mp4 \"Home Team\" \"Away Team\" 2024 2024-10-26 3")
        sys.exit(1)

    video_path   = sys.argv[1]
    home_team    = sys.argv[2] if len(sys.argv) >= 3 else ""
    away_team    = sys.argv[3] if len(sys.argv) >= 4 else ""
    season       = int(sys.argv[4]) if len(sys.argv) >= 5 and sys.argv[4] else 2024
    date_str     = sys.argv[5] if len(sys.argv) >= 6 else "2024-10-26"
    target_min   = float(sys.argv[6]) if len(sys.argv) >= 7 else 3.0

    ensure_work()
    audio_wav        = "work/audio.wav"
    transcript_json  = "work/transcript.json"
    script_md        = "work/script.md"
    voice_mp3        = "work/voiceover.mp3"

    t0 = time.perf_counter()
    print("[1/6] Extracting audio…")
    extract_audio(video_path, audio_wav, sr=16000, ffmpeg_cmd=FFMPEG_CMD)
    print(f"    Done in {time.perf_counter()-t0:.2f}s")

    t1 = time.perf_counter()
    print("[2/6] Running Whisper STT…")
    segments, transcript_text = stt_whisper(audio_wav, WHISPER_MODEL, WHISPER_BEAM, LANG_HINT)
    Path(transcript_json).write_text(json.dumps(segments, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"    Segments: {len(segments)} | words: {len(transcript_text.split())} | {time.perf_counter()-t1:.2f}s")

    t2 = time.perf_counter()
    print("[3/6] Computing audio excitement…")
    exc = audio_excitement(audio_wav)
    print(f"    Excitement z-peak: {exc:.2f} | {time.perf_counter()-t2:.2f}s")

    t3 = time.perf_counter()
    print("[4/6] Fetching match/team analytics (API-Football)…")
    analytics = build_match_analytics(home_team, away_team, season, date_str, excitement_peak=exc)
    print(f"    Analytics keys: {list(analytics.keys())} | {time.perf_counter()-t3:.2f}s")

    t4 = time.perf_counter()
    print("[5/6] Generating script (English)…")
    script_text = gpt_generate_script(transcript_text, analytics, target_min, SCRIPT_STYLE)
    Path(script_md).write_text(script_text, encoding="utf-8")
    print(f"    Script saved -> {script_md} | {time.perf_counter()-t4:.2f}s")

    t5 = time.perf_counter()
    print("[6/6] Synthesizing voiceover (ElevenLabs, MP3)…")
    tts_elevenlabs(script_text, voice_mp3, ELEVEN_VOICE_ID_OR_NAME)
    print(f"    Voiceover saved -> {voice_mp3} | {time.perf_counter()-t5:.2f}s")

    print("\n✔ DONE!")
    print(" - Transcript:", transcript_json)
    print(" - Script:    ", script_md)
    print(" - Voiceover: ", voice_mp3)

if __name__ == "__main__":
    main()
