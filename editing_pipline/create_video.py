from dotenv import load_dotenv
import os
import requests
import subprocess
from pathlib import Path

load_dotenv()

API_KEY = os.environ.get("HIGHLIGHTLY_API_KEY", "<YOUR_API_KEY>")
BASE = "https://basketball.highlightly.net"  # use RapidAPI base if you subscribed there

HEADERS = {
    "x-rapidapi-key": API_KEY,
    # If using RapidAPI:
    # "x-rapidapi-host": "basketball-highlights-api.p.rapidapi.com"
}

def fetch_highlights(date=None, league_name=None, country_code=None, limit=20, offset=0, timezone="Europe/Kyiv"):
    params = {"limit": limit, "offset": offset, "timezone": timezone}
    if date: params["date"] = date 
    if league_name: params["leagueName"] = league_name
    if country_code: params["countryCode"] = country_code

    r = requests.get(f"{BASE}/highlights", headers=HEADERS, params=params, timeout=30)
    r.raise_for_status()
    payload = r.json()
    return (payload or {}).get("data", [])

# example: NBA or NBL on a day
items = fetch_highlights(league_name="NBL", limit=40)
print(f"Fetched {len(items)} highlights")
for h in items:
    print(h["id"], h.get("type"), h.get("title"), "->", h.get("url") or h.get("embedUrl"))


def download_video(url: str, out_dir="downloads") -> Path | None:
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    # Best MP4 (falls back gracefully). Add --geo-bypass if needed.
    cmd = [
        "yt-dlp",
        "-S", "ext:mp4:m4a",          # prefer mp4
        "-o", f"{out_dir}/%(title).80s-%(id)s.%(ext)s",
        "--no-playlist",
        url
    ]
    try:
        subprocess.run(cmd, check=True)
        # Get last downloaded file by querying yt-dlp’s JSON if you want the exact filename;
        # here we just return the newest file:
        files = sorted(Path(out_dir).glob("*"), key=lambda p: p.stat().st_mtime, reverse=True)
        return files[0] if files else None
    except subprocess.CalledProcessError:
        return None

# pick a few VERIFIED clips for cleaner curation
selected = [h for h in items if h.get("type") == "VERIFIED"][:5] or items[:5]
local_paths = []
for h in selected[:1]:
    url = h.get("url") or h.get("embedUrl")
    if not url: 
        continue
    p = download_video(url)
    if p: 
        local_paths.append((p, h))   # keep (path, metadata)