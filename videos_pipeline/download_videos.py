import json
import re
from pathlib import Path
import subprocess

JSON_PATH = "highlights.json"     # <-- path to your saved JSON file
OUT_DIR   = "downloads"
ONLY_VERIFIED = True              # toggle if you want all items

def slugify(text: str, max_len=80) -> str:
    """Safer filename from title; keep ASCII, dashes and underscores."""
    text = text or "highlight"
    text = re.sub(r"[^\w\s-]", "", text, flags=re.UNICODE)
    text = re.sub(r"\s+", "-", text.strip())
    return text[:max_len] or "highlight"

def download_video(url: str, out_dir=OUT_DIR, filename_slug: str | None = None) -> Path | None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # If you pass an explicit slug, yt-dlp will use it as the filename stem
    output_tpl = f"{out_dir}/%(title).80s-%(id)s.%(ext)s" if not filename_slug else f"{out_dir}/{filename_slug}.%(ext)s"

    cmd = [
        "yt-dlp",
        "-S", "ext:mp4:m4a",            # prefer mp4
        "--merge-output-format", "mp4", # ensure mp4 container
        "--no-playlist",
        "--restrict-filenames",         # avoid weird chars
        "-o", output_tpl,
        url,
    ]
    try:
        subprocess.run(cmd, check=True)
        # Get newest file as a simple way to return the path
        files = sorted(out_dir.glob("*"), key=lambda p: p.stat().st_mtime, reverse=True)
        return files[0] if files else None
    except subprocess.CalledProcessError:
        return None

def load_items(json_path: str):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # Accept either a single dict or a list of dicts
    if isinstance(data, dict):
        # If Highlightly response was wrapped (e.g., {"data": [...]})
        if "data" in data and isinstance(data["data"], list):
            return data["data"]
        return [data]
    return data

def main():
    items = load_items(JSON_PATH)

    # Optional filter to VERIFIED clips
    if ONLY_VERIFIED:
        items = [h for h in items if str(h.get("type", "")).upper() == "VERIFIED"]

    print(f"Found {len(items)} items")

    downloaded = []
    for h in items:
        url = h.get("url") or h.get("embedUrl")
        if not url:
            continue

        # Build a neat filename slug: league-date-id-title
        league = (h.get("match") or {}).get("league", {}) or {}
        league_name = league.get("name") or ""
        date_iso = (h.get("match") or {}).get("date") or ""
        date_part = date_iso[:10] if len(date_iso) >= 10 else ""
        hid = h.get("id") or ""
        title = h.get("title") or ""
        slug = slugify(f"{league_name}-{date_part}-{hid}-{title}")

        p = download_video(url, filename_slug=slug)
        if p:
            downloaded.append({"path": str(p), "id": hid, "title": title})
            print("✔ Downloaded:", p.name)
        else:
            print("✖ Failed:", title, "->", url)

    print(f"\nDone. Downloaded {len(downloaded)} files to {OUT_DIR}/")

if __name__ == "__main__":
    main()
