from dotenv import load_dotenv
import os
import requests
import json

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
# print(f"Fetched {len(items)} highlights")
# for h in items:
#     print(h["id"], h.get("type"), h.get("title"), "->", h.get("url") or h.get("embedUrl"))

with open("highlights.json", "w", encoding="utf-8") as f:
    json.dump(items, f, ensure_ascii=False, indent=2)

print(f"Saved {len(items)} highlights to highlights.json")