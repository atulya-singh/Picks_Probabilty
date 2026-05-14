import requests
import json
import os
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

# Explicit path so load_dotenv finds .env regardless of cwd on Windows
_env_path = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=_env_path)

API_KEY = os.getenv("ODDS_API_KEY")
BASE_URL = "https://api.the-odds-api.com/v4"

_RAW_DIR = Path(__file__).resolve().parent / "data" / "raw"


def _print_quota(headers):
    print(f"  x-requests-used:      {headers.get('x-requests-used')}")
    print(f"  x-requests-remaining: {headers.get('x-requests-remaining')}")
    print(f"  x-requests-last:      {headers.get('x-requests-last')}")


def _save_json(data, filename):
    path = _RAW_DIR / filename
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  Saved -> {path}")


def fetch_live_odds():
    print("\n[fetch_live_odds]")
    if not API_KEY:
        raise RuntimeError(f"ODDS_API_KEY not loaded — checked {_env_path}")

    url = f"{BASE_URL}/sports/basketball_nba/odds"
    params = {
        "apiKey": API_KEY,
        "regions": "us",
        "markets": "h2h",
        "oddsFormat": "american",
        "bookmakers": "draftkings,fanduel,betmgm",
    }

    response = requests.get(url, params=params)
    _print_quota(response.headers)

    if response.status_code != 200:
        print(f"  Error {response.status_code}: {response.text}")
        return

    data = response.json()
    game_count = len(data)
    print(f"  Games returned: {game_count}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    _save_json(data, f"live_odds_{timestamp}.json")


def fetch_historical_odds(date_str="2024-01-15T00:00:00Z"):
    print(f"\n[fetch_historical_odds] date={date_str}")
    if not API_KEY:
        raise RuntimeError(f"ODDS_API_KEY not loaded — checked {_env_path}")

    url = f"{BASE_URL}/historical/sports/basketball_nba/odds"
    params = {
        "apiKey": API_KEY,
        "regions": "us",
        "markets": "h2h",
        "oddsFormat": "american",
        "bookmakers": "draftkings,fanduel,betmgm",
        "date": date_str,
    }

    response = requests.get(url, params=params)
    _print_quota(response.headers)

    if response.status_code != 200:
        print(f"  Error {response.status_code}: {response.text}")
        return

    data = response.json()
    game_count = len(data.get("data", []))
    print(f"  Snapshot timestamp:  {data.get('timestamp')}")
    print(f"  Games returned:      {game_count}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    _save_json(data, f"historical_odds_{timestamp}.json")


if __name__ == "__main__":
    fetch_live_odds()
    fetch_historical_odds("2024-01-15T00:00:00Z")
