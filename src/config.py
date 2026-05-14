import os
from pathlib import Path


def load_env_file(env_path=".env"):
    env_file = Path(env_path)

    if not env_file.exists():
        return

    for raw_line in env_file.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()

        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")

        os.environ.setdefault(key, value)


load_env_file()

START_DATE = "2021-03-01"
END_DATE = "2026-10-01"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    )
}

FEATURES = [
    "offense_edge_5",
    "offense_edge_10",
    "defense_edge_5",
    "momentum_edge",
    "pitching_era_edge",
    "pitching_whip_edge",
    "pitching_fip_edge",
    "pitching_k9_edge",
    "away_pitcher_era",
    "away_pitcher_whip",
    "away_pitcher_fip",
    "away_pitcher_k9",
    "home_pitcher_era",
    "home_pitcher_whip",
    "home_pitcher_fip",
    "home_pitcher_k9",
]

PREDICTION_THRESHOLD = 0.54
TARGET = "home_win"

# Backtest controls
BOOKMAKER_MARGIN = 0.04
MARKET_NOISE_STD = 0.03
MIN_EDGE = 0.02
MIN_EV = 0.01
RANDOM_SEED = 42

ODDS_API_KEY = os.getenv("ODDS_API_KEY", "")
API_SPORTS_KEY = os.getenv("API_SPORTS_KEY", "")
