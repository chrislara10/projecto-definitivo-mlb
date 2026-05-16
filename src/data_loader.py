from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

from src.config import HEADERS, MLB_BASE_URL, REQUEST_TIMEOUT


# =========================================================
# INTERNAL HELPERS
# =========================================================

def _month_chunks(start_date: str, end_date: str) -> list[tuple[str, str]]:
    """
    MLB schedule endpoint supports date ranges.
    Monthly chunks avoid one request per day and keep responses manageable.
    """
    start = pd.to_datetime(start_date).normalize()
    end = pd.to_datetime(end_date).normalize()

    chunks = []
    cursor = start

    while cursor <= end:
        month_end = min(cursor + pd.offsets.MonthEnd(0), end)
        chunks.append((cursor.strftime("%Y-%m-%d"), month_end.strftime("%Y-%m-%d")))
        cursor = month_end + pd.Timedelta(days=1)

    return chunks


def _safe_team_payload(game: dict, side: str) -> dict:
    team_block = game["teams"][side]
    team = team_block["team"]
    pitcher = team_block.get("probablePitcher") or {}

    return {
        f"{side}_team": team.get("name"),
        f"{side}_id": team.get("id"),
        f"{side}_score": team_block.get("score"),
        f"{side}_pitcher": pitcher.get("fullName"),
        f"{side}_pitcher_id": pitcher.get("id"),
    }


# =========================================================
# DOWNLOAD HISTORICAL GAMES
# =========================================================

def download_historical_games(
    start_date: str,
    end_date: str,
    cache_path: str | Path | None = None,
    force_download: bool = False,
) -> pd.DataFrame:
    """
    Download MLB final games for a date range.

    Improvements over the previous version:
    - Uses monthly range requests instead of one request per day.
    - Reuses one requests.Session.
    - Stores pitcher IDs directly from the schedule response.
    - Optional CSV cache to avoid downloading the same history repeatedly.
    """
    cache_path = Path(cache_path) if cache_path else None

    if cache_path and cache_path.exists() and not force_download:
        df = pd.read_csv(cache_path)
        df["date"] = pd.to_datetime(df["date"])
        return df.sort_values("date").drop_duplicates("gamePk").reset_index(drop=True)

    rows: list[dict] = []
    chunks = _month_chunks(start_date, end_date)

    print("\nDOWNLOADING HISTORICAL GAMES...\n")

    with requests.Session() as session:
        session.headers.update(HEADERS)

        for chunk_start, chunk_end in tqdm(chunks, desc="MLB schedule chunks"):
            url = (
                f"{MLB_BASE_URL}/schedule"
                f"?sportId=1"
                f"&startDate={chunk_start}"
                f"&endDate={chunk_end}"
                f"&hydrate=probablePitcher"
            )

            try:
                response = session.get(url, timeout=REQUEST_TIMEOUT)
                response.raise_for_status()
                data = response.json()
            except requests.RequestException as exc:
                print(f"ERROR {chunk_start} to {chunk_end}: {exc}")
                continue

            for date_block in data.get("dates", []):
                game_date = date_block.get("date")

                for game in date_block.get("games", []):
                    if game.get("status", {}).get("detailedState") != "Final":
                        continue

                    away = _safe_team_payload(game, "away")
                    home = _safe_team_payload(game, "home")

                    if away["away_score"] is None or home["home_score"] is None:
                        continue

                    rows.append(
                        {
                            "date": game_date,
                            "gamePk": game.get("gamePk"),
                            **away,
                            **home,
                            "home_win": int(home["home_score"] > away["away_score"]),
                        }
                    )

    df = pd.DataFrame(rows)

    if df.empty:
        return df

    df["date"] = pd.to_datetime(df["date"])
    df = (
        df.sort_values(["date", "gamePk"])
        .drop_duplicates("gamePk")
        .reset_index(drop=True)
    )

    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(cache_path, index=False)

    return df

