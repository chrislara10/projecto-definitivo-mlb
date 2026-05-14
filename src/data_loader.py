import requests
import pandas as pd
from pathlib import Path

from tqdm import tqdm

from src.config import (
    HEADERS
)

from src.utils import (
    daterange
)

# =========================================================
# DOWNLOAD HISTORICAL GAMES
# =========================================================

def download_historical_games(
    start_date,
    end_date
):
    return download_historical_games_incremental(
        start_date=start_date,
        end_date=end_date,
        cache_path="data/historical_games.csv",
    )


def download_historical_games_incremental(
    start_date,
    end_date,
    cache_path="data/historical_games.csv"
):

    cache_file = Path(cache_path)
    existing_df = pd.DataFrame()

    if cache_file.exists():

        try:

            existing_df = pd.read_csv(cache_file)

        except Exception:

            existing_df = pd.DataFrame()

    missing_dates = []

    if len(existing_df) > 0 and "date" in existing_df.columns:

        existing_df["date"] = pd.to_datetime(existing_df["date"], errors="coerce")
        last_cached_date = existing_df["date"].dropna().max()

        if pd.notna(last_cached_date):

            next_date = (last_cached_date + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
            print(f"\nÚltima fecha en cache: {last_cached_date.strftime('%Y-%m-%d')}")
            missing_dates = list(daterange(next_date, end_date))

    if len(missing_dates) == 0:

        missing_dates = list(
            daterange(
                start_date,
                end_date
            )
        ) if len(existing_df) == 0 else []

    if len(missing_dates) == 0:

        print("\nNo hay fechas faltantes. Reutilizando cache de juegos históricos.\n")

        return existing_df

    print(
        f"\nDescargando solo fechas faltantes: {len(missing_dates)} días.\n"
    )

    new_rows = []

    for date in tqdm(missing_dates):

        url = (
            f"https://statsapi.mlb.com/api/v1/schedule"
            f"?sportId=1"
            f"&date={date}"
            f"&hydrate=probablePitcher"
        )

        try:

            response = requests.get(
                url,
                headers=HEADERS,
                timeout=30
            )

            response.raise_for_status()

            data = response.json()

            for d in data.get("dates", []):

                for game in d.get("games", []):

                    status = (
                        game["status"]["detailedState"]
                    )

                    if status != "Final":

                        continue

                    away_team = (
                        game["teams"]["away"]["team"]["name"]
                    )

                    home_team = (
                        game["teams"]["home"]["team"]["name"]
                    )

                    away_id = (
                        game["teams"]["away"]["team"]["id"]
                    )

                    home_id = (
                        game["teams"]["home"]["team"]["id"]
                    )

                    away_score = (
                        game["teams"]["away"]["score"]
                    )

                    home_score = (
                        game["teams"]["home"]["score"]
                    )

                    away_pitcher = (
                        game["teams"]["away"]
                        .get("probablePitcher", {})
                        .get("fullName", "Unknown")
                    )

                    home_pitcher = (
                        game["teams"]["home"]
                        .get("probablePitcher", {})
                        .get("fullName", "Unknown")
                    )

                    new_rows.append({

                        "date": date,
                        "gamePk": game["gamePk"],
                        "away_team": away_team,
                        "home_team": home_team,
                        "away_id": away_id,
                        "home_id": home_id,
                        "away_pitcher": away_pitcher,
                        "home_pitcher": home_pitcher,
                        "away_score": away_score,
                        "home_score": home_score,
                        "home_win": int(home_score > away_score)
                    })

        except Exception as e:

            print(
                f"ERROR {date}: {e}"
            )

    new_df = pd.DataFrame(new_rows)

    if len(existing_df) == 0:

        return new_df

    if len(new_df) == 0:

        return existing_df

    return pd.concat(
        [existing_df, new_df],
        ignore_index=True
    )
