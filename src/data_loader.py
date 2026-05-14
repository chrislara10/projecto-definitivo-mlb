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

    rows = []

    all_dates = list(
        daterange(
            start_date,
            end_date
        )
    )

    print("\nDOWNLOADING HISTORICAL GAMES...\n")

    for date in tqdm(all_dates):

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

                    rows.append({

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

                        "home_win": int(
                            home_score > away_score
                        )
                    })

        except Exception as e:

            print(
                f"ERROR {date}: {e}"
            )

    return pd.DataFrame(rows)


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

    all_dates = set(
        daterange(
            start_date,
            end_date
        )
    )

    existing_dates = set()

    if len(existing_df) > 0 and "date" in existing_df.columns:

        existing_dates = set(
            pd.to_datetime(existing_df["date"])
            .dt.strftime("%Y-%m-%d")
            .unique()
            .tolist()
        )

    missing_dates = sorted(
        all_dates - existing_dates
    )

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
