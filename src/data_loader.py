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
    def sanitize_games_df(df):
        if len(df) == 0:
            return df

        clean_df = df.copy()
        clean_df["date"] = pd.to_datetime(clean_df["date"], errors="coerce")
        clean_df = clean_df.dropna(subset=["date"])
        clean_df = clean_df.sort_values("date")
        clean_df = clean_df.drop_duplicates(subset=["gamePk"], keep="last")
        clean_df["date"] = clean_df["date"].dt.strftime("%Y-%m-%d")
        return clean_df

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
        existing_df = existing_df.dropna(subset=["date"]).copy()
        existing_df["date"] = existing_df["date"].dt.strftime("%Y-%m-%d")

        cached_dates = set(existing_df["date"].unique().tolist())
        requested_dates = list(daterange(start_date, end_date))

        missing_dates = [
            date for date in requested_dates
            if date not in cached_dates
        ]

        if len(missing_dates) == 0:
            min_cached = min(cached_dates)
            max_cached = max(cached_dates)
            print(
                f"\nCache histórico completo para rango solicitado "
                f"({start_date} → {end_date})."
            )
            print(f"Rango encontrado en cache: {min_cached} → {max_cached}")

    if len(missing_dates) == 0 and len(existing_df) == 0:

        missing_dates = list(
            daterange(
                start_date,
                end_date
            )
        )

    if len(missing_dates) == 0:

        print("\nNo hay fechas faltantes. Reutilizando cache de juegos históricos.\n")

        return sanitize_games_df(existing_df)

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

    combined_df = pd.concat(
        [existing_df, new_df],
        ignore_index=True
    )
    combined_df["date"] = pd.to_datetime(combined_df["date"], errors="coerce")
    combined_df = combined_df.dropna(subset=["date"])
    combined_df = combined_df.sort_values("date")
    combined_df = combined_df.drop_duplicates(subset=["gamePk"], keep="last")
    combined_df["date"] = combined_df["date"].dt.strftime("%Y-%m-%d")

    return combined_df
