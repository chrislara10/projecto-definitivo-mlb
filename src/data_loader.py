import requests
import pandas as pd

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