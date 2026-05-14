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

    print(f"\n[historical_games] incremental cache_path: {cache_file.resolve()}")
    print(f"[historical_games] incremental rango solicitado: {start_date} -> {end_date}")

    if cache_file.exists():

        try:

            existing_df = pd.read_csv(cache_file)
            print(f"[historical_games] incremental cache rows: {len(existing_df)}")

        except Exception:

            existing_df = pd.DataFrame()

    missing_dates = []

    if len(existing_df) > 0:

        if "date" not in existing_df.columns:
            print(
                "\n[historical_games] Cache existente sin columna date. "
                "Se reutiliza tal cual y no se descarga histórico completo.\n"
            )
            return existing_df

        existing_df["date"] = pd.to_datetime(existing_df["date"], errors="coerce")
        last_cached_date = existing_df["date"].dropna().max()

        if pd.notna(last_cached_date):

            next_date = (last_cached_date + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
            print(f"\nÚltima fecha en cache: {last_cached_date.strftime('%Y-%m-%d')}")
            missing_dates = list(daterange(next_date, end_date))
        else:
            print(
                "\n[historical_games] No se pudo parsear columna date del cache. "
                "Se reutiliza cache sin redescargar para evitar full refresh.\n"
            )
            return existing_df

    if len(missing_dates) == 0 and len(existing_df) == 0:

        missing_dates = list(
            daterange(
                start_date,
                end_date
            )
        )
        print(f"[historical_games] cache vacío, fechas iniciales a descargar: {len(missing_dates)}")

    if len(missing_dates) == 0:

        print("\nNo hay fechas faltantes. Reutilizando cache de juegos históricos.\n")

        return existing_df

    print(
        f"\nDescargando solo fechas faltantes: {len(missing_dates)} días.\n"
    )
    print(f"[historical_games] primer faltante: {missing_dates[0]}")
    print(f"[historical_games] último faltante: {missing_dates[-1]}")

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


def update_historical_games_single_day(
    cache_path="data/historical_games.csv",
    target_date=None
):
    cache_file = Path(cache_path)
    print(f"\n[historical_games] single_day cache_path: {cache_file.resolve()}")

    if target_date is None:
        target_date = pd.Timestamp.utcnow().normalize().strftime("%Y-%m-%d")
    print(f"[historical_games] single_day target_date: {target_date}")

    if cache_file.exists():
        existing_df = pd.read_csv(cache_file)
        print(f"[historical_games] single_day cache rows: {len(existing_df)}")
    else:
        existing_df = pd.DataFrame()
        print("[historical_games] single_day cache no existe.")

    if len(existing_df) == 0:
        print("\n[historical_games] Cache vacío. Ejecuta una carga inicial completa primero.\n")
        return existing_df

    if "date" not in existing_df.columns:
        print("\n[historical_games] Cache sin columna date. Se omite actualización.\n")
        return existing_df

    existing_df["date"] = pd.to_datetime(existing_df["date"], errors="coerce")
    last_cached_date = existing_df["date"].dropna().max()
    print(f"[historical_games] single_day last_cached_date: {last_cached_date}")

    if pd.isna(last_cached_date):
        print("\n[historical_games] No hay fecha válida en cache. Se omite actualización.\n")
        return existing_df

    next_date = (last_cached_date + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    print(f"[historical_games] single_day next_date: {next_date}")

    if next_date > target_date:
        print("\n[historical_games] Cache al día. No hay fecha faltante.\n")
        return existing_df

    print(f"\n[historical_games] Descargando solo día faltante: {next_date}\n")

    new_df = download_historical_games_incremental(
        start_date=next_date,
        end_date=next_date,
        cache_path=cache_path
    )

    if len(new_df) > 0:
        new_df = new_df.sort_values("date").drop_duplicates(subset=["gamePk"])
        new_df.to_csv(cache_file, index=False)

    return new_df
