import requests
import pandas as pd
from pathlib import Path

from tqdm import tqdm

from src.config import HEADERS
from src.utils import daterange


# =========================================================
# HELPERS
# =========================================================

def _sanitize_games_df(df):
    if len(df) == 0:
        return df
    clean_df = df.copy()
    clean_df["date"] = pd.to_datetime(clean_df["date"], errors="coerce")
    clean_df = clean_df.dropna(subset=["date"])
    clean_df = clean_df.sort_values("date")
    clean_df = clean_df.drop_duplicates(subset=["gamePk"], keep="last")
    clean_df["date"] = clean_df["date"].dt.strftime("%Y-%m-%d")
    return clean_df


def _fetch_games_for_dates(dates):
    """
    Descarga juegos finalizados para una lista de fechas.
    Retorna lista de dicts listos para DataFrame.
    """
    new_rows = []

    for date in tqdm(dates, desc="Descargando fechas"):
        url = (
            f"https://statsapi.mlb.com/api/v1/schedule"
            f"?sportId=1&date={date}&hydrate=probablePitcher"
        )
        try:
            response = requests.get(url, headers=HEADERS, timeout=30)
            response.raise_for_status()
            data = response.json()

            for d in data.get("dates", []):
                for game in d.get("games", []):
                    if game["status"]["detailedState"] != "Final":
                        continue

                    away_team = game["teams"]["away"]["team"]["name"]
                    home_team = game["teams"]["home"]["team"]["name"]
                    away_id   = game["teams"]["away"]["team"]["id"]
                    home_id   = game["teams"]["home"]["team"]["id"]
                    away_score = game["teams"]["away"]["score"]
                    home_score = game["teams"]["home"]["score"]
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
                        "date":       date,
                        "gamePk":     game["gamePk"],
                        "away_team":  away_team,
                        "home_team":  home_team,
                        "away_id":    away_id,
                        "home_id":    home_id,
                        "away_pitcher": away_pitcher,
                        "home_pitcher": home_pitcher,
                        "away_score": away_score,
                        "home_score": home_score,
                        "home_win":   int(home_score > away_score),
                    })

        except Exception as e:
            print(f"ERROR {date}: {e}")

    return new_rows


# =========================================================
# DOWNLOAD HISTORICAL GAMES (FULL — uso interno / one-shot)
# =========================================================

def download_historical_games(start_date, end_date):
    all_dates = list(daterange(start_date, end_date))
    print("\nDOWNLOADING HISTORICAL GAMES...\n")
    rows = _fetch_games_for_dates(all_dates)
    return pd.DataFrame(rows)


# =========================================================
# DOWNLOAD HISTORICAL GAMES — INCREMENTAL
# FIX #10: la función ahora guarda el cache internamente después de
#          cada descarga parcial, de modo que un crash posterior no
#          pierde el trabajo ya realizado.
# =========================================================

def download_historical_games_incremental(
    start_date,
    end_date,
    cache_path="data/historical_games.csv"
):
    cache_file = Path(cache_path)
    cache_file.parent.mkdir(parents=True, exist_ok=True)

    existing_df = pd.DataFrame()

    # --------------------------------------------------
    # 1. Cargar cache existente
    # --------------------------------------------------
    if cache_file.exists():
        try:
            existing_df = pd.read_csv(cache_file)
        except Exception:
            existing_df = pd.DataFrame()

    # --------------------------------------------------
    # 2. Calcular fechas faltantes
    # --------------------------------------------------
    requested_dates = list(daterange(start_date, end_date))

    if len(existing_df) > 0 and "date" in existing_df.columns:
        existing_df["date"] = pd.to_datetime(
            existing_df["date"], errors="coerce"
        )
        existing_df = existing_df.dropna(subset=["date"]).copy()
        existing_df["date"] = existing_df["date"].dt.strftime("%Y-%m-%d")
        cached_dates = set(existing_df["date"].unique())
        missing_dates = [d for d in requested_dates if d not in cached_dates]
    else:
        missing_dates = requested_dates

    # --------------------------------------------------
    # 3. No hay nada nuevo → devolver cache
    # --------------------------------------------------
    if not missing_dates:
        min_d = existing_df["date"].min()
        max_d = existing_df["date"].max()
        print(
            f"\nCache histórico completo ({start_date} → {end_date})."
            f"\nRango en cache: {min_d} → {max_d}\n"
        )
        return _sanitize_games_df(existing_df)

    # --------------------------------------------------
    # 4. Descargar solo fechas faltantes
    # --------------------------------------------------
    print(f"\nDescargando {len(missing_dates)} fechas faltantes...\n")
    new_rows = _fetch_games_for_dates(missing_dates)
    new_df = pd.DataFrame(new_rows)

    # --------------------------------------------------
    # 5. Combinar y guardar cache (FIX #10)
    # --------------------------------------------------
    if len(existing_df) > 0 and len(new_df) > 0:
        combined_df = pd.concat([existing_df, new_df], ignore_index=True)
    elif len(new_df) > 0:
        combined_df = new_df
    else:
        combined_df = existing_df

    combined_df = _sanitize_games_df(combined_df)

    # Guardar inmediatamente para no perder progreso ante un crash posterior
    combined_df.to_csv(cache_path, index=False)
    print(f"Cache guardado en {cache_path} ({len(combined_df)} juegos totales).")

    return combined_df