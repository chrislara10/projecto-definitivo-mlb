import requests
import pandas as pd
from pathlib import Path


def _extract_h2h_rows(events):
    rows = []
    for event in events:
        home_team = event.get("home_team")
        away_team = event.get("away_team")
        commence_time = event.get("commence_time")

        home_price = None
        away_price = None

        bookmakers = event.get("bookmakers", [])
        for bookmaker in bookmakers:
            for market in bookmaker.get("markets", []):
                if market.get("key") != "h2h":
                    continue
                for outcome in market.get("outcomes", []):
                    if outcome.get("name") == home_team:
                        home_price = outcome.get("price")
                    elif outcome.get("name") == away_team:
                        away_price = outcome.get("price")
            if home_price is not None and away_price is not None:
                break

        if home_price is None or away_price is None:
            continue

        rows.append(
            {
                "date": pd.to_datetime(commence_time).tz_convert(None).normalize(),
                "home_team": home_team,
                "away_team": away_team,
                "market_home_odds": int(home_price),
                "market_away_odds": int(away_price),
            }
        )
    return rows


def fetch_mlb_h2h_odds(odds_api_key):
    if not odds_api_key:
        return pd.DataFrame()

    url = "https://api.the-odds-api.com/v4/sports/baseball_mlb/odds"
    params = {
        "apiKey": odds_api_key,
        "regions": "us",
        "markets": "h2h",
        "oddsFormat": "american",
    }

    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        events = response.json()
    except Exception as exc:
        print(f"No se pudieron descargar odds reales: {exc}")
        return pd.DataFrame()

    return pd.DataFrame(_extract_h2h_rows(events))


def _fetch_historical_odds_for_date(odds_api_key, date_str):
    url = "https://api.the-odds-api.com/v4/historical/sports/baseball_mlb/odds"
    params = {
        "apiKey": odds_api_key,
        "regions": "us",
        "markets": "h2h",
        "oddsFormat": "american",
        "date": f"{date_str}T12:00:00Z",
    }
    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        payload = response.json()
        events = payload.get("data", [])
    except Exception as exc:
        print(f"No se pudieron descargar odds históricos para {date_str}: {exc}")
        return pd.DataFrame()
    return pd.DataFrame(_extract_h2h_rows(events))


def download_historical_odds_incremental(
    odds_api_key,
    start_date,
    end_date,
    cache_path="data/odds_history.csv",
):
    if not odds_api_key:
        return pd.DataFrame()

    cache_file = Path(cache_path)
    existing_df = pd.DataFrame()
    if cache_file.exists():
        try:
            existing_df = pd.read_csv(cache_file)
            existing_df["date"] = pd.to_datetime(existing_df["date"]).dt.normalize()
        except Exception:
            existing_df = pd.DataFrame()

    requested_dates = pd.date_range(start=start_date, end=end_date, freq="D")
    existing_dates = set(existing_df["date"].dt.strftime("%Y-%m-%d")) if len(existing_df) > 0 else set()
    missing_dates = [d.strftime("%Y-%m-%d") for d in requested_dates if d.strftime("%Y-%m-%d") not in existing_dates]

    new_frames = []
    for date_str in missing_dates:
        day_df = _fetch_historical_odds_for_date(odds_api_key, date_str)
        if len(day_df) > 0:
            new_frames.append(day_df)

    if len(new_frames) > 0:
        new_df = pd.concat(new_frames, ignore_index=True)
        combined = pd.concat([existing_df, new_df], ignore_index=True) if len(existing_df) > 0 else new_df
    else:
        combined = existing_df

    if len(combined) > 0:
        combined["date"] = pd.to_datetime(combined["date"]).dt.normalize()
        combined = combined.drop_duplicates(subset=["date", "home_team", "away_team"], keep="last")
        combined = combined.sort_values(["date", "home_team", "away_team"])
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        combined.to_csv(cache_file, index=False)

    return combined


def update_historical_odds_single_day(
    odds_api_key,
    cache_path="data/odds_history.csv",
    target_date=None,
):
    if not odds_api_key:
        return pd.DataFrame()

    if target_date is None:
        target_date = pd.Timestamp.utcnow().normalize().strftime("%Y-%m-%d")

    cache_file = Path(cache_path)
    existing_df = pd.DataFrame()

    if cache_file.exists():
        try:
            existing_df = pd.read_csv(cache_file)
            existing_df["date"] = pd.to_datetime(existing_df["date"]).dt.normalize()
        except Exception:
            existing_df = pd.DataFrame()

    if len(existing_df) == 0:
        return download_historical_odds_incremental(
            odds_api_key=odds_api_key,
            start_date=target_date,
            end_date=target_date,
            cache_path=cache_path,
        )

    last_cached_date = existing_df["date"].dropna().max()
    if pd.isna(last_cached_date):
        return existing_df

    next_date = (last_cached_date + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    if next_date > target_date:
        return existing_df

    return download_historical_odds_incremental(
        odds_api_key=odds_api_key,
        start_date=next_date,
        end_date=next_date,
        cache_path=cache_path,
    )
