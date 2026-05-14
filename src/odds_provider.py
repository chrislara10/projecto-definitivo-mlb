import requests
import pandas as pd


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

    return pd.DataFrame(rows)
