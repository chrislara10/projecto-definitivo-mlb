import json
from pathlib import Path

import requests
import pandas as pd
import numpy as np

from tqdm import tqdm

# =====================================================
# MLB API
# =====================================================

BASE_URL = "https://statsapi.mlb.com/api/v1"

HEADERS = {
    "User-Agent": "Mozilla/5.0"
}

CACHE_DIR = Path("data/cache")
PITCHER_ID_CACHE_FILE = CACHE_DIR / "pitcher_id_map.json"
PITCHER_LOG_CACHE_DIR = CACHE_DIR / "pitcher_logs"


def _ensure_cache_dirs():

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    PITCHER_LOG_CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _load_pitcher_id_cache():

    _ensure_cache_dirs()

    if not PITCHER_ID_CACHE_FILE.exists():

        return {}

    try:

        return json.loads(PITCHER_ID_CACHE_FILE.read_text(encoding="utf-8"))

    except Exception:

        return {}


def _save_pitcher_id_cache(cache):

    _ensure_cache_dirs()
    PITCHER_ID_CACHE_FILE.write_text(
        json.dumps(cache, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )


def _pitcher_log_cache_path(pitcher_id, season):

    _ensure_cache_dirs()
    return PITCHER_LOG_CACHE_DIR / f"{pitcher_id}_{season}.csv"


def _load_pitcher_log_cache(pitcher_id, season):

    path = _pitcher_log_cache_path(pitcher_id, season)

    if not path.exists():

        return None

    try:

        df = pd.read_csv(path)

        if len(df) == 0:

            return pd.DataFrame()

        df["date"] = pd.to_datetime(df["date"])

        return df

    except Exception:

        return None


def _save_pitcher_log_cache(pitcher_id, season, logs_df):

    path = _pitcher_log_cache_path(pitcher_id, season)
    logs_df.to_csv(path, index=False)

# =====================================================
# SEARCH PITCHER ID
# =====================================================

def search_pitcher_id(
    pitcher_name,
    pitcher_id_cache
):

    if pitcher_name in pitcher_id_cache:

        return pitcher_id_cache[pitcher_name]

    try:

        url = (
            f"{BASE_URL}/people/search"
            f"?names={pitcher_name}"
        )

        response = requests.get(

            url,

            headers=HEADERS,

            timeout=30
        )

        data = response.json()

        people = data.get(
            "people",
            []
        )

        if len(people) == 0:

            pitcher_id_cache[pitcher_name] = None
            return None

        pitcher_id = people[0]["id"]
        pitcher_id_cache[pitcher_name] = pitcher_id

        return pitcher_id

    except:

        return None

# =====================================================
# DOWNLOAD GAME LOGS
# =====================================================

def download_pitcher_game_logs(
    pitcher_id,
    season
):

    cached = _load_pitcher_log_cache(
        pitcher_id=pitcher_id,
        season=season
    )

    if cached is not None:

        return cached

    url = (

        f"{BASE_URL}/people/"
        f"{pitcher_id}"
        f"/stats"

        f"?stats=gameLog"

        f"&group=pitching"

        f"&season={season}"
    )

    try:

        response = requests.get(

            url,

            headers=HEADERS,

            timeout=30
        )

        response.raise_for_status()

        data = response.json()

        stats = data.get(
            "stats",
            []
        )

        if len(stats) == 0:

            return pd.DataFrame()

        splits = stats[0].get(
            "splits",
            []
        )

        rows = []

        for s in splits:

            stat = s.get(
                "stat",
                {}
            )

            rows.append({

                "date":
                    s.get("date"),

                "innings_pitched":
                    float(
                        stat.get(
                            "inningsPitched",
                            0
                        )
                    ),

                "earned_runs":
                    stat.get(
                        "earnedRuns",
                        0
                    ),

                "hits":
                    stat.get(
                        "hits",
                        0
                    ),

                "walks":
                    stat.get(
                        "baseOnBalls",
                        0
                    ),

                "strikeouts":
                    stat.get(
                        "strikeOuts",
                        0
                    ),

                "home_runs":
                    stat.get(
                        "homeRuns",
                        0
                    )
            })

        df = pd.DataFrame(rows)

        if len(df) == 0:

            return pd.DataFrame()

        df["date"] = pd.to_datetime(
            df["date"]
        )

        df = df.sort_values(
            "date"
        )

        _save_pitcher_log_cache(
            pitcher_id=pitcher_id,
            season=season,
            logs_df=df
        )

        return df

    except Exception as e:

        print(
            f"ERROR pitcher {pitcher_id}: {e}"
        )

        return pd.DataFrame()



def download_pitcher_logs_for_seasons(
    pitcher_id,
    seasons
):

    frames = []

    for season in sorted(set(seasons)):

        season_logs = download_pitcher_game_logs(
            pitcher_id=pitcher_id,
            season=season
        )

        if len(season_logs) > 0:

            frames.append(season_logs)

    if len(frames) == 0:

        return pd.DataFrame()

    return (
        pd.concat(frames, ignore_index=True)
        .sort_values("date")
        .drop_duplicates(subset=["date"], keep="last")
    )

# =====================================================
# CALCULATE ROLLING METRICS
# =====================================================

def calculate_rolling_metrics(
    logs_df,
    current_date
):

    prior_games = logs_df[

        logs_df["date"] < current_date

    ].copy()

    # Need minimum sample
    if len(prior_games) < 3:

        return None

    # Last 30 days
    cutoff = current_date - pd.Timedelta(days=30)

    recent = prior_games[
        prior_games["date"] >= cutoff
    ].copy()

    if len(recent) < 3:

        recent = prior_games.tail(5)

    ip = recent[
        "innings_pitched"
    ].sum()

    if ip <= 0:

        return None

    er = recent[
        "earned_runs"
    ].sum()

    hits = recent[
        "hits"
    ].sum()

    walks = recent[
        "walks"
    ].sum()

    strikeouts = recent[
        "strikeouts"
    ].sum()

    hr = recent[
        "home_runs"
    ].sum()

    era = (
        er * 9
    ) / ip

    whip = (
        hits + walks
    ) / ip

    k9 = (
        strikeouts * 9
    ) / ip

    bb9 = (
        walks * 9
    ) / ip

    fip = (

        (
            13 * hr
        ) +

        (
            3 * walks
        ) -

        (
            2 * strikeouts
        )

    ) / ip + 3.2

    return {

        "era":
            round(era, 3),

        "whip":
            round(whip, 3),

        "k9":
            round(k9, 3),

        "bb9":
            round(bb9, 3),

        "fip":
            round(fip, 3)
    }

# =====================================================
# BUILD PITCHER FEATURES
# =====================================================

def build_pitcher_features(
    games_df
):

    games_df = games_df.copy()

    games_df["date"] = pd.to_datetime(
        games_df["date"]
    )

    # =================================================
    # BUILD PITCHER MAP
    # =================================================

    pitcher_names = pd.concat([

        games_df[[
            "away_pitcher"
        ]].rename(columns={
            "away_pitcher":
            "pitcher_name"
        }),

        games_df[[
            "home_pitcher"
        ]].rename(columns={
            "home_pitcher":
            "pitcher_name"
        })

    ])

    pitcher_names = (
        pitcher_names
        .dropna()
        .drop_duplicates()
    )

    pitcher_map = {}
    pitcher_id_cache = _load_pitcher_id_cache()

    all_pitchers = pitcher_names[
        "pitcher_name"
    ].tolist()
    missing_pitchers = [
        p for p in all_pitchers
        if p not in pitcher_id_cache
    ]

    print(
        f"\nSEARCHING PITCHER IDS... faltantes: {len(missing_pitchers)} de {len(all_pitchers)}\n"
    )

    for pitcher in tqdm(missing_pitchers):

        pitcher_id = search_pitcher_id(
            pitcher,
            pitcher_id_cache
        )

    for pitcher in all_pitchers:
        pitcher_id = pitcher_id_cache.get(pitcher)
        if pitcher_id is not None:
            pitcher_map[pitcher] = pitcher_id

    _save_pitcher_id_cache(pitcher_id_cache)

    # =================================================
    # DOWNLOAD LOGS
    # =================================================

    pitcher_logs = {}
    seasons = games_df["date"].dt.year.unique().tolist()

    print(
        "\nDOWNLOADING PITCHER LOGS...\n"
    )

    for pitcher_name, pitcher_id in tqdm(
        pitcher_map.items()
    ):

        logs = download_pitcher_logs_for_seasons(
            pitcher_id=pitcher_id,
            seasons=seasons
        )

        if len(logs) > 0:

            pitcher_logs[
                pitcher_name
            ] = logs

    # =================================================
    # BUILD FEATURES GAME BY GAME
    # =================================================

    rows = []

    print(
        "\nBUILDING TEMPORAL PITCHER FEATURES...\n"
    )

    for _, row in tqdm(
        games_df.iterrows(),
        total=len(games_df)
    ):

        game_date = row["date"]

        away_pitcher = row["away_pitcher"]

        home_pitcher = row["home_pitcher"]

        away_metrics = None
        home_metrics = None

        if away_pitcher in pitcher_logs:

            away_metrics = calculate_rolling_metrics(

                pitcher_logs[
                    away_pitcher
                ],

                game_date
            )

        if home_pitcher in pitcher_logs:

            home_metrics = calculate_rolling_metrics(

                pitcher_logs[
                    home_pitcher
                ],

                game_date
            )

        if (
            away_metrics is None or
            home_metrics is None
        ):

            continue

        rows.append({

            "date":
                game_date,

            "away_pitcher":
                away_pitcher,

            "home_pitcher":
                home_pitcher,

            "away_pitcher_era":
                away_metrics["era"],

            "away_pitcher_whip":
                away_metrics["whip"],

            "away_pitcher_fip":
                away_metrics["fip"],

            "away_pitcher_k9":
                away_metrics["k9"],

            "home_pitcher_era":
                home_metrics["era"],

            "home_pitcher_whip":
                home_metrics["whip"],

            "home_pitcher_fip":
                home_metrics["fip"],

            "home_pitcher_k9":
                home_metrics["k9"]
        })

    return pd.DataFrame(rows)

# =====================================================
# MERGE FEATURES
# =====================================================

def merge_pitcher_features(
    model_df,
    pitcher_features
):

    model_df["date"] = pd.to_datetime(
        model_df["date"]
    )

    pitcher_features["date"] = pd.to_datetime(
        pitcher_features["date"]
    )

    model_df = model_df.merge(

        pitcher_features,

        on=[
            "date",
            "away_pitcher",
            "home_pitcher"
        ],

        how="left"
    )

    # =================================================
    # EDGE FEATURES
    # =================================================

    model_df["pitching_era_edge"] = (

        model_df[
            "away_pitcher_era"
        ] -

        model_df[
            "home_pitcher_era"
        ]
    )

    model_df["pitching_whip_edge"] = (

        model_df[
            "away_pitcher_whip"
        ] -

        model_df[
            "home_pitcher_whip"
        ]
    )

    model_df["pitching_fip_edge"] = (

        model_df[
            "away_pitcher_fip"
        ] -

        model_df[
            "home_pitcher_fip"
        ]
    )

    model_df["pitching_k9_edge"] = (

        model_df[
            "home_pitcher_k9"
        ] -

        model_df[
            "away_pitcher_k9"
        ]
    )

    return model_df
