START_DATE = "2021-03-01"
END_DATE = "2026-10-01"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    )
}

FEATURES = [

    # =====================================
    # TEAM FEATURES
    # =====================================

    "offense_edge_5",
    "offense_edge_10",
    "defense_edge_5",
    "momentum_edge",

    # =====================================
    # PITCHER EDGE FEATURES
    # =====================================

    "pitching_era_edge",
    "pitching_whip_edge",
    "pitching_fip_edge",
    "pitching_k9_edge",

    # =====================================
    # AWAY PITCHER
    # =====================================

    "away_pitcher_era",
    "away_pitcher_whip",
    "away_pitcher_fip",
    "away_pitcher_k9",

    # =====================================
    # HOME PITCHER
    # =====================================

    "home_pitcher_era",
    "home_pitcher_whip",
    "home_pitcher_fip",
    "home_pitcher_k9"
]
# =====================================
# PREDICTION THRESHOLD
# =====================================

PREDICTION_THRESHOLD = 0.54
# =====================================
# TARGET COLUMN
# =====================================

TARGET = "home_win"
