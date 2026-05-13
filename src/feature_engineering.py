import pandas as pd


# =========================================================
# TEAM GAME LOGS
# =========================================================

def build_team_game_logs(games_df):

    rows = []

    for _, row in games_df.iterrows():

        # AWAY
        rows.append({

            "date": row["date"],
            "team": row["away_team"],
            "team_id": row["away_id"],
            "opponent": row["home_team"],
            "is_home": 0,
            "runs_scored": row["away_score"],
            "runs_allowed": row["home_score"],
            "win": int(
                row["away_score"] >
                row["home_score"]
            )
        })

        # HOME
        rows.append({

            "date": row["date"],
            "team": row["home_team"],
            "team_id": row["home_id"],
            "opponent": row["away_team"],
            "is_home": 1,
            "runs_scored": row["home_score"],
            "runs_allowed": row["away_score"],
            "win": int(
                row["home_score"] >
                row["away_score"]
            )
        })

    df = pd.DataFrame(rows)

    df["date"] = pd.to_datetime(
        df["date"]
    )

    df = df.sort_values([
        "team",
        "date"
    ])

    return df


# =========================================================
# ROLLING FEATURES
# =========================================================

def add_rolling_features(team_logs):

    # OFFENSE LAST 5
    team_logs["runs_scored_last5"] = (

        team_logs
        .groupby("team")["runs_scored"]

        .transform(

            lambda x:
            x.shift(1)
             .rolling(5)
             .mean()
        )
    )

    # OFFENSE LAST 10
    team_logs["runs_scored_last10"] = (

        team_logs
        .groupby("team")["runs_scored"]

        .transform(

            lambda x:
            x.shift(1)
             .rolling(10)
             .mean()
        )
    )

    # DEFENSE LAST 5
    team_logs["runs_allowed_last5"] = (

        team_logs
        .groupby("team")["runs_allowed"]

        .transform(

            lambda x:
            x.shift(1)
             .rolling(5)
             .mean()
        )
    )

    # WIN RATE LAST 10
    team_logs["win_rate_last10"] = (

        team_logs
        .groupby("team")["win"]

        .transform(

            lambda x:
            x.shift(1)
             .rolling(10)
             .mean()
        )
    )

    return team_logs


# =========================================================
# BUILD MODEL DATASET
# =========================================================

def build_model_dataset(
    games_df,
    team_logs
):

    away_features = team_logs.rename(columns={

        "team": "away_team",
        "date": "game_date",

        "runs_scored_last5":
            "away_runs_last5",

        "runs_scored_last10":
            "away_runs_last10",

        "runs_allowed_last5":
            "away_ra_last5",

        "win_rate_last10":
            "away_winrate_last10"
    })

    home_features = team_logs.rename(columns={

        "team": "home_team",
        "date": "game_date",

        "runs_scored_last5":
            "home_runs_last5",

        "runs_scored_last10":
            "home_runs_last10",

        "runs_allowed_last5":
            "home_ra_last5",

        "win_rate_last10":
            "home_winrate_last10"
    })

    games_df["date"] = pd.to_datetime(
        games_df["date"]
    )

    # MERGE AWAY
    df = games_df.merge(

        away_features[[

            "away_team",
            "game_date",
            "away_runs_last5",
            "away_runs_last10",
            "away_ra_last5",
            "away_winrate_last10"
        ]],

        left_on=[
            "away_team",
            "date"
        ],

        right_on=[
            "away_team",
            "game_date"
        ],

        how="left"
    )

    # MERGE HOME
    df = df.merge(

        home_features[[

            "home_team",
            "game_date",
            "home_runs_last5",
            "home_runs_last10",
            "home_ra_last5",
            "home_winrate_last10"
        ]],

        left_on=[
            "home_team",
            "date"
        ],

        right_on=[
            "home_team",
            "game_date"
        ],

        how="left"
    )

    # FEATURES
    df["offense_edge_5"] = (
        df["home_runs_last5"] -
        df["away_runs_last5"]
    )

    df["offense_edge_10"] = (
        df["home_runs_last10"] -
        df["away_runs_last10"]
    )

    df["defense_edge_5"] = (
        df["away_ra_last5"] -
        df["home_ra_last5"]
    )

    df["momentum_edge"] = (
        df["home_winrate_last10"] -
        df["away_winrate_last10"]
    )

    return df