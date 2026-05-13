import numpy as np
import pandas as pd

from src.utils import (
    american_to_probability,
    calculate_ev
)

# =====================================================
# RUN BACKTEST
# =====================================================

def run_backtest(
    test_df,
    pred_probs
):

    backtest_df = test_df.copy()

    # =================================================
    # MODEL PROBABILITIES
    # =================================================

    backtest_df["model_probability"] = (
        pred_probs
    )

    # =================================================
    # RANDOM MARKET ODDS
    # =================================================

    np.random.seed(42)

    possible_odds = (

        list(range(-200, -101)) +

        list(range(100, 201))
    )

    backtest_df["market_odds"] = np.random.choice(

        possible_odds,

        size=len(backtest_df)
    )

    # =================================================
    # IMPLIED PROBABILITY
    # =================================================

    backtest_df["implied_probability"] = (

        backtest_df["market_odds"]

        .apply(
            american_to_probability
        )
    )

    # =================================================
    # EDGE
    # =================================================

    backtest_df["edge"] = (

        backtest_df["model_probability"] -

        backtest_df["implied_probability"]
    )

    # =================================================
    # EXPECTED VALUE
    # =================================================

    backtest_df["expected_value"] = (

        backtest_df.apply(

            lambda row:

            calculate_ev(

                row["model_probability"],

                row["market_odds"]
            ),

            axis=1
        )
    )

    # =================================================
    # BET FILTER
    # =================================================

    backtest_df["bet"] = (

        (
            backtest_df["edge"] > 0.05
        )

        &

        (
            backtest_df["expected_value"] > 0
        )

    ).astype(int)

    # =================================================
    # BET RESULTS
    # =================================================

    backtest_df["bet_result"] = np.where(

        (backtest_df["bet"] == 1)

        &

        (backtest_df["home_win"] == 1),

        np.where(

            backtest_df["market_odds"] > 0,

            backtest_df["market_odds"] / 100,

            100 / abs(
                backtest_df["market_odds"]
            )
        ),

        np.where(

            backtest_df["bet"] == 1,

            -1,

            0
        )
    )

    # =================================================
    # ROI
    # =================================================

    total_bets = (
        backtest_df["bet"] == 1
    ).sum()

    total_profit = (
        backtest_df["bet_result"]
    ).sum()

    roi = (

        total_profit / total_bets

        if total_bets > 0

        else 0
    )

    return (
        backtest_df,
        total_bets,
        total_profit,
        roi
    )