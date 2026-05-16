from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import MIN_EDGE, RANDOM_SEED
from src.utils import (
    american_profit_multiplier,
    american_to_probability,
    calculate_ev,
)


# =====================================================
# RUN BACKTEST
# =====================================================

def run_backtest(
    test_df: pd.DataFrame,
    pred_probs,
    market_odds=None,
    min_edge: float = MIN_EDGE,
):
    """
    Vectorized backtest.

    market_odds can be passed as a Series/array if you have real odds.
    If omitted, synthetic odds are generated only for testing the pipeline.
    """
    backtest_df = test_df.copy().reset_index(drop=True)
    pred_probs = np.asarray(pred_probs, dtype=float)

    if len(backtest_df) != len(pred_probs):
        raise ValueError("test_df and pred_probs must have the same length")

    backtest_df["model_probability"] = pred_probs

    if market_odds is None:
        rng = np.random.default_rng(RANDOM_SEED)
        possible_odds = np.array(list(range(-200, -101)) + list(range(100, 201)))
        backtest_df["market_odds"] = rng.choice(possible_odds, size=len(backtest_df))
    else:
        if len(market_odds) != len(backtest_df):
            raise ValueError("market_odds must have the same length as test_df")
        backtest_df["market_odds"] = np.asarray(market_odds, dtype=float)

    odds = backtest_df["market_odds"].to_numpy(dtype=float)

    backtest_df["implied_probability"] = american_to_probability(odds)
    backtest_df["edge"] = backtest_df["model_probability"] - backtest_df["implied_probability"]
    backtest_df["expected_value"] = calculate_ev(backtest_df["model_probability"], odds)

    backtest_df["bet"] = (
        (backtest_df["edge"] > min_edge)
        & (backtest_df["expected_value"] > 0)
    ).astype(int)

    profit_multiplier = american_profit_multiplier(odds)
    won = backtest_df["home_win"].to_numpy(dtype=int) == 1
    bet = backtest_df["bet"].to_numpy(dtype=int) == 1

    backtest_df["bet_result"] = np.where(
        bet,
        np.where(won, profit_multiplier, -1.0),
        0.0,
    )

    total_bets = int(backtest_df["bet"].sum())
    total_profit = float(backtest_df["bet_result"].sum())
    roi = total_profit / total_bets if total_bets > 0 else 0.0

    return backtest_df, total_bets, total_profit, roi