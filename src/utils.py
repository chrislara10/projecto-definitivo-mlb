from __future__ import annotations

from datetime import datetime, timedelta
from typing import Iterable

import numpy as np
import pandas as pd


# =========================================================
# DATE HELPERS
# =========================================================

def daterange(start_date: str, end_date: str) -> Iterable[str]:
    """Yield dates as YYYY-MM-DD strings, inclusive."""
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")

    if end < start:
        raise ValueError("end_date must be greater than or equal to start_date")

    for n in range((end - start).days + 1):
        yield (start + timedelta(days=n)).strftime("%Y-%m-%d")


def seasons_between(start_date: str, end_date: str) -> list[int]:
    """Return MLB seasons touched by the date range."""
    start_year = pd.to_datetime(start_date).year
    end_year = pd.to_datetime(end_date).year
    return list(range(start_year, end_year + 1))


# =========================================================
# ODDS / EXPECTED VALUE
# =========================================================

def american_to_probability(odds):
    """
    Convert American odds to implied probability.
    Supports scalars, numpy arrays, and pandas Series.
    """
    odds_arr = np.asarray(odds, dtype=float)

    result = np.where(
        odds_arr > 0,
        100 / (odds_arr + 100),
        np.where(odds_arr < 0, np.abs(odds_arr) / (np.abs(odds_arr) + 100), np.nan),
    )

    if np.isscalar(odds):
        return float(result)

    return result


def american_profit_multiplier(odds):
    """
    Profit returned per 1 unit staked, excluding original stake.
    Example: +150 -> 1.5, -200 -> 0.5
    """
    odds_arr = np.asarray(odds, dtype=float)

    result = np.where(
        odds_arr > 0,
        odds_arr / 100,
        np.where(odds_arr < 0, 100 / np.abs(odds_arr), np.nan),
    )

    if np.isscalar(odds):
        return float(result)

    return result


def calculate_ev(win_probability, american_odds):
    """
    Expected profit per 1 unit staked.
    Supports scalars, numpy arrays, and pandas Series.
    """
    profit = american_profit_multiplier(american_odds)
    ev = np.asarray(win_probability, dtype=float) * profit - (1 - np.asarray(win_probability, dtype=float))

    if np.isscalar(win_probability) and np.isscalar(american_odds):
        return float(ev)

    return ev


def calculate_profit(odds, won, stake: float = 1.0):
    """Profit/loss for a settled bet."""
    profit = american_profit_multiplier(odds)
    result = np.where(won, profit * stake, -stake)

    if np.isscalar(odds) and np.isscalar(won):
        return float(result)

    return result
