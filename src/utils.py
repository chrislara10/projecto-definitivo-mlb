from datetime import datetime, timedelta

import numpy as np

# =========================================================
# DATE RANGE
# =========================================================

def daterange(
    start_date,
    end_date
):

    start = datetime.strptime(
        start_date,
        "%Y-%m-%d"
    )

    end = datetime.strptime(
        end_date,
        "%Y-%m-%d"
    )

    for n in range(

        (end - start).days + 1
    ):

        yield (

            start +

            timedelta(days=n)

        ).strftime("%Y-%m-%d")

# =========================================================
# AMERICAN ODDS TO IMPLIED PROBABILITY
# =========================================================

def american_to_probability(
    odds
):

    if odds == 0:

        return np.nan

    if odds > 0:

        return 100 / (
            odds + 100
        )

    return abs(odds) / (
        abs(odds) + 100
    )

# =========================================================
# EXPECTED VALUE
# =========================================================

def calculate_ev(
    win_probability,
    american_odds
):

    if american_odds == 0:

        return np.nan

    # =====================================================
    # POSITIVE ODDS
    # =====================================================

    if american_odds > 0:

        profit = (
            american_odds / 100
        )

    # =====================================================
    # NEGATIVE ODDS
    # =====================================================

    else:

        profit = (
            100 / abs(american_odds)
        )

    ev = (

        win_probability * profit -

        (1 - win_probability)
    )

    return ev