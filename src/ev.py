import numpy as np

def american_to_probability(
    odds
):

    if odds > 0:

        return 100 / (
            odds + 100
        )

    return abs(odds) / (
        abs(odds) + 100
    )

def calculate_ev(
    win_probability,
    american_odds
):

    if american_odds == 0:

        return np.nan

    if american_odds > 0:

        profit = (
            american_odds / 100
        )

    else:

        profit = (
            100 / abs(american_odds)
        )

    return (

        win_probability * profit -

        (1 - win_probability)
    )

def calculate_profit(
    odds,
    won,
    stake=1
):

    if won:

        if odds > 0:

            return (
                odds / 100
            ) * stake

        return (
            100 / abs(odds)
        ) * stake

    return -stake