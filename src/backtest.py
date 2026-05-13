import numpy as np

from src.utils import american_to_probability, calculate_ev


def probability_to_american(probability):
    probability = float(np.clip(probability, 1e-6, 1 - 1e-6))

    if probability >= 0.5:
        return int(round(-(100 * probability) / (1 - probability)))

    return int(round((100 * (1 - probability)) / probability))


def build_market_probabilities(model_probabilities, margin=0.04, noise_std=0.03, random_seed=42):
    rng = np.random.default_rng(random_seed)
    noise = rng.normal(loc=0.0, scale=noise_std, size=len(model_probabilities))

    fair_probability = np.clip(model_probabilities + noise, 0.05, 0.95)
    market_home_probability = np.clip(fair_probability * (1 + margin), 0.05, 0.98)
    market_away_probability = np.clip((1 - fair_probability) * (1 + margin), 0.05, 0.98)

    return market_home_probability, market_away_probability


def run_backtest(test_df, pred_probs, min_edge=0.02, min_ev=0.01, margin=0.04, noise_std=0.03, random_seed=42):
    backtest_df = test_df.copy()
    backtest_df["model_probability_home"] = pred_probs
    backtest_df["model_probability_away"] = 1 - backtest_df["model_probability_home"]

    market_home_probability, market_away_probability = build_market_probabilities(
        model_probabilities=backtest_df["model_probability_home"].values,
        margin=margin,
        noise_std=noise_std,
        random_seed=random_seed,
    )

    backtest_df["market_home_probability"] = market_home_probability
    backtest_df["market_away_probability"] = market_away_probability
    backtest_df["market_home_odds"] = backtest_df["market_home_probability"].apply(probability_to_american)
    backtest_df["market_away_odds"] = backtest_df["market_away_probability"].apply(probability_to_american)

    backtest_df["implied_home_probability"] = backtest_df["market_home_odds"].apply(american_to_probability)
    backtest_df["implied_away_probability"] = backtest_df["market_away_odds"].apply(american_to_probability)

    backtest_df["edge_home"] = backtest_df["model_probability_home"] - backtest_df["implied_home_probability"]
    backtest_df["edge_away"] = backtest_df["model_probability_away"] - backtest_df["implied_away_probability"]

    backtest_df["ev_home"] = backtest_df.apply(
        lambda row: calculate_ev(row["model_probability_home"], row["market_home_odds"]),
        axis=1,
    )
    backtest_df["ev_away"] = backtest_df.apply(
        lambda row: calculate_ev(row["model_probability_away"], row["market_away_odds"]),
        axis=1,
    )

    home_candidate = (backtest_df["edge_home"] >= min_edge) & (backtest_df["ev_home"] >= min_ev)
    away_candidate = (backtest_df["edge_away"] >= min_edge) & (backtest_df["ev_away"] >= min_ev)

    backtest_df["bet_side"] = np.select(
        [home_candidate & (backtest_df["ev_home"] >= backtest_df["ev_away"]), away_candidate],
        ["home", "away"],
        default="none",
    )
    backtest_df["bet"] = (backtest_df["bet_side"] != "none").astype(int)

    chosen_odds = np.where(backtest_df["bet_side"] == "home", backtest_df["market_home_odds"], backtest_df["market_away_odds"])
    home_won = backtest_df["home_win"] == 1
    bet_won = np.where(backtest_df["bet_side"] == "home", home_won, ~home_won)

    payout = np.where(chosen_odds > 0, chosen_odds / 100, 100 / np.abs(chosen_odds))
    backtest_df["bet_result"] = np.where(backtest_df["bet"] == 0, 0, np.where(bet_won, payout, -1))

    total_bets = int(backtest_df["bet"].sum())
    total_profit = float(backtest_df["bet_result"].sum())
    roi = total_profit / total_bets if total_bets > 0 else 0.0

    return backtest_df, total_bets, total_profit, roi
