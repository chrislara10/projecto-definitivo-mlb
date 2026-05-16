import pandas as pd
import numpy as np
import os
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss, brier_score_loss

from src.config import (
    START_DATE, END_DATE, FEATURES, TARGET, PREDICTION_THRESHOLD,
    MIN_EDGE, MIN_EV, BOOKMAKER_MARGIN, MARKET_NOISE_STD, RANDOM_SEED, ODDS_API_KEY,
    INITIAL_BANKROLL_USD, KELLY_FRACTION_CAP
)
from src.data_loader import download_historical_games_incremental
from src.feature_engineering import build_team_game_logs, add_rolling_features, build_model_dataset
from src.pitcher_features import build_pitcher_features, merge_pitcher_features
from src.modeling import train_model
from src.backtest import run_backtest
from src.odds_provider import fetch_mlb_h2h_odds
from src.utils import american_to_probability, calculate_ev, kelly_fraction

# =====================================================
# 1. PREPARACIÓN DEL ENTORNO
# =====================================================
os.makedirs("data", exist_ok=True)

print("\n--- INICIANDO PIPELINE DE MLB ---")
print("Versión: pipeline incremental + paper bets Kelly")

# =====================================================
# 2. ADQUISICIÓN Y LIMPIEZA DE DATOS
# =====================================================
print("\n[1/7] Descargando juegos históricos...")

# FIX: end_date limitado a hoy para no pedir fechas futuras
end_date = min(
    pd.Timestamp(END_DATE),
    pd.Timestamp.utcnow().normalize()
).strftime("%Y-%m-%d")

# FIX #10: download_historical_games_incremental ya guarda el cache
# internamente; el to_csv de main.py es ahora redundante pero inofensivo.
games_df = download_historical_games_incremental(
    START_DATE,
    end_date,
    cache_path="data/historical_games.csv"
)

games_df["date"] = pd.to_datetime(games_df["date"])
games_df = games_df.sort_values("date").drop_duplicates(subset=["gamePk"])
print(f"Total juegos: {len(games_df)}")

# =====================================================
# 3. INGENIERÍA DE CARACTERÍSTICAS (TEAMS)
# =====================================================
print("\n[2/7] Construyendo estadísticas de equipos...")
team_logs = build_team_game_logs(games_df)
team_logs = add_rolling_features(team_logs)
model_df  = build_model_dataset(games_df, team_logs)

# =====================================================
# 4. INGENIERÍA DE CARACTERÍSTICAS (PITCHERS)
# =====================================================
print("\n[3/7] Construyendo estadísticas de pitchers...")
pitcher_features = build_pitcher_features(games_df)
pitcher_features.to_csv("data/pitcher_features.csv", index=False)

model_df = merge_pitcher_features(model_df, pitcher_features)

model_df = model_df.replace([np.inf, -np.inf], np.nan)
required_columns = FEATURES + [TARGET]
model_df = model_df.dropna(subset=required_columns).copy()

print(f"Dataset final: {model_df.shape}")

# =====================================================
# 5. DIVISIÓN TEMPORAL
# =====================================================
split_index = int(len(model_df) * 0.8)
train_df = model_df.iloc[:split_index].copy()
test_df  = model_df.iloc[split_index:].copy()

X_train, y_train = train_df[FEATURES], train_df[TARGET]
X_test,  y_test  = test_df[FEATURES],  test_df[TARGET]

# =====================================================
# 6. ENTRENAMIENTO DEL MODELO
# =====================================================
print("\n[4/7] Entrenando modelo XGBoost...")
model = train_model(X_train, y_train, FEATURES)

# =====================================================
# 7. EVALUACIÓN
# =====================================================
print("\n[5/7] Evaluando rendimiento...")
pred_probs = model.predict_proba(X_test)[:, 1]
preds = (pred_probs > PREDICTION_THRESHOLD).astype(int)

print(f"Accuracy: {accuracy_score(y_test, preds):.4f}")
print(f"ROC AUC:  {roc_auc_score(y_test, pred_probs):.4f}")
print(f"LogLoss:  {log_loss(y_test, pred_probs):.4f}")
print(f"Brier:    {brier_score_loss(y_test, pred_probs):.4f}")

importance_df = pd.DataFrame({
    "feature": FEATURES,
    "importance": model.feature_importances_,
})
importance_df.sort_values("importance", ascending=False).to_csv(
    "data/feature_importance.csv", index=False
)

# =====================================================
# 8. BACKTEST
# =====================================================
print("\n[6/7] Ejecutando Backtest...")
backtest_df, total_bets, total_profit, roi = run_backtest(
    test_df,
    pred_probs,
    min_edge=MIN_EDGE,
    min_ev=MIN_EV,
    margin=BOOKMAKER_MARGIN,
    noise_std=MARKET_NOISE_STD,
    random_seed=RANDOM_SEED,
)

backtest_df.to_csv("data/mlb_backtest_results.csv", index=False)

print(f"\nRESULTADOS FINALES:")
print(f"Total Apuestas: {total_bets}")
print(f"Ganancia Total: {total_profit:.2f} unidades")
print(f"ROI:            {roi:.4f}")

if total_bets > 0:
    settled  = backtest_df[backtest_df["bet"] == 1]
    win_rate = (settled["bet_result"] > 0).mean()
    home_bets = (settled["bet_side"] == "home").mean()
    print(f"Win Rate:       {win_rate:.4f}")
    print(f"% Bets Home:    {home_bets:.4f}")

print("\n[7/7] Proceso completado. Archivos en /data.")

# =====================================================
# 9. APUESTAS DEL DÍA (PAPER TRADING + KELLY)
# =====================================================
print("\n[EXTRA] Generando apuestas del día (paper trading)...")

today_utc   = pd.Timestamp.utcnow().normalize()
live_odds_df = fetch_mlb_h2h_odds(ODDS_API_KEY)

if len(live_odds_df) == 0:
    print("No se encontraron odds del día o falta ODDS_API_KEY.")
else:
    live_odds_df["date"] = pd.to_datetime(live_odds_df["date"]).dt.normalize()
    today_odds = live_odds_df[live_odds_df["date"] == today_utc].copy()

    if len(today_odds) == 0:
        print("No hay juegos para hoy en el feed de odds.")
    else:
        # Últimas métricas por equipo
        latest_team = (
            team_logs.sort_values("date")
            .groupby("team")
            .tail(1)[["team", "runs_scored_last5", "runs_scored_last10",
                       "runs_allowed_last5", "win_rate_last10"]]
            .rename(columns={
                "runs_scored_last5":  "last_runs5",
                "runs_scored_last10": "last_runs10",
                "runs_allowed_last5": "last_ra5",
                "win_rate_last10":    "last_wr10",
            })
        )

        today_df = today_odds.merge(
            latest_team.rename(columns={"team": "home_team"}),
            on="home_team", how="left",
        ).merge(
            latest_team.rename(columns={"team": "away_team"}),
            on="away_team", how="left",
            suffixes=("_home", "_away"),
        )

        today_df["offense_edge_5"]  = today_df["last_runs5_home"]  - today_df["last_runs5_away"]
        today_df["offense_edge_10"] = today_df["last_runs10_home"] - today_df["last_runs10_away"]
        today_df["defense_edge_5"]  = today_df["last_ra5_away"]    - today_df["last_ra5_home"]
        today_df["momentum_edge"]   = today_df["last_wr10_home"]   - today_df["last_wr10_away"]

        # Columnas de pitcher: rellenar con mediana del train
        pitcher_cols = [
            "pitching_era_edge", "pitching_whip_edge", "pitching_fip_edge", "pitching_k9_edge",
            "away_pitcher_era", "away_pitcher_whip", "away_pitcher_fip", "away_pitcher_k9",
            "home_pitcher_era", "home_pitcher_whip", "home_pitcher_fip", "home_pitcher_k9",
        ]
        train_medians = train_df[FEATURES].median()

        # FIX #7: crear columnas faltantes de forma segura antes de indexar
        for col in pitcher_cols:
            if col not in today_df.columns:
                today_df[col] = train_medians.get(col, np.nan)

        # Rellenar NaN con mediana del train
        for feat in FEATURES:
            if feat not in today_df.columns:
                today_df[feat] = train_medians.get(feat, np.nan)
            today_df[feat] = today_df[feat].fillna(train_medians.get(feat, np.nan))

        today_df["model_probability_home"] = model.predict_proba(today_df[FEATURES])[:, 1]
        today_df["model_probability_away"] = 1 - today_df["model_probability_home"]

        today_df["edge_home"] = today_df["model_probability_home"] - today_df["market_home_odds"].apply(american_to_probability)
        today_df["edge_away"] = today_df["model_probability_away"] - today_df["market_away_odds"].apply(american_to_probability)
        today_df["ev_home"]   = today_df.apply(lambda r: calculate_ev(r["model_probability_home"], r["market_home_odds"]), axis=1)
        today_df["ev_away"]   = today_df.apply(lambda r: calculate_ev(r["model_probability_away"], r["market_away_odds"]), axis=1)

        home_ok = (today_df["edge_home"] >= MIN_EDGE) & (today_df["ev_home"] >= MIN_EV)
        away_ok = (today_df["edge_away"] >= MIN_EDGE) & (today_df["ev_away"] >= MIN_EV)

        today_df["bet_side"] = np.select(
            [home_ok & (today_df["ev_home"] >= today_df["ev_away"]), away_ok],
            ["home", "away"],
            default="none",
        )
        today_df = today_df[today_df["bet_side"] != "none"].copy()

        if len(today_df) == 0:
            print("No hay apuestas con edge/EV suficiente para hoy.")
        else:
            today_df["selected_odds"] = np.where(
                today_df["bet_side"] == "home",
                today_df["market_home_odds"],
                today_df["market_away_odds"],
            )
            today_df["selected_prob"] = np.where(
                today_df["bet_side"] == "home",
                today_df["model_probability_home"],
                today_df["model_probability_away"],
            )
            today_df["kelly_raw"]  = today_df.apply(
                lambda r: kelly_fraction(r["selected_prob"], r["selected_odds"]), axis=1
            )
            today_df["kelly_used"] = today_df["kelly_raw"].clip(upper=KELLY_FRACTION_CAP)
            today_df["stake_usd"]  = (INITIAL_BANKROLL_USD * today_df["kelly_used"]).round(2)
            today_df["track_status"] = "pending"

            track_cols = [
                "date", "home_team", "away_team", "bet_side",
                "selected_odds", "selected_prob",
                "edge_home", "edge_away", "ev_home", "ev_away",
                "kelly_raw", "kelly_used", "stake_usd", "track_status",
            ]
            today_df[track_cols].to_csv("data/daily_bets_to_track.csv", index=False)

            print("\nAPUESTAS DEL DÍA (PAPER):")
            print(
                today_df[[
                    "away_team", "home_team", "bet_side",
                    "selected_odds", "selected_prob",
                    "kelly_used", "stake_usd",
                ]].to_string(index=False)
            )
            print(f"\nBank inicial paper: ${INITIAL_BANKROLL_USD:.2f}")
            print("Tracking guardado en: data/daily_bets_to_track.csv")