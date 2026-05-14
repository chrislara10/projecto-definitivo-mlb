import pandas as pd
import numpy as np
import os
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss, brier_score_loss

# Importaciones de configuración y módulos locales
from src.config import (
    START_DATE, END_DATE, FEATURES, TARGET, PREDICTION_THRESHOLD,
    MIN_EDGE, MIN_EV, BOOKMAKER_MARGIN, MARKET_NOISE_STD, RANDOM_SEED, ODDS_API_KEY
)
from src.data_loader import download_historical_games_incremental
from src.feature_engineering import build_team_game_logs, add_rolling_features, build_model_dataset
from src.pitcher_features import build_pitcher_features, merge_pitcher_features
from src.modeling import train_model
from src.backtest import run_backtest
from src.odds_provider import fetch_mlb_h2h_odds
from src.odds_provider import download_historical_odds_incremental
from src.utils import american_to_probability, calculate_ev

# =====================================================
# 1. PREPARACIÓN DEL ENTORNO
# =====================================================
os.makedirs("data", exist_ok=True)

print("\n--- INICIANDO PIPELINE DE MLB ---")

# =====================================================
# 2. ADQUISICIÓN Y LIMPIEZA DE DATOS
# =====================================================
print("\n[1/7] Descargando juegos históricos...")
end_date = min(pd.Timestamp(END_DATE), pd.Timestamp.utcnow().normalize()).strftime("%Y-%m-%d")
games_df = download_historical_games_incremental(
    START_DATE,
    end_date,
    cache_path="data/historical_games.csv"
)
games_df["date"] = pd.to_datetime(games_df["date"])
games_df = games_df.sort_values("date").drop_duplicates(subset=["gamePk"])

games_df.to_csv("data/historical_games.csv", index=False)
print(f"Total juegos descargados: {len(games_df)}")

# =====================================================
# 3. INGENIERÍA DE CARACTERÍSTICAS (TEAMS)
# =====================================================
print("\n[2/7] Construyendo estadísticas de equipos...")
team_logs = build_team_game_logs(games_df)
team_logs = add_rolling_features(team_logs)
model_df = build_model_dataset(games_df, team_logs)

# =====================================================
# 4. INGENIERÍA DE CARACTERÍSTICAS (PITCHERS)
# =====================================================
print("\n[3/7] Construyendo estadísticas de pitchers...")
pitcher_features = build_pitcher_features(games_df)
pitcher_features.to_csv("data/pitcher_features.csv", index=False)

# Unión de datos de pitchers al dataset principal
model_df = merge_pitcher_features(model_df, pitcher_features)

# Limpieza de valores infinitos y nulos en columnas críticas
model_df = model_df.replace([np.inf, -np.inf], np.nan)
required_columns = FEATURES + [TARGET]
model_df = model_df.dropna(subset=required_columns).copy()

print(f"Dataset final listo con forma: {model_df.shape}")

# =====================================================
# 5. DIVISIÓN TEMPORAL (EVITAR LEAKAGE)
# =====================================================
# Usamos el 80% para entrenar y 20% para test (los juegos más recientes)
split_index = int(len(model_df) * 0.8)
train_df = model_df.iloc[:split_index].copy()
test_df = model_df.iloc[split_index:].copy()

X_train, y_train = train_df[FEATURES], train_df[TARGET]
X_test, y_test = test_df[FEATURES], test_df[TARGET]

# Enriquecer test con odds históricas reales (cache incremental)
odds_history_df = download_historical_odds_incremental(
    odds_api_key=ODDS_API_KEY,
    start_date=START_DATE,
    end_date=end_date,
    cache_path="data/odds_history.csv",
)
if len(odds_history_df) > 0:
    test_df["date"] = pd.to_datetime(test_df["date"]).dt.normalize()
    odds_history_df["date"] = pd.to_datetime(odds_history_df["date"]).dt.normalize()
    test_df = test_df.merge(
        odds_history_df.rename(
            columns={
                "market_home_odds": "real_market_home_odds",
                "market_away_odds": "real_market_away_odds",
            }
        ),
        on=["date", "home_team", "away_team"],
        how="left",
    )

# =====================================================
# 6. ENTRENAMIENTO DEL MODELO
# =====================================================
print("\n[4/7] Entrenando modelo XGBoost...")
# Nota: Asegúrate de que tu función train_model en src.modeling 
# esté configurada para manejar estos datos.
model = train_model(X_train, y_train, FEATURES)

# =====================================================
# 7. EVALUACIÓN Y PREDICCIONES
# =====================================================
print("\n[5/7] Evaluando rendimiento...")
pred_probs = model.predict_proba(X_test)[:, 1]
# Threshold de seguridad: solo predecimos victoria local si prob > 52%

preds = (
    pred_probs > PREDICTION_THRESHOLD
).astype(int)

print(f"Accuracy: {accuracy_score(y_test, preds):.4f}")
print(f"ROC AUC:  {roc_auc_score(y_test, pred_probs):.4f}")
print(f"LogLoss:  {log_loss(y_test, pred_probs):.4f}")
print(f"Brier:    {brier_score_loss(y_test, pred_probs):.4f}")

# Guardar importancia de variables
importance_df = pd.DataFrame({"feature": FEATURES, "importance": model.feature_importances_})
importance_df.sort_values("importance", ascending=False).to_csv("data/feature_importance.csv", index=False)

# =====================================================
# 8. BACKTEST Y RESULTADOS DE APUESTAS
# =====================================================
print("\n[6/7] Ejecutando Backtest...")
backtest_df, total_bets, total_profit, roi = run_backtest(
    test_df,
    pred_probs,
    min_edge=MIN_EDGE,
    min_ev=MIN_EV,
    margin=BOOKMAKER_MARGIN,
    noise_std=MARKET_NOISE_STD,
    random_seed=RANDOM_SEED
)

# Guardar resultados
backtest_df.to_csv("data/mlb_backtest_results.csv", index=False)

print(f"\nRESULTADOS FINALES:")
print(f"Total Apuestas: {total_bets}")
print(f"Ganancia Total: {total_profit:.2f} unidades")
print(f"ROI:            {roi:.4f}")

if total_bets > 0:
    settled = backtest_df[backtest_df["bet"] == 1]
    win_rate = (settled["bet_result"] > 0).mean()
    home_bets = (settled["bet_side"] == "home").mean()
    print(f"Win Rate:       {win_rate:.4f}")
    print(f"% Bets Home:    {home_bets:.4f}")

    # ==============================================
    # 9. APUESTAS DETECTADAS EN LA JORNADA MÁS RECIENTE
    # ==============================================
    last_date = pd.to_datetime(backtest_df["date"]).max()
    today_bets = backtest_df[
        (backtest_df["bet"] == 1) &
        (pd.to_datetime(backtest_df["date"]) == last_date)
    ][[
        "date",
        "away_team",
        "home_team",
        "bet_side",
        "model_probability_home",
        "model_probability_away",
        "market_home_odds",
        "market_away_odds",
        "edge_home",
        "edge_away",
        "ev_home",
        "ev_away",
    ]].copy()

    today_bets = today_bets.sort_values(
        by=["ev_home", "ev_away"],
        ascending=False
    )

    # Si hay API key, intentamos reemplazar odds sintéticas por odds reales del día
    real_odds_df = fetch_mlb_h2h_odds(ODDS_API_KEY)
    if len(real_odds_df) > 0:
        today_bets["date"] = pd.to_datetime(today_bets["date"]).dt.normalize()
        today_bets = today_bets.merge(
            real_odds_df,
            on=["date", "home_team", "away_team"],
            how="left",
            suffixes=("", "_real"),
        )
        today_bets["market_home_odds"] = today_bets["market_home_odds_real"].combine_first(today_bets["market_home_odds"])
        today_bets["market_away_odds"] = today_bets["market_away_odds_real"].combine_first(today_bets["market_away_odds"])
        today_bets = today_bets.drop(columns=["market_home_odds_real", "market_away_odds_real"])

        today_bets["implied_home_probability"] = today_bets["market_home_odds"].apply(american_to_probability)
        today_bets["implied_away_probability"] = today_bets["market_away_odds"].apply(american_to_probability)
        today_bets["edge_home"] = today_bets["model_probability_home"] - today_bets["implied_home_probability"]
        today_bets["edge_away"] = today_bets["model_probability_away"] - today_bets["implied_away_probability"]
        today_bets["ev_home"] = today_bets.apply(
            lambda row: calculate_ev(row["model_probability_home"], row["market_home_odds"]),
            axis=1,
        )
        today_bets["ev_away"] = today_bets.apply(
            lambda row: calculate_ev(row["model_probability_away"], row["market_away_odds"]),
            axis=1,
        )
    today_bets.to_csv("data/today_bets.csv", index=False)

    print(f"\nApuestas detectadas en la jornada más reciente ({last_date.date()}): {len(today_bets)}")
    if len(today_bets) > 0:
        print(today_bets.to_string(index=False))
    else:
        print("No se detectaron apuestas para la jornada más reciente.")
    print("Archivo generado: data/today_bets.csv")

print("\n[7/7] Proceso completado. Archivos guardados en /data.")
