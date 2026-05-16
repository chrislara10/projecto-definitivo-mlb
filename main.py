from __future__ import annotations

import os

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score

from src.backtest import run_backtest
from src.config import (
    CACHE_DIR,
    DATA_DIR,
    END_DATE,
    FEATURES,
    PREDICTION_THRESHOLD,
    START_DATE,
    TARGET,
)
from src.data_loader import download_historical_games
from src.feature_engineering import (
    add_rolling_features,
    build_model_dataset,
    build_team_game_logs,
)
from src.modeling import train_model
from src.pitcher_features import build_pitcher_features, merge_pitcher_features


# =====================================================
# MAIN PIPELINE
# =====================================================

def main(force_download: bool = False) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    print("\n--- INICIANDO PIPELINE DE MLB ---")

    # =================================================
    # 1. DATA
    # =================================================
    print("\n[1/7] Descargando/cargando juegos históricos...")

    games_df = download_historical_games(
        START_DATE,
        END_DATE,
        cache_path=DATA_DIR / "historical_games.csv",
        force_download=force_download,
    )

    if games_df.empty:
        raise RuntimeError("No games were downloaded. Check date range or API response.")

    games_df["date"] = pd.to_datetime(games_df["date"])
    games_df = (
        games_df.sort_values(["date", "gamePk"])
        .drop_duplicates(subset=["gamePk"])
        .reset_index(drop=True)
    )

    print(f"Total juegos disponibles: {len(games_df)}")

    # =================================================
    # 2. TEAM FEATURES
    # =================================================
    print("\n[2/7] Construyendo estadísticas de equipos...")

    team_logs = build_team_game_logs(games_df)
    team_logs = add_rolling_features(team_logs)
    model_df = build_model_dataset(games_df, team_logs)

    # =================================================
    # 3. PITCHER FEATURES
    # =================================================
    print("\n[3/7] Construyendo estadísticas de pitchers...")

    pitcher_features = build_pitcher_features(
        games_df,
        start_date=START_DATE,
        end_date=END_DATE,
        cache_path=DATA_DIR / "pitcher_features.csv",
        force_download=force_download,
    )

    model_df = merge_pitcher_features(model_df, pitcher_features)

    # =================================================
    # 4. CLEAN DATASET
    # =================================================
    model_df = model_df.replace([np.inf, -np.inf], np.nan)

    required_columns = FEATURES + [TARGET]
    missing_columns = [col for col in required_columns if col not in model_df.columns]

    if missing_columns:
        raise RuntimeError(f"Missing required columns after feature engineering: {missing_columns}")

    model_df = (
        model_df.dropna(subset=required_columns)
        .sort_values(["date", "gamePk"])
        .reset_index(drop=True)
    )

    model_df.to_csv(DATA_DIR / "model_dataset.csv", index=False)

    print(f"Dataset final listo con forma: {model_df.shape}")

    if len(model_df) < 100:
        raise RuntimeError(
            "Dataset final demasiado pequeño. Revisa fechas, pitcher IDs o features faltantes."
        )

    # =================================================
    # 5. TEMPORAL SPLIT
    # =================================================
    print("\n[4/7] Dividiendo train/test temporal...")

    split_index = int(len(model_df) * 0.8)

    train_df = model_df.iloc[:split_index].copy()
    test_df = model_df.iloc[split_index:].copy()

    X_train = train_df[FEATURES]
    y_train = train_df[TARGET].astype(int)

    X_test = test_df[FEATURES]
    y_test = test_df[TARGET].astype(int)

    # =================================================
    # 6. TRAIN
    # =================================================
    print("\n[5/7] Entrenando modelo XGBoost...")

    model = train_model(X_train, y_train, FEATURES)

    # =================================================
    # 7. EVALUATE
    # =================================================
    print("\n[6/7] Evaluando rendimiento...")

    pred_probs = model.predict_proba(X_test)[:, 1]
    preds = (pred_probs > PREDICTION_THRESHOLD).astype(int)

    print(f"Accuracy: {accuracy_score(y_test, preds):.4f}")
    print(f"ROC AUC:  {roc_auc_score(y_test, pred_probs):.4f}")
    print(f"LogLoss:  {log_loss(y_test, pred_probs):.4f}")

    importance_df = pd.DataFrame(
        {
            "feature": FEATURES,
            "importance": model.feature_importances_,
        }
    ).sort_values("importance", ascending=False)

    importance_df.to_csv(DATA_DIR / "feature_importance.csv", index=False)

    # =================================================
    # 8. BACKTEST
    # =================================================
    print("\n[7/7] Ejecutando backtest...")

    backtest_df, total_bets, total_profit, roi = run_backtest(test_df, pred_probs)
    backtest_df.to_csv(DATA_DIR / "mlb_backtest_results.csv", index=False)

    print("\nRESULTADOS FINALES:")
    print(f"Total Apuestas: {total_bets}")
    print(f"Ganancia Total: {total_profit:.2f} unidades")
    print(f"ROI:            {roi:.4f}")

    if total_bets > 0:
        bet_rows = backtest_df["bet"] == 1
        win_rate = (backtest_df.loc[bet_rows, "home_win"] == 1).mean()
        print(f"Win Rate:       {win_rate:.4f}")

    print("\nProceso completado. Archivos guardados en /data.")


if __name__ == "__main__":
    force = os.getenv("FORCE_DOWNLOAD", "0") == "1"
    main(force_download=force)
