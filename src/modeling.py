from xgboost import XGBClassifier

# =====================================================
# TRAIN MODEL
# FIX #8: se agrega early_stopping_rounds para evitar overfitting.
#         El eval_set que ya existía ahora se usa efectivamente.
# =====================================================

def train_model(X_train, y_train, features):
    print("\nUSING FEATURES:\n")
    for feature in features:
        print(f"  {feature}")

    # --------------------------------------------------
    # Split interno de validación (últimos 10 % del train)
    # --------------------------------------------------
    split_idx = int(len(X_train) * 0.9)
    X_train_sub = X_train.iloc[:split_idx]
    X_val       = X_train.iloc[split_idx:]
    y_train_sub = y_train.iloc[:split_idx]
    y_val       = y_train.iloc[split_idx:]

    # --------------------------------------------------
    # Modelo
    # --------------------------------------------------
    model = XGBClassifier(
        n_estimators=500,
        max_depth=3,
        learning_rate=0.01,
        subsample=0.7,
        colsample_bytree=0.7,
        gamma=1,
        reg_lambda=2,
        eval_metric="logloss",
        # FIX #8: detiene el entrenamiento si logloss no mejora en 40 rondas
        early_stopping_rounds=40,
        random_state=42,
    )

    model.fit(
        X_train_sub,
        y_train_sub,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    best = model.best_iteration
    print(f"\nMejor iteración (early stopping): {best}")

    return model