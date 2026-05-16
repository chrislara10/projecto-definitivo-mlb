from __future__ import annotations

import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier


# =====================================================
# TRAIN MODEL
# =====================================================

def train_model(X_train, y_train, features):
    """
    Train an XGBoost classifier with temporal validation.

    shuffle=False preserves time ordering and reduces leakage risk.
    """
    missing = [feature for feature in features if feature not in X_train.columns]

    if missing:
        raise ValueError(f"Missing training features: {missing}")

    print("\nUSING FEATURES:\n")
    for feature in features:
        print(feature)

    positives = int(np.sum(y_train == 1))
    negatives = int(np.sum(y_train == 0))
    scale_pos_weight = negatives / positives if positives > 0 else 1

    if len(X_train) >= 50:
        X_train_sub, X_val, y_train_sub, y_val = train_test_split(
            X_train,
            y_train,
            test_size=0.1,
            shuffle=False,
        )
        eval_set = [(X_val, y_val)]
    else:
        X_train_sub, y_train_sub = X_train, y_train
        eval_set = None

    model = XGBClassifier(
        n_estimators=500,
        max_depth=3,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        gamma=1,
        reg_lambda=3,
        objective="binary:logistic",
        eval_metric="logloss",
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_jobs=-1,
        tree_method="hist",
    )

    fit_kwargs = {"verbose": False}

    if eval_set is not None:
        fit_kwargs["eval_set"] = eval_set

    model.fit(X_train_sub, y_train_sub, **fit_kwargs)

    return model
