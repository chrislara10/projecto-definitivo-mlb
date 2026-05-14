from xgboost import XGBClassifier

# =====================================================
# TRAIN MODEL
# =====================================================

def train_model(
    X_train,
    y_train,
    features
):

    print("\nUSING FEATURES:\n")

    for feature in features:

        print(feature)

    # =================================================
    # SPLIT VALIDATION
    # =================================================

    split_idx = int(len(X_train) * 0.9)
    X_train_sub, X_val = X_train.iloc[:split_idx], X_train.iloc[split_idx:]
    y_train_sub, y_val = y_train.iloc[:split_idx], y_train.iloc[split_idx:]

    # =================================================
    # MODEL
    # =================================================

    model = XGBClassifier(

        n_estimators=500,

        max_depth=3,

        learning_rate=0.01,

        subsample=0.7,

        colsample_bytree=0.7,

        gamma=1,

        reg_lambda=2,

        eval_metric="logloss",

        random_state=42
    )

    # =================================================
    # TRAIN
    # =================================================

    model.fit(

        X_train_sub,
        y_train_sub,

        eval_set=[(X_val, y_val)],

        verbose=False
    )

    return model
