from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split

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

    X_train_sub, X_val, y_train_sub, y_val = train_test_split(

        X_train,
        y_train,

        test_size=0.1,

        shuffle=False
    )

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