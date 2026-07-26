"""Train and cross-validate models, then write a submission.

Usage:
    kaggle competitions download -c spaceship-titanic -p data/ --unzip
    python -m src.train
"""

from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

from src.features import engineer

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
SUBMISSIONS_DIR = BASE_DIR / "submissions"


def build_preprocessor(X: pd.DataFrame, for_lgbm: bool = False) -> ColumnTransformer:
    categorical = X.select_dtypes(include=["object", "bool"]).columns.tolist()
    numeric = [c for c in X.columns if c not in categorical]
    cat_encoder = (
        OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        if for_lgbm
        else OneHotEncoder(handle_unknown="ignore")
    )
    return ColumnTransformer(
        [
            ("num", SimpleImputer(strategy="median"), numeric),
            ("cat", Pipeline([
                ("impute", SimpleImputer(strategy="most_frequent")),
                ("encode", cat_encoder),
            ]), categorical),
        ]
    )


def main() -> None:
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    test_df = pd.read_csv(DATA_DIR / "test.csv")
    passenger_ids = test_df["PassengerId"]

    y = train_df["Transported"].astype(int)
    X = engineer(train_df.drop(columns=["Transported"]))
    X_test = engineer(test_df)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    models = {
        "hist_gb": Pipeline([
            ("prep", build_preprocessor(X)),
            ("model", HistGradientBoostingClassifier(random_state=42)),
        ]),
        "lightgbm": Pipeline([
            ("prep", build_preprocessor(X, for_lgbm=True)),
            ("model", lgb.LGBMClassifier(
                n_estimators=600, learning_rate=0.03, num_leaves=48,
                subsample=0.9, colsample_bytree=0.8, random_state=42, verbosity=-1,
            )),
        ]),
    }

    best_name, best_score = None, -np.inf
    for name, pipeline in models.items():
        scores = cross_val_score(pipeline, X, y, cv=cv, scoring="accuracy")
        print(f"{name}: CV accuracy {scores.mean():.4f} ± {scores.std():.4f}")
        if scores.mean() > best_score:
            best_name, best_score = name, scores.mean()

    print(f"\nBest model: {best_name} ({best_score:.4f}), fitting on full data")
    best = models[best_name].fit(X, y)

    SUBMISSIONS_DIR.mkdir(exist_ok=True)
    submission = pd.DataFrame({
        "PassengerId": passenger_ids,
        "Transported": best.predict(X_test).astype(bool),
    })
    out = SUBMISSIONS_DIR / f"submission_{best_name}.csv"
    submission.to_csv(out, index=False)
    print(f"Submission written to {out}")


if __name__ == "__main__":
    main()
