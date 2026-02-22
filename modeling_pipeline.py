"""End-to-end modeling pipeline for insurance conversion propensity.

Steps:
1. EDA and quality checks
2. Preprocessing
3. Baseline + advanced models
4. Balanced Random Forest
5. LightGBM + Optuna hyperparameter tuning

Run:
    python modeling_pipeline.py
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import optuna
import pandas as pd
from imblearn.ensemble import BalancedRandomForestClassifier
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")

DATA_PATH = "data_(2).csv"
RANDOM_STATE = 42
TARGET = "has_sale"
ID_COLS = ["id"]
TIME_COL = "dt"


@dataclass
class ModelResult:
    name: str
    pr_auc: float
    roc_auc: float


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df[TARGET] = df[TARGET].astype(int)
    return df


def run_eda(df: pd.DataFrame) -> None:
    print("=" * 80)
    print("DATA OVERVIEW")
    print("=" * 80)
    print(f"Shape: {df.shape}")
    print("\nMissingness (%):")
    print((df.isna().mean() * 100).sort_values(ascending=False).round(2))

    y = df[TARGET]
    print(f"\nTarget rate: {y.mean():.4f} ({y.sum()} / {len(y)})")

    for col in ["platform", "gender", "membership_level"]:
        tmp = (
            df.groupby(col, dropna=False)[TARGET]
            .agg(["count", "mean"])
            .sort_values("count", ascending=False)
        )
        print(f"\nConversion by {col}:")
        print(tmp)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # Date-derived behavior features
    dt = pd.to_datetime(out[TIME_COL])
    out["day_of_week"] = dt.dt.dayofweek
    out["is_weekend"] = (out["day_of_week"] >= 5).astype(int)
    out["day_of_month"] = dt.dt.day

    # Ratio features
    out["cost_income_ratio"] = out["monthly_cost"] / out["household_income"].replace(0, np.nan)
    out["session_per_age"] = out["session_time"] / out["age"].replace(0, np.nan)

    return out


def build_preprocessor(df: pd.DataFrame) -> tuple[ColumnTransformer, list[str], list[str]]:
    feature_df = df.drop(columns=[TARGET, *ID_COLS, TIME_COL])

    num_cols = feature_df.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = feature_df.select_dtypes(exclude=["number"]).columns.tolist()

    num_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    cat_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipe, num_cols),
            ("cat", cat_pipe, cat_cols),
        ]
    )

    return preprocessor, num_cols, cat_cols


def evaluate_cv(model: Pipeline, X: pd.DataFrame, y: pd.Series, name: str) -> ModelResult:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    prob = cross_val_predict(model, X, y, cv=cv, method="predict_proba", n_jobs=-1)[:, 1]
    return ModelResult(
        name=name,
        pr_auc=average_precision_score(y, prob),
        roc_auc=roc_auc_score(y, prob),
    )


def tune_lgbm_with_optuna(X: pd.DataFrame, y: pd.Series, preprocessor: ColumnTransformer) -> dict:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 800),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 15, 120),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 120),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
            "class_weight": "balanced",
            "random_state": RANDOM_STATE,
            "n_jobs": -1,
            "verbosity": -1,
        }

        lgbm = Pipeline(
            steps=[("prep", preprocessor), ("model", LGBMClassifier(**params))]
        )
        prob = cross_val_predict(lgbm, X, y, cv=cv, method="predict_proba", n_jobs=-1)[:, 1]
        return average_precision_score(y, prob)

    study = optuna.create_study(direction="maximize", study_name="lgbm_pr_auc")
    study.optimize(objective, n_trials=40, show_progress_bar=False)
    return study.best_params


def main() -> None:
    df = load_data(DATA_PATH)
    run_eda(df)

    df = add_features(df)
    preprocessor, _, _ = build_preprocessor(df)

    X = df.drop(columns=[TARGET, *ID_COLS, TIME_COL])
    y = df[TARGET]

    # Simple holdout to report threshold recommendation
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    models = []

    logreg = Pipeline(
        steps=[
            ("prep", preprocessor),
            (
                "model",
                LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE),
            ),
        ]
    )
    models.append(("Logistic Regression", logreg))

    brf = Pipeline(
        steps=[
            ("prep", preprocessor),
            (
                "model",
                BalancedRandomForestClassifier(
                    n_estimators=500,
                    max_depth=None,
                    min_samples_leaf=2,
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    models.append(("Balanced Random Forest", brf))

    lgbm_base = Pipeline(
        steps=[
            ("prep", preprocessor),
            (
                "model",
                LGBMClassifier(
                    n_estimators=400,
                    learning_rate=0.05,
                    num_leaves=31,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                    verbosity=-1,
                ),
            ),
        ]
    )
    models.append(("LGBM (baseline)", lgbm_base))

    print("\n" + "=" * 80)
    print("CROSS-VALIDATION RESULTS (metric optimized: PR-AUC)")
    print("=" * 80)

    results = []
    for name, model in models:
        res = evaluate_cv(model, X, y, name)
        results.append(res)
        print(f"{name:28s} | PR-AUC: {res.pr_auc:.4f} | ROC-AUC: {res.roc_auc:.4f}")

    print("\nTuning LightGBM with Optuna...")
    best_params = tune_lgbm_with_optuna(X_train, y_train, preprocessor)
    print("Best params:", best_params)

    lgbm_tuned = Pipeline(
        steps=[
            ("prep", preprocessor),
            (
                "model",
                LGBMClassifier(
                    **best_params,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                    verbosity=-1,
                ),
            ),
        ]
    )

    lgbm_tuned.fit(X_train, y_train)
    test_prob = lgbm_tuned.predict_proba(X_test)[:, 1]
    test_pr_auc = average_precision_score(y_test, test_prob)
    test_roc_auc = roc_auc_score(y_test, test_prob)
    print(f"\nTuned LGBM holdout | PR-AUC: {test_pr_auc:.4f} | ROC-AUC: {test_roc_auc:.4f}")

    precision, recall, thresholds = precision_recall_curve(y_test, test_prob)
    f2 = (5 * precision * recall) / (4 * precision + recall + 1e-9)
    best_idx = np.nanargmax(f2)
    best_threshold = thresholds[max(best_idx - 1, 0)] if len(thresholds) > 0 else 0.5
    print(
        f"Best F2 threshold: {best_threshold:.3f} "
        f"(precision={precision[best_idx]:.3f}, recall={recall[best_idx]:.3f})"
    )


if __name__ == "__main__":
    main()
