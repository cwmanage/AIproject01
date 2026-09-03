"""Train four classification models on the Titanic data and evaluate them.

End-to-end flow (mirrors the courseware "unified case" pipeline):
    1. Load data + build the shared preprocessing transformer
    2. Split 80/20 with random_state=42 and stratify=y (same split for all)
    3. Preprocess features (fit on train only -> no data leakage)
    4. Train Logistic Regression / SVM / Decision Tree / Random Forest
    5. Evaluate each on the test split (accuracy, precision, recall, F1)
    6. Save artifacts to outputs/:
         figures/   - 6 data-understanding charts (see visualize.py)
         csv/predictions_<model>.csv  - test-set true vs predicted
         csv/metrics_all_models.csv   - comparison table of every model
         csv/test_predictions_best.csv- best model's final submission table
         models/    - serialized pipeline (preprocessor + model) for the API

Run from the project root with:  python -m titanic.train
(It also regenerates all visualizations at the same time.)
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,  # area under the ROC curve (courseware p.38-40)
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from . import config
from .preprocessing import build_preprocessor, load_data, make_features, make_target
from .visualize import generate_all, save_model_comparison_chart

# ---------------------------------------------------------------------------
# model zoo: id -> (label, unfitted classifier)
# ---------------------------------------------------------------------------
MODEL_ZOO = {
    "logistic": (
        "Logistic Regression",
        # courseware p.29: max_iter=1000, random_state=42
        LogisticRegression(max_iter=1000, random_state=config.RANDOM_STATE),
    ),
    "svm": (
        "Support Vector Machine",
        # courseware p.31: SVC(kernel="rbf", probability=True, random_state=42)
        # sklearn 1.9+ deprecates SVC(probability=True); wrapping the SVC in a
        # calibrated classifier keeps predict_proba() for the ROC curve.
        CalibratedClassifierCV(SVC(random_state=config.RANDOM_STATE), ensemble=False),
    ),
    "decision_tree": (
        "Decision Tree",
        # courseware p.33: max_depth=4 (limits depth -> less overfitting)
        DecisionTreeClassifier(max_depth=4, random_state=config.RANDOM_STATE),
    ),
    "random_forest": (
        "Random Forest",
        # courseware p.35: n_estimators=300, max_depth=6
        RandomForestClassifier(
            n_estimators=300, max_depth=6, random_state=config.RANDOM_STATE
        ),
    ),
}


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def split_data(df: pd.DataFrame):
    """80/20 train/test split with fixed seed and stratified target.

    `stratify=y` keeps the survived ratio identical in both splits, so every
    model sees the same distribution (courseware reproduction setting).
    """
    X = make_features(df)
    y = make_target(df)
    stratify_arg = y if config.STRATIFY else None
    return train_test_split(
        X, y,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=stratify_arg,
    )


def evaluate(y_true: pd.Series, y_pred: np.ndarray, y_proba: np.ndarray | None = None) -> dict:
    """Compute the standard classification metrics for one model.

    ``y_proba`` is the probability of the positive class (survived). It is
    needed only for the AUC (area under the ROC curve, courseware p.38-40):
    AUC ranks the test rows by predicted probability and is therefore
    threshold-independent. ``None`` (a model without probabilities) -> NaN.
    """
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }
    if y_proba is not None:
        metrics["auc"] = roc_auc_score(y_true, y_proba)
    else:
        metrics["auc"] = float("nan")
    return metrics


def _fit_and_evaluate(name: str, clf, X_train, X_test, y_train, y_test):
    """Fit one model inside a pipeline (preprocessor fit on train only!)."""
    pipe = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", clf)])
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    # probability of the positive class -> used for AUC and the ROC curve
    y_proba = pipe.predict_proba(X_test)[:, 1]
    metrics = evaluate(y_test, y_pred, y_proba)
    return pipe, y_pred, y_proba, metrics


# ---------------------------------------------------------------------------
# main pipeline
# ---------------------------------------------------------------------------

# module-level cache: fitted pipelines reused by app.py via train_pipelines()
_pipelines: dict = {}
_meta: dict = {}
preprocessor = build_preprocessor()  # shared by every pipeline


def run_training() -> dict:
    """Execute the whole training pipeline; return structured results."""
    global preprocessor
    config.ensure_dirs()

    # -- 1. data ------------------------------------------------------------
    # NOTE: we deliberately keep the original row indices of X_test so that
    # `df.loc[X_test.index]` later maps every prediction back to the correct
    # PassengerId (the split is random, so index order is not 0..n-1).
    df = load_data()
    X_train, X_test, y_train, y_test = split_data(df)

    # -- 2. regenerate data-understanding visualizations ---------------------
    fig_paths = generate_all()  # uses the full dataset (understanding, not modelling)
    print(f"[1/4] Data loaded: {len(df)} rows; charts -> {config.FIG_DIR.name}/")

    # -- 3. train + evaluate every model -------------------------------------
    rows, preds_holder, probas_holder = [], {}, {}
    for name, (label, clf) in MODEL_ZOO.items():
        pipe, y_pred, y_proba, metrics = _fit_and_evaluate(
            name, clf, X_train, X_test, y_train, y_test
        )
        _pipelines[name] = pipe          # cache for the API
        preds_holder[name] = y_pred
        probas_holder[name] = y_proba
        rows.append(
            {
                "model_id": name,
                "model": label,
                **{k: round(v, 4) for k, v in metrics.items()},
            }
        )
        print(f"    {label:<24} acc={metrics['accuracy']:.4f}  "
              f"prec={metrics['precision']:.4f}  rec={metrics['recall']:.4f}  "
              f"f1={metrics['f1']:.4f}  auc={metrics['auc']:.4f}")

    # -- 4. persist artifacts ------------------------------------------------
    # 4a. prediction CSVs: PassengerId + true label + predicted label per model
    test_ids = df.loc[X_test.index, "PassengerId"]
    for name, y_pred in preds_holder.items():
        out = config.CSV_DIR / f"predictions_{name}.csv"
        pd.DataFrame(
            {
                "PassengerId": test_ids.values,
                "Survived_true": y_test.values,
                "Survived_pred": y_pred,
            }
        ).to_csv(out, index=False)
        print(f"    saved {out.name}")

    # 4b. metrics table (all models side by side, incl. AUC column)
    metrics_df = pd.DataFrame(rows)
    metrics_path = config.CSV_DIR / "metrics_all_models.csv"
    metrics_df.to_csv(metrics_path, index=False)
    print(f"[2/4] Metrics table -> {metrics_path.name}")

    # 4b2. model-comparison grouped bar chart + ROC curves (courseware p.40/44)
    #      - p.40 table -> grouped bars for the five metrics
    #      - p.44 ROC chart -> one ROC curve per model (AUC in the legend)
    fig_paths += save_model_comparison_chart(metrics_df)   # grouped bars
    roc_series = {}
    for name, y_proba in probas_holder.items():
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_series[name] = (fpr, tpr)
    fig_paths += save_model_comparison_chart(metrics_df, roc_series=roc_series)

    # 4c. best model (highest accuracy) -> final submission-style CSV
    best_id = metrics_df.loc[metrics_df["accuracy"].idxmax(), "model_id"]
    best_path = config.CSV_DIR / "test_predictions_best.csv"
    pd.DataFrame(
        {
            "PassengerId": test_ids.values,
            "Survived_true": y_test.values,
            "Survived_pred": preds_holder[best_id],
        }
    ).to_csv(best_path, index=False)
    print(f"[3/4] Best model: {best_id} (acc={metrics_df['accuracy'].max():.4f})")

    # 4d. serialized pipelines -> used later by the FastAPI app
    model_dir = config.PROJECT_ROOT / "models"
    model_dir.mkdir(exist_ok=True)
    for name, pipe in _pipelines.items():
        joblib.dump(pipe, model_dir / f"pipeline_{name}.joblib")
    joblib.dump(preprocessor, model_dir / "preprocessor.joblib")
    print(f"[4/4] Pipelines serialized -> {model_dir.name}/")

    _meta.update(
        {
            "n_rows": int(len(df)),
            "test_ids": test_ids.tolist(),
            "y_test": y_test.tolist(),
            "y_pred": {k: v.tolist() for k, v in preds_holder.items()},
            "figures": {Path(p).name: str(p) for p in fig_paths},
            "metrics": rows,
            "best_model_id": best_id,
        }
    )
    return _meta


# ---------------------------------------------------------------------------
# API-facing helpers (used by app.py)
# ---------------------------------------------------------------------------

def ensure_trained() -> dict:
    """Return cached training results; train once if the cache is empty."""
    if not _pipelines:
        run_training()
    return _meta


def get_pipeline(model_id: str) -> Pipeline:
    """Return a fitted pipeline by model id (trains first if needed)."""
    ensure_trained()
    return _pipelines[model_id]


def get_meta() -> dict:
    """Return the last training run's metadata (trains first if needed)."""
    ensure_trained()
    return _meta


if __name__ == "__main__":  # allow: python -m titanic.train
    run_training()
