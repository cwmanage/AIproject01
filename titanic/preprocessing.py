"""Data preprocessing: numeric/categorical pipelines joined in one transformer.

Rules follow the courseware exactly:
  * Age        -> median imputation (numerical)
  * Embarked   -> most-frequent (mode) imputation (categorical)
  * Cabin      -> dropped (687/891 missing, not a model feature)
  * Numeric    -> StandardScaler (Age/SibSp/Parch/Fare)
  * Categorical-> One-Hot encoding, unknown levels tolerated (Pclass/Sex/Embarked)
  * No data leakage: every imputer/scaler is fit on the TRAINING set only,
    then applied to the test set (ColumnTransformer/Pipeline do this for us).

The *same* fitted transformer is later used by train.py for the test split and
by app.py to convert a web-form request into a model-ready feature vector.
"""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import config


def load_data(path=config.DATA_PATH) -> pd.DataFrame:
    """Load the raw Titanic CSV from disk."""
    df = pd.read_csv(path)
    return df


def build_preprocessor() -> ColumnTransformer:
    """Build the column-separated preprocessing transformer (unfitted).

    Returns:
        ColumnTransformer: numeric + categorical sub-pipelines stacked into a
        single object that knows which column feeds which pipeline.
    """
    # --- numeric sub-pipeline: median imputation -> standardisation ---------
    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),  # fill NaN with median
            ("scaler", StandardScaler()),                   # zero mean, unit variance
        ]
    )

    # --- categorical sub-pipeline: mode imputation -> one-hot encoding ------
    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),  # fill NaN with mode
            # handle_unknown="ignore": unseen category becomes all-zero vector
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    # --- combine both branches, each applied only to its own columns --------
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, config.NUM_COLS),
            ("cat", categorical_pipe, config.CAT_COLS),
        ]
    )
    return preprocessor


def make_features(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only the model features (drop id/text/noisy columns)."""
    return df.drop(columns=config.DROP_COLS + [config.TARGET_COL], errors="ignore")


def make_target(df: pd.DataFrame) -> pd.Series:
    """Extract the target column (Survived) as a Series."""
    return df[config.TARGET_COL]
