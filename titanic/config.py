"""Central configuration: file paths and reproduction settings.

All constants used across the project live here so that a single change
(e.g. a new data path) never requires touching business logic.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Project directories (resolved relative to this file -> works from anywhere)
# ---------------------------------------------------------------------------
# Project root: .../AIproject01
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Raw input data
DATA_DIR = PROJECT_ROOT / "data"
DATA_PATH = DATA_DIR / "titanic.csv"

# Every generated artifact (figures + CSV results) goes to outputs/
OUTPUT_DIR = PROJECT_ROOT / "outputs"
FIG_DIR = OUTPUT_DIR / "figures"      # visualization images
CSV_DIR = OUTPUT_DIR / "csv"          # prediction / metric result tables

# Web static assets (served by FastAPI)
STATIC_DIR = PROJECT_ROOT / "static"
TEMPLATE_DIR = PROJECT_ROOT / "templates"
TEMPLATE_PATH = TEMPLATE_DIR / "index.html"

# ---------------------------------------------------------------------------
# Reproduction settings (from the courseware: same split for all models)
# ---------------------------------------------------------------------------
TEST_SIZE = 0.20            # 80% train / 20% test
RANDOM_STATE = 42           # fixed seed -> reproducible results
STRATIFY = True             # keep the same survived ratio in train/test

# ---------------------------------------------------------------------------
# Data columns
# ---------------------------------------------------------------------------
TARGET_COL = "Survived"                 # what we predict (0 = died, 1 = survived)
DROP_COLS = ["PassengerId", "Name", "Ticket", "Cabin"]  # not used for modelling
NUM_COLS = ["Age", "SibSp", "Parch", "Fare"]            # numerical features
CAT_COLS = ["Pclass", "Sex", "Embarked"]                # categorical features

# ---------------------------------------------------------------------------
# API / model metadata
# ---------------------------------------------------------------------------
# (name, human readable label) -- label used in plots, tables and the web page
MODELS = [
    ("logistic", "Logistic Regression"),
    ("svm", "Support Vector Machine"),
    ("decision_tree", "Decision Tree"),
    ("random_forest", "Random Forest"),
]

# Metric columns shown in the comparison table / grouped bar chart.
# Order here drives the column order everywhere (courseware p.40 table).
METRIC_KEYS = ["accuracy", "precision", "recall", "f1", "auc"]

# Column order expected by the live-prediction form (Pclass/Sex/Age/...)
FORM_COLS = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]


def ensure_dirs() -> None:
    """Create every directory we write artifacts into (idempotent)."""
    for d in (DATA_DIR, FIG_DIR, CSV_DIR, STATIC_DIR, TEMPLATE_DIR):
        d.mkdir(parents=True, exist_ok=True)
