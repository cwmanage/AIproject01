"""训练四个分类模型并评估（泰坦尼克数据）。

端到端流程(镜像课件"统一案例"管道)：
    1. 加载数据 + 构建共享的预处理转换器
    2. 80/20 划分，random_state=42 且 stratify=y(所有模型共用同一划分)
    3. 预处理特征(只在训练集上 fit -> 无数据泄漏)
    4. 训练 Logistic Regression / SVM / Decision Tree / Random Forest
    5. 在测试集上评估(accuracy, precision, recall, F1, AUC)
    6. 保存产物到 outputs/：
         figures/   - 8 张数据理解/对比图(见 visualize.py)
         csv/predictions_<model>.csv  - 测试集真实 vs 预测
         csv/metrics_all_models.csv   - 四模型指标对比表
         csv/test_predictions_best.csv- 最优模型最终提交表
         models/    - 序列化的管道(预处理 + 模型)，供 API 使用

在项目根目录运行：  python -m titanic.train
(会同时重新生成全部可视化图。)
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
    roc_auc_score,  # ROC 曲线下面积(课件 p.38-40)
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
# 模型仓库: id -> (展示名称, 未训练的模型)
# ---------------------------------------------------------------------------
MODEL_ZOO = {
    "logistic": (
        "Logistic Regression",
        # 课件 p.29: max_iter=1000, random_state=42
        LogisticRegression(max_iter=1000, random_state=config.RANDOM_STATE),
    ),
    "svm": (
        "Support Vector Machine",
        # 课件 p.31: SVC(kernel="rbf", probability=True, random_state=42)
        # sklearn 1.9+ 已弃用 SVC(probability=True)；外面包一层校准分类器，
        # 仍然能拿到 predict_proba() 概率输出，供 ROC 曲线使用。
        CalibratedClassifierCV(SVC(random_state=config.RANDOM_STATE), ensemble=False),
    ),
    "decision_tree": (
        "Decision Tree",
        # 课件 p.33: max_depth=4 (限制深度 -> 减少过拟合)
        DecisionTreeClassifier(max_depth=4, random_state=config.RANDOM_STATE),
    ),
    "random_forest": (
        "Random Forest",
        # 课件 p.35: n_estimators=300, max_depth=6
        RandomForestClassifier(
            n_estimators=300, max_depth=6, random_state=config.RANDOM_STATE
        ),
    ),
}


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------

def split_data(df: pd.DataFrame):
    """80/20 训练/测试划分：固定随机种子 + 按目标分层。

    `stratify=y` 保证两份数据里生还者的比例一致，这样每个模型看到的
    分布都相同(课件复现规范)。
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
    """计算单个模型的标准分类指标。

    ``y_proba`` 是正类(生还)的预测概率。它只用于计算 AUC(ROC 曲线下
    面积，课件 p.38-40)：AUC 按预测概率给测试样本排序，因此不依赖
    阈值。``None``(拿不到概率的模型) -> AUC 记为 NaN。
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
    """在管道里训练一个模型(预处理只在训练集上 fit！)。"""
    pipe = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", clf)])
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    # 正类概率 -> 用于 AUC 和 ROC 曲线
    y_proba = pipe.predict_proba(X_test)[:, 1]
    metrics = evaluate(y_test, y_pred, y_proba)
    return pipe, y_pred, y_proba, metrics


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

# 模块级缓存：训练好的管道，app.py 通过 get_pipeline() 复用
_pipelines: dict = {}
_meta: dict = {}
preprocessor = build_preprocessor()  # 所有管道共用同一个预处理


def run_training() -> dict:
    """执行整个训练流程；返回结构化的结果。"""
    global preprocessor
    config.ensure_dirs()

    # -- 1. 数据 ------------------------------------------------------------
    # 注意：我们刻意保留 X_test 的原始行索引，这样后面用
    # `df.loc[X_test.index]` 能把每条预测映射回正确的 PassengerId
    # (划分是随机的，所以索引顺序并不是 0..n-1)。
    df = load_data()
    X_train, X_test, y_train, y_test = split_data(df)

    # -- 2. 重新生成数据理解可视化图 ----------------------------------------
    fig_paths = generate_all()  # 用全量数据(理解数据用，不是建模)
    print(f"[1/4] Data loaded: {len(df)} rows; charts -> {config.FIG_DIR.name}/")

    # -- 3. 训练 + 评估每个模型 ---------------------------------------------
    rows, preds_holder, probas_holder = [], {}, {}
    for name, (label, clf) in MODEL_ZOO.items():
        pipe, y_pred, y_proba, metrics = _fit_and_evaluate(
            name, clf, X_train, X_test, y_train, y_test
        )
        _pipelines[name] = pipe          # 缓存，供 API 使用
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

    # -- 4. 持久化产物 ------------------------------------------------------
    # 4a. 预测 CSV：每个模型 = PassengerId + 真实标签 + 预测标签
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

    # 4b. 指标表(四个模型并列，含 AUC 列)
    metrics_df = pd.DataFrame(rows)
    metrics_path = config.CSV_DIR / "metrics_all_models.csv"
    metrics_df.to_csv(metrics_path, index=False)
    print(f"[2/4] Metrics table -> {metrics_path.name}")

    # 4b2. 模型对比分组条形图 + ROC 曲线(课件 p.40/p.44)
    #      - p.40 表格 -> 五指标的分组条形图
    #      - p.44 ROC 图 -> 每个模型一条 ROC 曲线(AUC 标在图例)
    fig_paths += save_model_comparison_chart(metrics_df)   # 分组条形图
    roc_series = {}
    for name, y_proba in probas_holder.items():
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_series[name] = (fpr, tpr)
    fig_paths += save_model_comparison_chart(metrics_df, roc_series=roc_series)

    # 4c. 最优模型(准确率最高) -> 最终提交格式的 CSV
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

    # 4d. 序列化管道 -> 之后由 FastAPI 应用加载
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
# 供 API 调用的辅助函数(app.py 使用)
# ---------------------------------------------------------------------------

def ensure_trained() -> dict:
    """返回缓存的训练结果；缓存为空时先训练一次。"""
    if not _pipelines:
        run_training()
    return _meta


def get_pipeline(model_id: str) -> Pipeline:
    """按模型 id 返回训练好的管道(必要时先训练)。"""
    ensure_trained()
    return _pipelines[model_id]


def get_meta() -> dict:
    """返回最近一次训练的元信息(必要时先训练)。"""
    ensure_trained()
    return _meta


if __name__ == "__main__":  # 支持: python -m titanic.train
    run_training()
