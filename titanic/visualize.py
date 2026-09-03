"""数据理解可视化(保存到 outputs/figures)。

每张图都刻意保持简洁(课件风格)。标题/坐标轴标签/图例都是双语
(English + 简体中文)，这样同一张 PNG 在英文页和中文页都能用。
程序会自动探测当前机器上的中文字体(雅黑 / 文泉驿 / Noto CJK /
思源黑体 / PingFang …)；如果找不到，就丢弃中文部分、图表只显示
英文——保证在任何电脑上都不会出现方块乱码。

图表清单(镜像课件"数据理解"步骤)：
    1_survived_counts       - 目标分布：生还 vs 未生还人数
    2_missing_values        - 各列缺失值数量(条形图)
    3_sex_survival          - 按性别看生还率
    4_pclass_survival       - 按舱位等级看生还率
    5_age_distribution      - 年龄分布(按是否生还，直方图)
    6_fare_distribution     - 票价分布，log(1+Fare) 变换
    7_model_comparison      - 模型对比：五指标分组条形图
    8_roc_curves            - 四个模型的 ROC 曲线(AUC 在图例)

图会由 train.py 在整条管道里(重新)生成，也可以单独运行：
  python -m titanic.visualize
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # 无头后端：不需要显示器，直接保存成文件
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config
from .preprocessing import load_data

# ---------------------------------------------------------------------------
# 中文字体处理 —— 在任何电脑上都不打印方块乱码
# ---------------------------------------------------------------------------

# 候选中文字体名，大致按各平台的常见程度排序。
_CJK_CANDIDATES = [
    "Microsoft YaHei",          # Windows
    "SimHei", "SimSun",         # Windows 备选
    "PingFang SC",              # macOS
    "Heiti SC", "STHeiti",
    "Noto Sans CJK SC",         # Linux / 常见
    "Noto Sans SC",
    "Source Han Sans SC",
    "Source Han Sans CN",
    "WenQuanYi Zen Hei",        # Linux 备选
    "WenQuanYi Micro Hei",
    "AR PL UMing CN",
]
_CJK_READY: bool | None = None   # 缓存：是否找到了可用的中文字体


def _setup_cjk_font() -> bool:
    """注册一个可用的中文字体；能画中文就返回 True。"""
    global _CJK_READY
    if _CJK_READY is not None:
        return _CJK_READY
    from matplotlib import font_manager

    installed = {f.name for f in font_manager.fontManager.ttflist}
    chosen = next((n for n in _CJK_CANDIDATES if n in installed), None)
    if chosen is None:                       # 已知字体一个都不存在
        _CJK_READY = False
        return False
    # matplotlib >=3.7: rcParams["font.family"] 接受字体列表；
    # 把中文字体排在默认 sans 栈后面，拉丁字符仍用默认字体。
    plt.rcParams["font.sans-serif"] = [chosen, "DejaVu Sans",
                                       "Arial", "Helvetica", "sans-serif"]
    plt.rcParams["axes.unicode_minus"] = False  # 负号正常显示
    _CJK_READY = True
    print(f"[visualize] CJK font in use: {chosen}")
    return True


def zh(text: str) -> str:
    """只有中文字体可用时才返回 ``text``(否则返回空串)。

    供 :func:`_b` 使用，保证在没有中文字体的电脑上不出方块。
    """
    return text if _CJK_READY else ""


def _b(en: str, zh_cn: str) -> str:
    """双语标签：中文字体可用时显示 'English\\n中文'，否则只显示英文。"""
    return f"{en}\n{zh_cn}" if _CJK_READY else en


# ---------------------------------------------------------------------------
# 通用辅助
# ---------------------------------------------------------------------------

def _save(fig, name: str) -> Path:
    """把图保存为 PNG 并关闭，释放内存。"""
    out = config.FIG_DIR / f"{name}.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# 单张图(标签都是双语 EN / 中文)
# ---------------------------------------------------------------------------

def chart_survived_counts(df: pd.DataFrame):
    """条形图：生还(1) vs 未生还(0) 的人数。"""
    counts = df["Survived"].value_counts().sort_index()  # 索引 0,1
    fig, ax = plt.subplots(figsize=(5.2, 4))
    bars = ax.bar([_b("Not survived", "未生还"), _b("Survived", "生还")],
                  counts, color=["#d95f5f", "#5fa8d9"])
    ax.bar_label(bars)  # 在每根柱子顶上标出人数
    ax.set_ylabel(_b("Number of passengers", "人数"))
    ax.set_title(_b("Target balance: survived vs not survived",
                     "目标变量分布：生还与未生还人数"))
    ax.grid(axis="y", alpha=0.3)
    return _save(fig, "1_survived_counts")


def chart_missing_values(df: pd.DataFrame):
    """条形图：每列缺失值(NaN)数量。"""
    missing = df.isna().sum()
    missing = missing[missing > 0].sort_values()  # 只保留有缺失的列
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.barh(missing.index, missing.values, color="#e0a458")  # 横向条形
    for i, v in enumerate(missing.values):
        ax.text(v + 4, i, str(v), va="center")
    ax.set_xlabel(_b("Number of missing values", "缺失值数量"))
    ax.set_title(_b("Missing values per column", "各字段缺失值统计"))
    return _save(fig, "2_missing_values")


def _survival_rate_by_group(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """分组生还率表：`col` 每个取值对应的生还率。"""
    grouped = df.groupby(col)["Survived"].agg(["mean", "count"])  # 比例 + 人数
    grouped["mean"] = grouped["mean"] * 100  # 小数 -> 百分比
    return grouped.sort_values("mean", ascending=False)


def chart_sex_survival(df: pd.DataFrame):
    """条形图：女性 vs 男性乘客的生还率。"""
    rates = _survival_rate_by_group(df, "Sex")
    # 保持双语刻度顺序稳定：存在时女性在前
    tick_labels = {"female": _b("Female", "女"), "male": _b("Male", "男")}
    labels = [tick_labels.get(str(v), str(v)) for v in rates.index]
    fig, ax = plt.subplots(figsize=(5.2, 4))
    bars = ax.bar(labels, rates["mean"],
                  color=["#5fa8d9", "#d95f5f"])
    ax.bar_label(bars, fmt="%.1f%%")  # 柱顶标百分比
    ax.set_ylabel(_b("Survival rate (%)", "生还率(%)"))
    ax.set_title(_b("Survival rate by sex", "性别与生还率"))
    ax.set_ylim(0, 105)
    ax.grid(axis="y", alpha=0.3)
    return _save(fig, "3_sex_survival")


def chart_pclass_survival(df: pd.DataFrame):
    """条形图：一等/二等/三等舱乘客的生还率。"""
    rates = _survival_rate_by_group(df, "Pclass")
    labels = [_b(f"Class {v}", f"{v} 等舱") for v in rates.index]
    fig, ax = plt.subplots(figsize=(5.2, 4))
    bars = ax.bar(labels, rates["mean"], color="#7fb37f")
    ax.bar_label(bars, fmt="%.1f%%")
    ax.set_ylabel(_b("Survival rate (%)", "生还率(%)"))
    ax.set_title(_b("Survival rate by passenger class", "舱位等级与生还率"))
    ax.set_ylim(0, 105)
    ax.grid(axis="y", alpha=0.3)
    return _save(fig, "4_pclass_survival")


def chart_age_distribution(df: pd.DataFrame):
    """直方图：生还者 vs 未生还者的年龄分布(两组叠加)。"""
    survived = df.loc[df["Survived"] == 1, "Age"].dropna()
    not_survived = df.loc[df["Survived"] == 0, "Age"].dropna()
    fig, ax = plt.subplots(figsize=(6.4, 4))
    ax.hist([not_survived, survived], bins=20, stacked=False, alpha=0.55,
            label=[_b("Not survived", "未生还"), _b("Survived", "生还")],
            color=["#d95f5f", "#5fa8d9"])
    ax.set_xlabel(_b("Age", "年龄"))
    ax.set_ylabel(_b("Number of passengers", "人数"))
    ax.set_title(_b("Age distribution by survival", "年龄分布(按是否生还)"))
    ax.legend()
    return _save(fig, "5_age_distribution")


def chart_fare_distribution(df: pd.DataFrame):
    """直方图：票价按 log(1+Fare) 尺度(课件推荐做法)。

    票价分布严重右偏，所以画 log(1+Fare)。+1 保证最便宜的票
    (Fare == 0) 落在 0 而不是 -inf。
    """
    log_fare = np.log1p(df["Fare"].dropna())  # ln(1 + Fare)
    fig, ax = plt.subplots(figsize=(6.4, 4))
    ax.hist(log_fare, bins=40, color="#9b8ec4")
    ax.set_xlabel(_b("log(1 + Fare)", "票价对数 log(1+Fare)"))
    ax.set_ylabel(_b("Number of passengers", "人数"))
    ax.set_title(_b("Fare distribution on log(1+Fare) scale",
                     "票价分布(log(1+Fare) 变换)"))
    ax.set_xlim(0, None)  # log(1+Fare) 恒 >= 0
    return _save(fig, "6_fare_distribution")


# ---------------------------------------------------------------------------
# 模型对比图(为课件 p.40 / p.44 新增)
# ---------------------------------------------------------------------------

# id -> 指标双语标签(只有存在中文字体时才画中文)
_METRIC_LABELS = {
    "accuracy": ("Accuracy", "准确率"),
    "precision": ("Precision", "精确率"),
    "recall": ("Recall", "召回率"),
    "f1": ("F1", "F1 分数"),
    "auc": ("AUC", "AUC"),
}

# id -> 该模型在所有对比图里用的颜色
_MODEL_COLORS = {
    "logistic": "#5fa8d9",      # 蓝
    "svm": "#e0a458",           # 金
    "decision_tree": "#7fb37f",  # 绿
    "random_forest": "#d98c8c",  # 浅红
}


def _model_short_label(model_id: str) -> str:
    """极短的刻度标签：LR / SVM / DT / RF。"""
    return {"logistic": "LR", "svm": "SVM",
            "decision_tree": "DT", "random_forest": "RF"}.get(model_id, model_id)


def save_model_comparison_chart(metrics_df: pd.DataFrame,
                                roc_series: dict | None = None) -> list[Path]:
    """画两张课件对比图之一，并返回 PNG 路径列表。

    - ``roc_series=None``: 分组条形图(课件 p.40)——每个指标
      (accuracy / precision / recall / F1 / AUC) 一组四根柱子，
      每个模型一根。五个指标并列，一眼就能看出精确率-召回率的取舍。
    - 给了 ``roc_series``: ROC 曲线(课件 p.44)——每个模型一条曲线，
      外加随机机会对角线；AUC 打印在图例里。
    """
    if roc_series is not None:
        return [_save(_draw_roc(roc_series), "8_roc_curves")]
    return [_save(_draw_metric_bars(metrics_df), "7_model_comparison")]


def _draw_metric_bars(metrics_df: pd.DataFrame) -> "plt.Figure":
    """分组条形图：五个指标 x 四个模型(p.40)。"""
    keys = config.METRIC_KEYS
    # 与课件表格行顺序对齐：保持 config 里的模型顺序
    order = [mid for mid, _ in config.MODELS]
    frame = metrics_df.set_index("model_id").reindex(order)

    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    n_models = len(order)
    width = 0.8 / n_models                    # 每组总宽 0.8
    x = np.arange(len(keys))                  # 每个指标一组
    colors = [_MODEL_COLORS.get(mid, "#999") for mid in order]
    label_en = dict(config.MODELS)       # id -> 完整英文名
    label_zh = {
        "logistic": "逻辑回归", "svm": "支持向量机",
        "decision_tree": "决策树", "random_forest": "随机森林",
    }
    for i, mid in enumerate(order):
        vals = frame.loc[mid, keys].astype(float).values
        bars = ax.bar(x + (i - n_models / 2 + 0.5) * width, vals, width,
                      label=_b(label_en.get(mid, mid), label_zh.get(mid, mid)),
                      color=colors[i])
        # 每根柱子顶上的 3 位小数字标签
        ax.bar_label(bars, fmt="%.3f", fontsize=7, padding=1)

    ax.set_xticks(x)
    ax.set_xticklabels([_b(en, zh) for en, zh in
                        (_METRIC_LABELS[k] for k in keys)])
    ax.set_ylim(0, 1.06)
    ax.set_ylabel(_b("Score", "得分"))
    ax.set_title(_b("Model comparison on the test set (random_state=42)",
                    "四模型对比：测试集五指标(准确率/精确率/召回率/F1/AUC)"))
    ax.grid(axis="y", alpha=0.3)
    ax.legend(ncol=4, fontsize=9, framealpha=0.9)
    return fig


def _draw_roc(roc_series: dict) -> "plt.Figure":
    """ROC 曲线：每个模型一条线 + 图例里的 AUC(p.44)。"""
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    order = [mid for mid, _ in config.MODELS]
    for mid in order:
        if mid not in roc_series:
            continue
        fpr, tpr = roc_series[mid]
        auc_val = float(np.trapezoid(tpr, fpr))  # 与指标表里的值一致
        ax.plot(fpr, tpr, lw=1.8, color=_MODEL_COLORS.get(mid, "#555"),
                label=f"{_model_short_label(mid)} (AUC={auc_val:.4f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.6,
            label=_b("Chance", "随机基线"))
    ax.set_xlabel(_b("False positive rate", "假正率 FPR"))
    ax.set_ylabel(_b("True positive rate", "真正率 TPR"))
    ax.set_title(_b("ROC curves — all models",
                    "ROC 曲线：四模型概率排序能力对比"))
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=8.5)
    return fig


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------

def generate_all(data_path: Path | None = None) -> list[Path]:
    """重新生成全部可视化图，返回 PNG 路径列表。"""
    config.ensure_dirs()
    _setup_cjk_font()          # 只判定一次：要不要画中文标签
    df = load_data(data_path) if data_path else load_data()
    charts = [
        chart_survived_counts(df),
        chart_missing_values(df),
        chart_sex_survival(df),
        chart_pclass_survival(df),
        chart_age_distribution(df),
        chart_fare_distribution(df),
    ]
    return charts


if __name__ == "__main__":  # 支持: python -m titanic.visualize
    config.ensure_dirs()
    generated = generate_all()
    print(f"Generated {len(generated)} charts into {config.FIG_DIR}")
    for p in generated:
        print(" -", p.name)
