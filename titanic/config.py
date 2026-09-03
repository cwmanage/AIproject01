"""全局配置：文件路径与复现实验参数。

项目里用到的所有常量都集中在这个文件，改一处(比如换数据路径)不用
去翻业务代码。
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# 项目目录(基于本文件位置解析 -> 放到任何电脑都能运行)
# ---------------------------------------------------------------------------
# 项目根目录: .../AIproject01
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# 原始输入数据
DATA_DIR = PROJECT_ROOT / "data"
DATA_PATH = DATA_DIR / "titanic.csv"

# 所有生成产物(图 + CSV 结果)都输出到 outputs/
OUTPUT_DIR = PROJECT_ROOT / "outputs"
FIG_DIR = OUTPUT_DIR / "figures"      # 可视化图片
CSV_DIR = OUTPUT_DIR / "csv"          # 预测/指标结果表格

# Web 静态资源(FastAPI 使用)
STATIC_DIR = PROJECT_ROOT / "static"
TEMPLATE_DIR = PROJECT_ROOT / "templates"
TEMPLATE_PATH = TEMPLATE_DIR / "index.html"

# ---------------------------------------------------------------------------
# 复现设置(来自课件：四个模型共用同一个划分)
# ---------------------------------------------------------------------------
TEST_SIZE = 0.20            # 80% 训练 / 20% 测试
RANDOM_STATE = 42           # 固定随机种子 -> 结果可复现
STRATIFY = True             # 分层抽样：保证训练/测试集生还比例一致

# ---------------------------------------------------------------------------
# 数据列
# ---------------------------------------------------------------------------
TARGET_COL = "Survived"                 # 预测目标(0 = 未生还, 1 = 生还)
DROP_COLS = ["PassengerId", "Name", "Ticket", "Cabin"]  # 不参与建模的列
NUM_COLS = ["Age", "SibSp", "Parch", "Fare"]            # 数值特征
CAT_COLS = ["Pclass", "Sex", "Embarked"]                # 类别特征

# ---------------------------------------------------------------------------
# API / 模型元信息
# ---------------------------------------------------------------------------
# (模型id, 人类可读名称) -- 用于图、表格和网页展示
MODELS = [
    ("logistic", "Logistic Regression"),
    ("svm", "Support Vector Machine"),
    ("decision_tree", "Decision Tree"),
    ("random_forest", "Random Forest"),
]

# 对比表 / 分组条形图里展示的指标列。
# 这里的顺序决定所有地方的列顺序(对齐课件 p.40 的表格)。
METRIC_KEYS = ["accuracy", "precision", "recall", "f1", "auc"]

# 网页实时预测表单期望的字段顺序(Pclass/Sex/Age/...)
FORM_COLS = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]


def ensure_dirs() -> None:
    """创建所有需要写产物的目录(幂等，可重复调用)。"""
    for d in (DATA_DIR, FIG_DIR, CSV_DIR, STATIC_DIR, TEMPLATE_DIR):
        d.mkdir(parents=True, exist_ok=True)
