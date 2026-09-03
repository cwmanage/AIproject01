"""数据预处理：数值/类别两个子管道，合并成一个转换器。

规则严格对齐课件：
  * Age        -> 中位数填补(数值)
  * Embarked   -> 众数填补(类别)
  * Cabin      -> 直接丢弃(缺失 687/891，不参与建模)
  * 数值列      -> StandardScaler 标准化(Age/SibSp/Parch/Fare)
  * 类别列      -> One-Hot 独热编码，容忍未知类别(Pclass/Sex/Embarked)
  * 无数据泄漏   -> 所有填补器/标准化器只在训练集上 fit，再应用到测试集
                  (ColumnTransformer/Pipeline 自动保证这一点)

这个"训练好的"转换器后面被 train.py(切测试集) 和 app.py(把网页表单
请求转成模型可用的特征向量) 共同复用。
"""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import config


def load_data(path=config.DATA_PATH) -> pd.DataFrame:
    """从磁盘加载原始 Titanic CSV 数据。"""
    df = pd.read_csv(path)
    return df


def build_preprocessor() -> ColumnTransformer:
    """构建按列分工的预处理转换器(尚未 fit)。

    Returns:
        ColumnTransformer: 数值 + 类别两个子管道叠加成的单一对象，
        它知道每一列该走哪条管道。
    """
    # --- 数值子管道：中位数填补 -> 标准化 ----------------------------------
    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),  # 用中位数填 NaN
            ("scaler", StandardScaler()),                   # 零均值、单位方差
        ]
    )

    # --- 类别子管道：众数填补 -> 独热编码 ----------------------------------
    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),  # 用众数填 NaN
            # handle_unknown="ignore": 没见过的类别会变成全零向量
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    # --- 合并两条分支，每列只走对应的管道 --------------------------------
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, config.NUM_COLS),
            ("cat", categorical_pipe, config.CAT_COLS),
        ]
    )
    return preprocessor


def make_features(df: pd.DataFrame) -> pd.DataFrame:
    """只保留模型特征(丢掉 id/文本/噪声列)。"""
    return df.drop(columns=config.DROP_COLS + [config.TARGET_COL], errors="ignore")


def make_target(df: pd.DataFrame) -> pd.Series:
    """取出目标列(Survived)作为 Series。"""
    return df[config.TARGET_COL]
