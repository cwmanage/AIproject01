# 🚢 泰坦尼克号乘客生还预测系统 — 用户使用手册

> 适用版本：2026-09-03（中英文双语界面版）
> 项目性质：机器学习课程作业（Titanic 生还预测 + 数据可视化 + Web 演示）

---

## 1. 这是什么

一个基于 **891 名泰坦尼克乘客数据** 的机器学习演示系统，包含：

- **6 张数据可视化图**：生还分布、缺失值、性别/舱位生还率、年龄分布、票价分布
- **4 个分类模型对比**：Logistic 回归 / SVM / 决策树 / 随机森林
  （同一 80/20 划分、random_state=42、分层抽样，严格复现课件规范）
- **网页交互演示**：填写乘客信息 → 实时预测是否生还（默认用最优模型 SVM）
- **REST API**：可供其他程序调用

界面支持 **中英文一键切换**（图表本身为双语标注，无需重新加载图片）。

---

## 2. 系统要求

| 项目 | 要求 |
|---|---|
| 操作系统 | Windows 10/11、macOS、Linux 均可 |
| Python | **3.12 或更高**（3.11 及以下无法安装本项目依赖） |
| 网络 | 首次运行需联网下载依赖包（约 200 MB，含 matplotlib/scikit-learn） |
| 浏览器 | Chrome / Edge / Firefox 等现代浏览器 |

> 无需安装 CUDA、无需 GPU、无需科学上网。

---

## 3. 快速开始（三平台一键启动）

### Windows

**方式 A：双击** `start.bat`

**方式 B：命令行**
```bat
cd 项目解压目录
python start.py
```

### macOS / Linux

```bash
cd 项目解压目录
chmod +x start.sh        # 仅首次需要
./start.sh
# 或 python3 start.py
```

### 启动过程（自动完成，无需干预）

```
[1/4] 检测 Python 版本            -> 低于 3.12 会提示
[2/4] 创建虚拟环境 .venv          -> 首次约 1 分钟
[3/4] 安装依赖                    -> 首次约 2-5 分钟（之后秒过）
[4/4] 启动 Web 服务               -> 自动等待就绪
```

> **首次运行**会额外执行一次模型训练 + 图表生成（约 10-30 秒），
> 产物保存在 `outputs/` 与 `models/`，之后启动直接加载、无需重训。

看到如下输出即表示成功：

```
[start.py] Server is UP. Press Ctrl+C to stop it.
```

浏览器打开 **http://127.0.0.1:8000** 即可使用。

---

## 4. 中英文切换

页面右上角有一个圆形按钮：

- 英文界面时显示 **「中文」** → 点击切换到中文界面
- 中文界面时显示 **「EN」** → 点击切换回英文界面

**切换范围（全部生效）**：

| 内容 | 是否随语言切换 |
|---|---|
| 页面导航 / 标题 / 说明文字 | ✅ |
| 模型指标表（列名、说明） | ✅ |
| 预测表单（字段名、选项） | ✅ |
| 预测结果提示 | ✅ |
| 测试结果表（状态文字） | ✅ |
| API 说明 | ✅ |
| **6 张图表** | ✅ 双语标注（中英同图，不换图） |
| 浏览器标签页标题 | ✅ |

> 语言选择会保存在浏览器中，刷新/再次打开保持上次选择。

---

## 5. 功能说明

### 5.1 数据可视化（第 1 区）

6 张图各自展示一个数据理解维度，每张图下方有一行中文（或英文）说明。

### 5.2 四模型对比（第 2 区）

表格展示 4 个模型在**测试集（179 人）**上的 Accuracy / Precision / Recall / F1。
⭐ 高亮的行 = 测试集准确率最高的模型（SVM，约 81.6%）。

### 5.3 交互预测（第 3 区）

按下面示例填写后点 **Predict Survival / 预测生还**：

| 字段 | 示例 | 说明 |
|---|---|---|
| 舱位等级 | 1st | 1/2/3 等舱 |
| 性别 | Female | 女/男 |
| 年龄 | 29 | 岁 |
| 兄弟姐妹/配偶 | 0 | 船上同行的兄弟姐妹或配偶数 |
| 父母/子女 | 0 | 船上同行的父母或子女人数 |
| 票价 | 110 | 英镑 |
| 登船港口 | C | S=南安普顿 C=瑟堡 Q=皇后镇 |
| 模型 | Best model | 可选四个模型或最优 |

结果示例：

> ✅ **预测生还**（SURVIVED）
> 模型: Support Vector Machine
> 生还概率: **88%** · 未生还概率: 12%

### 5.4 测试集结果（第 4 区）

选择模型 → Load Table，展示 179 名测试乘客的 PassengerId、真实结果、预测结果与是否正确。

### 5.5 REST API（第 5 区）

| 接口 | 说明 |
|---|---|
| `GET /api/health` | 健康检查 |
| `GET /api/summary` | 全部模型指标 + 图表清单 |
| `GET /api/predictions?model=svm` | 测试集真实 vs 预测（可换 logistic / decision_tree / random_forest / best） |
| `GET /api/predict?Pclass=1&Sex=female&Age=29&Fare=110&Embarked=C` | 查询参数预测（**大小写均可**） |
| `POST /api/predict` | JSON 请求体预测 |
| `/docs` | Swagger 交互式文档 |

JSON 预测示例：
```json
POST /api/predict
{"Pclass":1,"Sex":"female","Age":29,"SibSp":0,"Parch":0,"Fare":110,"Embarked":"C"}
```

---

## 6. 项目结构速览

```
项目解压目录/
├── start.py / start.bat / start.ps1 / start.sh   一键启动
├── app.py                    Web 服务主程序
├── pyproject.toml            依赖声明（uv）
├── data/titanic.csv          原始数据（891 行）
├── titanic/                  核心代码包
│   ├── config.py             配置（路径/列名/随机种子）
│   ├── preprocessing.py      预处理流水线（防数据泄漏）
│   ├── visualize.py          6 张图生成（双语）
│   ├── train.py              四模型训练 + 评估
│   └── i18n.py               中英文界面词条
├── templates/index.html      Web 页面
├── outputs/figures/          6 张 PNG 图表
├── outputs/csv/              指标与预测结果 CSV
└── models/                   4 个训练好的模型文件
```

> ⚠️ 上方的「项目解压目录」指你**实际解压到的位置**——本项目全部使用相对路径，
> **整个文件夹可以放在任意路径/任意电脑，解压即用**，无需改任何配置。

---

## 7. 常见问题（FAQ）

**Q1：双击 start.bat 一闪而过 / 无法运行**
→ 未安装 Python 或不在 PATH。请安装 **Python 3.12+**（安装时勾选 *Add Python to PATH*），然后在文件夹地址栏输入 `cmd` 回车，运行 `python start.py` 查看具体报错。

**Q2：提示 "Python 3.12+ is required"**
→ 你的 Python 版本过低。请安装 3.12 或更高版本后重试（3.11 无法安装 numpy 2.5 / pandas 3.0）。

**Q3：首次启动下载很慢 / 失败**
→ 依赖包较大。可配置国内镜像后重试：
```bash
# Windows (PowerShell)
$env:UV_DEFAULT_INDEX="https://pypi.tuna.tsinghua.edu.cn/simple"
python start.py
```
```bash
# macOS / Linux
export UV_DEFAULT_INDEX="https://pypi.tuna.tsinghua.edu.cn/simple"
python3 start.py
```

**Q4：端口 8000 被占用**
→ 换端口启动：
```bash
# Windows PowerShell
$env:TITANIC_PORT="8080"; python start.py
# macOS / Linux
TITANIC_PORT=8080 python3 start.py
```
然后访问 http://127.0.0.1:8080

**Q5：图表显示为方块（□□□）**
→ 系统缺少中文字体（极少见，仅 Linux 精简版可能出现）。安装任一中文字体即可：
`fonts-noto-cjk`（Debian/Ubuntu）或 `wqy-microhei`。安装后删除 `outputs/figures/` 内 PNG 再启动会自动重绘。

**Q6：想重新训练 / 重新出图**
→ 删除 `outputs/` 与 `models/` 目录后重新运行 start.py（会自动重建）。
或手动执行：
```bash
python -m titanic.train
```

**Q7：uv 相关的 hardlink 警告**
→ 无害提示，可忽略。

**Q8：语言按钮切换后概率/结果要重新预测吗？**
→ 不需要。切换只刷新页面文字，模型与数据不变。

---

## 8. 卸载 / 清理

直接删除整个项目文件夹即可（无系统级安装、无注册表残留）。
如需释放虚拟环境空间，删除 `.venv` 文件夹（约 500 MB）后下次启动会重建。

---

## 9. 结果文件说明（提交作业用）

| 文件 | 内容 |
|---|---|
| `outputs/csv/metrics_all_models.csv` | 四模型指标对比 |
| `outputs/csv/predictions_logistic.csv` 等 4 个 | 各模型测试集真实 vs 预测（179 行） |
| `outputs/csv/test_predictions_best.csv` | 最优模型预测结果 |
| `outputs/figures/1~6_*.png` | 6 张可视化图（双语） |
| `models/pipeline_*.joblib` | 训练好的模型（可复用） |

> 若压缩包不含 `outputs/`、`models/`（为减小体积已剔除），
> 首次启动会自动重新训练并生成，结果与文档一致（固定随机种子 42）。
