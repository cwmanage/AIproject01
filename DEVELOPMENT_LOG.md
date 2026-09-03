# 开发日志 — Titanic 乘客生还预测与数据可视化

> 课程：机器学习 第02次课 统一案例
> 最后更新：2026-09-03（双语版）

---

## 一、项目目标（用户确认的需求）

1. **预测结果输出**：测试集（20%）上四个模型的「真实 vs 预测」对比 CSV
2. **结合方式**：Web 页面 + API —— 既能看图表，也能填表预测做课堂演示
3. **界面语言**：①图内标注用中英双语（一套图两种语言通用，避免每图双份）；②页面/按钮/文案支持中英文一键切换；③完整文档
4. **模型**：Logistic Regression / SVM / Decision Tree / Random Forest 四模型对比

## 二、课件复现规范（红线，全部遵守）

| 规范 | 实现 |
|---|---|
| 891 条全量训练 | `data/titanic.csv`（原始 891 行，无独立测试集） |
| 80% / 20% 划分 | `train_test_split(test_size=0.2)` |
| random_state=42 | 全项目统一种子 |
| stratify=y | 保持生还比例一致 |
| 四模型同一划分 | 一次划分，四个 pipeline 共享 |
| Age 中位数填补 | `SimpleImputer(strategy="median")` |
| Embarked 众数填补 | `SimpleImputer(strategy="most_frequent")` |
| Cabin 丢弃 | 不参与建模 |
| Pclass/Sex/Embarked One-Hot | `OneHotEncoder(handle_unknown="ignore")` |
| Age/SibSp/Parch/Fare 数值标准化 | `StandardScaler` |
| 防数据泄漏 | preprocessor 只在训练集 fit，测试集仅 transform |
| PassengerId 不建模 | 仅用于结果映射/回填 |
| 不只评 Accuracy | 同时输出 Precision/Recall/F1（生还率 38.4% 不均衡） |
| 票价可视化 | `log(1+Fare)` 对数变换 |

## 三、目录结构

```
D:\homework\AIproject01\
├── app.py                  # FastAPI 应用（首页/图表/预测 API，支持 ?lang=）
├── start.py                # 跨平台一键启动脚本（核心，Python 实现）
├── start.bat / start.ps1 / start.sh   # 三平台壳脚本（调 start.py）
├── pyproject.toml          # uv 项目依赖声明
├── requirements.txt        # 依赖清单（start.py 首次运行时自动生成）
├── uv.lock                 # 依赖锁定文件
├── smoke_test.py           # API + 双语页面冒烟测试（可重复运行）
├── USER_MANUAL.md          # 用户使用手册（面向使用/演示者）
├── README_启动步骤.md       # 启动步骤速查
├── data\
│   └── titanic.csv         # 原始数据集（891 行）
├── titanic\                # 核心包
│   ├── __init__.py
│   ├── config.py           # 常量：路径/列名/种子/模型注册表
│   ├── preprocessing.py    # 预处理 Pipeline（防泄漏）
│   ├── visualize.py        # 6 张可视化图表生成（中英双语标注）
│   ├── train.py            # 四模型训练+评估+CSV/模型落盘
│   └── i18n.py             # 中英文界面词条（UI 字典 + 图注字典）
├── templates\
│   └── index.html          # Web 演示页（双语，右上角切换）
├── outputs\
│   ├── figures\            # 6 张 PNG 图（可视化产物）
│   └── csv\                # 指标表 + 各模型预测对比 CSV
└── models\                 # 序列化模型 pipeline_*.joblib
```

## 四、实现进度与关键决策

### ✅ 已完成（经验证）

1. **依赖安装**（uv）：fastapi / uvicorn / pandas / numpy / scikit-learn / matplotlib / joblib / jinja2
2. **数据准备**：`data/titanic.csv` 复制自 `D:\Download\titanic.csv`
3. **核心包四个模块**全部带注释，逻辑清晰
4. **训练已跑通**（结果稳定，SVM 最优，与经典结果一致）：
   - Logistic Regression: acc=0.8045, prec=0.7931, rec=0.6667, f1=0.7244
   - SVM: acc=0.8156, prec=0.8214, rec=0.6667, f1=0.7360
   - Decision Tree: acc=0.8045, prec=0.7656, rec=0.7101, f1=0.7368
   - Random Forest: acc=0.7933, prec=0.7667, rec=0.6667, f1=0.7132
   - **Best model: svm (acc=0.8156)**，用于默认预测
5. **Web 应用（app.py）**已完成并全接口验证通过：
   - `GET /` 演示页：6 图 + 指标表 + 预测表单 + 测试集结果表 + API 说明
   - `GET /api/health` 健康检查
   - `GET /api/summary` 训练汇总（指标/图/测试标签）
   - `GET /api/predictions?model=` 测试集真实 vs 预测
   - `GET /api/predict?..` / `POST /api/predict` 单乘客预测
   - 启动时优先从 `outputs/`、`models/` 加载已训练产物（免重训）
6. **一键启动**：`start.py`（跨平台）+ bat/ps1/sh 壳

### 🔧 train.py 的三处修复记录

1. 移除 `reset_index(drop=True)`，保留原始索引映射 PassengerId（防错位）
2. `evaluate()` 清理 dict 中不存在的 `"confusion_matrix"` 键和 `.tolist()`，删除多余 import
3. 按 sklearn 1.9 建议，`SVC(probability=True)` → `CalibratedClassifierCV(SVC(), ensemble=False)`（消除 FutureWarning，probability 参数将在 1.11 移除）

### 🖼 图 6 票价分布修复记录

原实现用 `ax.hist(df["Fare"])` + `ax.set_xscale("log")`：Fare 含 0 值，对数轴下 0 被压到 1e-14 位置，左侧出现"满格矩形"渲染异常。
修复：改为 `np.log1p(df["Fare"])`（即 log(1+Fare)，严格贴合课件要求）后用普通线性直方图，0 票价安全映射为 0。图 6 现已为正常双峰右偏分布。图 1-5 经视觉校验全部正常。

### 🐛 预测接口参数大小写 bug 修复记录（2026-09-03）

**现象**：表单提交后概率恒为 15.5% 且不随输入变化。

**根因**：HTML 表单字段名是大写（`Pclass`/`Sex`/`Age`），而 `GET /api/predict` 的函数签名参数是小写（`pclass`/`sex`）。HTTP 查询参数大小写敏感，FastAPI 匹配不到 → 全部静默回退 Query 默认值（male/30岁/S港/Fare32），模型一直在预测同一个“默认乘客”。

**修复**：`predict_get` 改为接收 `Request`，直接读 `request.query_params`，用 `pick("pclass", "Pclass")` 模式**大小写兼容**取值；非法值（Pclass 不在 1-3、sex/embarked 不合法）返回 400。POST JSON 与 GET 大小写均正常。

**验证**：女性头等舱 25 岁 Fare110 C 港 → SURVIVED 87.16%；男性三等舱 60 岁 Fare8 Q 港 → NOT SURVIVED 33.35%；非法 sex → 400。

### ⚠️ 遗留说明（不影响运行）

- 原骨架遗留 `main.py` / `test_main.http` 已删除（2026-09-03 用户确认），无残留依赖
- `smoke_test.py` 为测试辅助脚本，可保留可删

### 🌐 中英文双语版改造记录（2026-09-03，用户三需求）

**需求**：① 中英文切换按钮（页面与图表都随语言）；② 消除写死路径（压缩包在另一台电脑解压即用）；③ 新增用户使用手册。

**设计决策**：matplotlib 渲染中文依赖系统中文字体（微软雅黑/Noto CJK 等），换机后字体不一定存在，故**不做每图 en/zh 两版**，而是每张图用中英双语标注（一套图两种语言通用）。

**四个改造点**：

1. **图表双语标注**（`titanic/visualize.py`）：
   - 新增 `_setup_cjk_font()`：遍历常见 CJK 字体（Microsoft YaHei / SimHei / Noto Sans CJK / Source Han Sans SC / PingFang / WenQuanYi 等），探测到即设为 rcParams 字体并置 `_CJK_READY=True`
   - 新增 `_b(en, zh)`：运行时选词——`_CJK_READY` 时拼 `"en\nzh"` 双行，否则仅英文（任何电脑不出方块）
   - 全部 6 张图的标题/轴标签/图例/刻度改走 `_b()`（如 `"Target balance: survived vs not survived\n目标变量分布：生还与未生还人数"`）
2. **页面 i18n**（新建 `titanic/i18n.py` + 重写 `templates/index.html`）：
   - `i18n.py` 提供 `UI` 中英字典（约 40 键）+ `CHART_CAPTIONS` 图注元组 + `pick(key, lang)`
   - `GET /?lang=en|zh`（Query 校验只收 en/zh），lang 传入模板，模板内 `{{ t('key') }}` 取词
   - 导航/标题/表格列名/表单字段/按钮/结果提示/API 说明全量双语；右上角切换按钮 JS 跳 `/?lang=其他` 并 localStorage 持久化
3. **JS 动态文案**：模板内嵌 `JS_UI = {en:{...}, zh:{...}}`，预测结果、表格加载、行状态（生还/未生还/是/否）均随当前语言
4. **路径可移植**：审计全项目运行文件（.py/.html/.bat/.sh/.toml/.txt）——**零绝对路径**（config.py 用 `Path(__file__).resolve().parent.parent` 推导根目录）；实际做了"新目录移植模拟"验证：复制运行文件到全新目录 → 建 venv → 装依赖 → 自动训练（SVM 81.56% 一致）→ 服务 UP、双语页/预测/179 行全通 ✅

**验证**：`smoke_test.py`（双语回归，EN/ZH 全量断言）PASS；6 图重生成后逐张视觉校验（中文无方块、无重叠、无溢出）；预测 API 无回归（女性头等舱 25 岁 → SURVIVED 87.16%）。

## 五、技术备注

- 环境：`.venv` Python 3.12.10（uv 管理）；系统 Python 3.11.10
- sklearn 1.9：`probability=True` 已弃用，改用 `CalibratedClassifierCV`
- Starlette 新签名：`TemplateResponse(request, name, context)`（旧式 `(name, context)` 已移除）
- Windows 端口占用排查：`Get-NetTCPConnection -LocalPort 8000 -State Listen` → `Stop-Process`
- 图表与页面均为中英双语：图内标注双语（一套图通用），页面文案按 `?lang=` 切换（`titanic/i18n.py` 词条）
- 字体探测：matplotlib 找不到 CJK 字体时图表自动回退英文-only，不会出现方块
- 端口/主机可配：`TITANIC_PORT`（默认 8000）、`TITANIC_HOST`（默认 127.0.0.1）
- 移植：运行代码零绝对路径，整个目录复制到任意位置即可运行；文档内路径仅作示例

## 六、后续可选优化（不在本次范围）

- 前端浏览器截图目视终验（策略限制未做，接口级验证已全通过）
- 增加混淆矩阵/ROC 曲线图（课件未要求）
- Docker 化部署（当前一键启动已覆盖本地演示需求）
- 压缩包体积优化（剔除 outputs/models/.venv 已支持，首次启动自动重建）

## 七、文档索引

| 文档 | 面向 | 内容 |
|---|---|---|
| `README_启动步骤.md` | 所有人 | 启动步骤速查（三平台一键启动 + 手动调试） |
| `USER_MANUAL.md` | 使用/演示者 | 功能说明、中英文切换、FAQ（镜像/端口/字体/重训） |
| `DEVELOPMENT_LOG.md`（本文件） | 开发者 | 目标、规范、技术决策、踩坑修复记录 |
| `smoke_test.py` | 测试 | API + 中英文页面冒烟测试（服务运行后执行，可带端口参数） |
