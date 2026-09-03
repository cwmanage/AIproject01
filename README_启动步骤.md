# 🚢 Titanic 乘客生还预测 — 启动步骤

> 技术栈：Python 3.12+ · FastAPI · Scikit-learn · Matplotlib · uv
> 界面语言：中英文一键切换（默认英文，右上角切换）

---

## 方式一：一键启动（推荐）

**Windows**：双击 `start.bat`，或在 PowerShell/CMD 中：

```powershell
cd 项目解压目录        # 例如 cd D:\yourpath\AIproject01
python start.py
```

**macOS / Linux**：

```bash
cd 项目解压目录
chmod +x start.sh        # 首次执行一次
./start.sh
```

`start.py` 会自动完成（幂等，可反复运行）：
1. 检查 Python 版本（新建虚拟环境时需 **3.12+**）
2. 创建/复用 `.venv` 虚拟环境
3. 用 `uv sync` 安装全部依赖
4. 若 `outputs/` 无训练产物 → 自动运行 `python -m titanic.train`（出图+训练+CSV）
5. 启动 FastAPI 服务器并等待就绪

启动成功后浏览器访问：**http://127.0.0.1:8000**

> 本包使用**相对路径**，解压到任意位置均可运行，无需修改任何配置。

---

## 方式二：分步手动执行（开发调试用）

```powershell
cd 项目解压目录

# 1) 若还没有虚拟环境（需 Python 3.12+）
uv venv .venv                       # 或 python -m venv .venv

# 2) 安装依赖（二选一）
uv sync
# 或
.\.venv\Scripts\python -m pip install -r requirements.txt

# 3) 训练四模型 + 生成 6 张图 + 输出对比 CSV
.\.venv\Scripts\python -m titanic.train

# 4) 启动 Web 服务
.\.venv\Scripts\python -m uvicorn app:app --host 127.0.0.1 --port 8000 --reload
```

> 注意：所有命令用 `.venv\Scripts\python.exe`（Windows）或 `.venv/bin/python`（macOS/Linux）。
> 直接敲 `python` 可能命中系统旧版本而非项目 3.12。

---

## 启动后可以做什么

| 入口 | 说明 |
|---|---|
| **http://127.0.0.1:8000** | 演示主页：6 张数据可视化图 + 四模型指标对比表 + 交互式乘客预测表单 + 测试集 179 人真实vs预测表 + API 速查 |
| **http://127.0.0.1:8000/?lang=zh** | 中文界面版（也可点页面右上角按钮切换） |
| **http://127.0.0.1:8000/docs** | Swagger 交互式 API 文档（可在线测试每个接口） |
| **http://127.0.0.1:8000/redoc** | ReDoc 版 API 文档 |
| **http://127.0.0.1:8000/api/health** | 健康检查 `{"status":"ok"}` |
| **http://127.0.0.1:8000/api/summary** | 四模型全部指标 + 图表清单 JSON |
| **http://127.0.0.1:8000/api/predictions?model=svm** | 测试集对比（换 logistic/decision_tree/random_forest/best 均可） |
| **GET /api/predict?pclass=1&sex=female&age=30&fare=80&embarked=C** | 填参数预测生还 |

### 预测接口 POST 示例（JSON）

```bash
curl -X POST http://127.0.0.1:8000/api/predict ^
  -H "Content-Type: application/json" ^
  -d "{\"Pclass\":1,\"Sex\":\"female\",\"Age\":29,\"SibSp\":0,\"Parch\":0,\"Fare\":110,\"Embarked\":\"C\"}"
```

返回：

```json
{
  "model_id": "svm",
  "model_label": "Support Vector Machine",
  "survived": 1,
  "survived_probability": 0.8787,
  "not_survived_probability": 0.1213,
  "prediction_text": "SURVIVED"
}
```

---

## 常用调试命令

```powershell
# 单独重新出图（不重训模型）
.\.venv\Scripts\python -c "from titanic.visualize import generate_all; generate_all()"

# API 冒烟测试 + 页面双语回归（需服务已启动；EN/ZH 全量断言）
.\.venv\Scripts\python smoke_test.py


# Windows 端口被占时清理
Get-NetTCPConnection -LocalPort 8000 -State Listen | ForEach-Object { Stop-Process -Id $_.OwningProcess -Force }
```

---

## 常见问题

**Q1：双击 start.bat 一闪而过 / 提示 Python 3.12+ required？**
→ 说明 Python 版本不足或不在 PATH。本项目依赖 numpy 2.5 / pandas 3.0，**必须 Python 3.12+**（系统若只有 3.11 且无 .venv，start.py 会拒绝创建虚拟环境）。请安装 3.12+（勾选 Add to PATH）后重试；或在终端运行 `python start.py` 查看具体报错。

**Q2：页面图表不显示？**
→ 确认 `outputs\figures\` 下有 6 张 png；没有就先运行 `python -m titanic.train`。

**Q3：首页提示 metrics 空？**
→ 同上，首次需先训练。`start.py` 会自动处理，手动模式需先跑第 3 步。

**Q4：改端口？**
→ 设环境变量 `TITANIC_PORT=8888` 再运行 start.py；或 `uvicorn ... --port 8888`。
   跨设备访问：`TITANIC_HOST=0.0.0.0`（默认仅本机 127.0.0.1）。

**Q5：uv 安装报 hardlink 警告？**
→ 无害。缓存与目标在不同文件系统所致，可忽略，或设 `UV_LINK_MODE=copy`。

**Q6：页面/图表中文变方块？**
→ 系统缺中文字体（仅个别精简 Linux 出现）。安装 fonts-noto-cjk 或 wqy-microhei 后，
   删除 `outputs/figures/` 内 PNG 重新运行 start.py 自动重绘。本机 Windows 不受影响。

**Q7：想重新训练 / 重置结果？**
→ 删除 `outputs/` 与 `models/` 目录后重新运行 start.py（自动重建，随机种子 42 结果一致）。

---

## 交付产物清单

| 产物 | 位置 |
|---|---|
| 6 张可视化图（中英双语标注） | `outputs\figures\1~6_*.png` |
| 四模型指标汇总 | `outputs\csv\metrics_all_models.csv` |
| 各模型测试集预测对比 | `outputs\csv\predictions_{logistic,svm,decision_tree,random_forest}.csv` |
| 最优模型预测 CSV | `outputs\csv\test_predictions_best.csv` |
| 序列化模型 | `models\pipeline_*.joblib`（4 个）+ `preprocessor.joblib` |
| Web 演示 | 本机 http://127.0.0.1:8000 |

> 测试集 179 行（891×20%），CSV 含 PassengerId / Survived_true / Survived_pred，
> 与 PassengerId 一一对应可回溯原乘客。

---

## 详细文档

- **用户使用手册**（面向使用/演示者，含 FAQ 与提交作业说明）→ `USER_MANUAL.md`
- **开发日志**（面向开发者，含技术决策与踩坑记录）→ `DEVELOPMENT_LOG.md`