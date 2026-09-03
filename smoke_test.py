# -*- coding: utf-8 -*-
"""Smoke test for the running FastAPI server (run with .venv python).

Covers both languages: EN page (default) and ZH page (?lang=zh).
Usage: python smoke_test.py [base_url]   (default http://127.0.0.1:8000)
"""
import json
import sys
import urllib.request

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000"


def get(path):
    with urllib.request.urlopen(BASE + path, timeout=10) as r:
        return r.status, r.read()


# ---- EN page (default) -------------------------------------------------
status, body = get("/")
html = body.decode("utf-8")
print("EN index status:", status, "len:", len(html))
for kw in [
    "Support Vector Machine",   # model label in metrics table
    "Try a Prediction",         # predict button
    "SURVIVED",                 # result template in JS
    "Data Visualization",       # nav link
    "Accuracy",                 # table column
    "Service health check",     # API description table
    "scikit-learn",             # footer
    "langBtn",                  # language switch button
]:
    print("EN", repr(kw), "->", kw in html)

# ---- ZH page -----------------------------------------------------------
status, body = get("/?lang=zh")
zh = body.decode("utf-8")
print("ZH index status:", status, "len:", len(zh))
for kw in [
    "泰坦尼克号生还预测",   # page title
    "数据可视化",          # nav link
    "预测生还",            # predict button
    "准确率",              # table column
    "性别与生还率",        # chart caption
    "最优模型",            # best-model hint
    "模型：",              # results-section model label
    "预测中",              # JS busy text
    "服务健康检查",        # API description table
    "langBtn",             # language switch button
]:
    print("ZH", repr(kw), "->", kw in zh)

# ---- language switch button present on both ----------------------------
print("lang button on EN:", "langBtn" in html, "| on ZH:", "langBtn" in zh)

# ---- charts (6 figures, bilingual PNG) ---------------------------------
print("img tags:", zh.count('<img src="/figures/'))
status, body = get("/figures/1_survived_counts.png")
print("png status:", status, "bytes:", len(body))

# ---- predictions API ----------------------------------------------------
for model in ["logistic", "svm", "decision_tree", "random_forest"]:
    status, body = get(f"/api/predictions?model={model}")
    rows = json.loads(body.decode("utf-8"))
    print(f"predictions[{model}]:", status, "rows:", len(rows),
          "first:", rows[0]["PassengerId"], rows[0]["Survived_true"])

# ---- predict API --------------------------------------------------------
status, body = get("/api/predict?Pclass=1&Sex=female&Age=25&Fare=110&Embarked=C")
j = json.loads(body.decode("utf-8"))
print("predict GET:", status, j["prediction_text"], "prob:", j["survived_probability"])
print("docs status:", get("/docs")[0])
print("SMOKE TEST DONE")
