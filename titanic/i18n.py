"""Lightweight i18n dictionary for the web UI (English <-> Simplified Chinese).

Kept in one module so the template context stays tidy and every string has a
single source of truth. The charts themselves are bilingual (EN + ZH baked
into the PNG), so switching the page language never requires swapping images.
"""

# ---------------------------------------------------------------------------
# UI strings: key -> (en, zh)
# ---------------------------------------------------------------------------
UI = {
    # page chrome
    "page_title": ("Titanic Survival Prediction — AI Coursework",
                   "泰坦尼克号生还预测 — AI 课程作业"),
    "header_subtitle": ("Titanic Dataset · Classification · Scikit-learn · "
                        "FastAPI · Data Visualization",
                        "泰坦尼克数据集 · 分类 · Scikit-learn · FastAPI · 数据可视化"),
    "lang_btn_en": ("中文", "EN"),          # button shows the *other* language
    "lang_hint_en": ("Switch to Chinese", "切换到英文"),
    # navigation
    "nav_charts": ("📊 Data Visualization", "📊 数据可视化"),
    "nav_metrics": ("📈 Model Comparison", "📈 模型对比"),
    "nav_predict": ("🔮 Try a Prediction", "🔮 预测演示"),
    "nav_results": ("📄 Test Results", "📄 测试结果"),
    "nav_api": ("⚡ API", "⚡ 接口文档"),
    # section titles
    "sec_charts": ("1 · Data Visualization", "1 · 数据可视化"),
    "sec_metrics": ("2 · Model Comparison", "2 · 四模型对比"),
    "sec_predict": ("3 · Try a Prediction", "3 · 填写信息预测生还"),
    "sec_results": ("4 · Test Set Predictions", "4 · 测试集(20%)预测结果"),
    "sec_api": ("5 · REST API", "5 · 接口说明"),
    # charts intro
    "charts_intro": ("8 charts from <code>data/titanic.csv</code> (891 passengers). "
                     "Every figure is bilingual (EN + 中文).",
                     "基于 <code>data/titanic.csv</code>(891 名乘客) 生成的 8 张图。"
                     "每张图均含中英双语标注。"),
    # metrics
    "metrics_intro": ("Same 80/20 split, <code>random_state=42</code>, "
                      "<code>stratify=y</code> for all models (courseware "
                      "reproduction). Metrics are computed on the untouched test set.",
                      "四个模型共用同一划分：80/20、<code>random_state=42</code>、"
                      "<code>stratify=y</code>(课件复现规范)，指标在未经触碰的测试集上计算。"),
    "col_model": ("Model", "模型"),
    "col_accuracy": ("Accuracy", "准确率"),
    "col_precision": ("Precision", "精确率"),
    "col_recall": ("Recall", "召回率"),
    "col_f1": ("F1 Score", "F1 分数"),
    "col_auc": ("AUC", "AUC"),
    "auc_note": ("AUC = area under the ROC curve (threshold-independent "
                  "ranking quality, courseware p.38-40)",
                  "AUC = ROC 曲线下面积(不依赖阈值、衡量概率排序能力，课件 p.38-40)"),
    "best_model_line": ("⭐ Best model (highest test accuracy):", 
                        "⭐ 最优模型(测试集准确率最高)："),
    "default_predict_hint": ("— used by default in the prediction demo below",
                             "—— 默认用于下方预测演示"),
    "metrics_missing": ("Metrics not available yet. Run "
                        "<code>python -m titanic.train</code> first.",
                        "暂无指标数据。请先运行 <code>python -m titanic.train</code>。"),
    # prediction form
    "form_class": ("Passenger Class", "舱位等级"),
    "form_class_1": ("1st — First", "一等舱"),
    "form_class_2": ("2nd — Second", "二等舱"),
    "form_class_3": ("3rd — Third", "三等舱"),
    "form_sex": ("Sex", "性别"),
    "form_female": ("Female", "女"),
    "form_male": ("Male", "男"),
    "form_age": ("Age", "年龄(岁)"),
    "form_sibsp": ("Siblings / Spouse aboard", "兄弟姐妹/配偶人数"),
    "form_parch": ("Parents / Children aboard", "父母/子女人数"),
    "form_fare": ("Fare (£)", "票价(英镑)"),
    "form_embarked": ("Port of Embarkation", "登船港口"),
    "emb_S": ("S — Southampton", "S — 南安普顿"),
    "emb_C": ("C — Cherbourg", "C — 瑟堡"),
    "emb_Q": ("Q — Queenstown", "Q — 皇后镇"),
    "form_model": ("Model", "预测模型"),
    "model_best": ("Best model (auto)", "最优模型(自动)"),
    "btn_predict": ("Predict Survival", "预测生还"),
    "predicting": ("⏳ Predicting…", "⏳ 预测中…"),
    "res_survived": ("✅ SURVIVED", "✅ 预测生还"),
    "res_not_survived": ("❌ NOT SURVIVED", "❌ 预测未生还"),
    "res_model": ("Model", "模型"),
    "res_surv_prob": ("Survival probability", "生还概率"),
    "res_not_prob": ("Not-survived probability", "未生还概率"),
    "res_error": ("⚠️ Error", "⚠️ 错误"),
    # results table
    "results_model_label": ("Model:", "模型："),
    "btn_load": ("Load Table", "加载表格"),
    "loading": ("Loading…", "加载中…"),
    "loading_rows": ("Loading rows…", "加载中…"),
    "hint_best_default": ("Showing best model by default", "默认展示最优模型"),
    "col_pid": ("PassengerId", "乘客编号"),
    "col_true": ("True", "真实"),
    "col_pred": ("Predicted", "预测"),
    "col_correct": ("Correct?", "是否正确"),
    "yes": ("✔ yes", "✔ 是"),
    "no": ("✘ no", "✘ 否"),
    "survived_short": ("✅ survived", "✅ 生还"),
    "not_survived_short": ("💀 not survived", "💀 未生还"),
    "rows_correct": ("{n} test rows · {c} correct ({pct}% accuracy)",
                     "{n} 条测试数据 · 正确 {c} 条(准确率 {pct}%)"),
    # API section
    "api_desc": ("Description", "说明"),
    "api_health": ("Service health check", "服务健康检查"),
    "api_summary": ("All model metrics + figure list", "四模型全部指标 + 图表清单"),
    "api_predictions": ("Test true-vs-predicted rows", "测试集真实 vs 预测明细"),
    "api_predict_get": ("Predict via query string", "通过查询参数预测"),
    "api_predict_post": ("Predict via JSON body", "通过 JSON 请求体预测"),
    "api_note_title": ("POST example", "POST 请求示例"),
    "api_docs_label": ("Interactive API docs", "交互式 API 文档"),
    # footer
    "footer_line": ("AI Coursework · Titanic Survival Prediction & Data "
                    "Visualization · Python + FastAPI + Scikit-learn",
                    "AI 课程作业 · 泰坦尼克号生还预测与数据可视化 · Python + FastAPI + Scikit-learn"),
    "footer_files": ("Data: <code>data/titanic.csv</code> · Results: "
                     "<code>outputs/</code> · Models: <code>models/</code>",
                     "数据：<code>data/titanic.csv</code> · 结果：<code>outputs/</code>"
                     " · 模型：<code>models/</code>"),
}

# Chart captions (short one-line Chinese note under each figure).
CHART_CAPTIONS = {
    "1_survived_counts": ("Target balance: survived vs not survived",
                          "目标变量分布：生还/未生还人数对比"),
    "2_missing_values": ("Missing values per column", "各字段缺失值数量(Cabin 缺失最多)"),
    "3_sex_survival": ("Survival rate by sex", "性别与生还率：女性远高于男性"),
    "4_pclass_survival": ("Survival rate by passenger class", "舱位等级与生还率：头等舱最高"),
    "5_age_distribution": ("Age distribution by survival", "年龄分布(按是否生还分组)"),
    "6_fare_distribution": ("Fare distribution log(1+Fare)", "票价分布(log(1+Fare) 变换)"),
    "7_model_comparison": ("Model comparison: five metrics per model",
                           "四模型对比条形图：每模型五指标(含 AUC)"),
    "8_roc_curves": ("ROC curves of the four models (AUC in legend)",
                     "四模型 ROC 曲线(AUC 见图例)"),
}


def pick(key: str, lang: str) -> str:
    """Return the UI string for ``key`` in ``lang`` ('en' | 'zh')."""
    en, zh = UI[key]
    return zh if lang == "zh" else en
