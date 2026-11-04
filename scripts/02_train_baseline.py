import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from sklearn.exceptions import ConvergenceWarning
import warnings
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning)

from src.config import (
    SEED, LABEL_NAMES, ENCODING, TRAIN_CSV, VAL_CSV,
    TFIDF_WORD, TFIDF_CHAR,
    BASELINE_PKL, BASELINE_VAL_PRED, BASELINE_EXP_CSV,
    BASELINE_PARAMS_JSON, BASELINE_CM_VAL, N_CV_FOLDS,
)

os.makedirs(os.path.dirname(BASELINE_PKL), exist_ok=True)
os.makedirs(os.path.dirname(BASELINE_VAL_PRED), exist_ok=True)

print("--- 1. ĐỌC DỮ LIỆU ---")
train = pd.read_csv(TRAIN_CSV, keep_default_na=False, encoding=ENCODING)
val = pd.read_csv(VAL_CSV, keep_default_na=False, encoding=ENCODING)

train = train[train['text_seg'] != '']
val = val[val['text_seg'] != '']
train['label'] = train['label'].astype(int)
val['label'] = val['label'].astype(int)

X_train, y_train = train['text_seg'], train['label']
X_val, y_val = val['text_seg'], val['label']

print(f"Train size: {len(X_train)} | Val size: {len(X_val)}")

print("--- 2. VECTOR HÓA & 3. CHỌN C BẰNG CV TRÊN TRAIN (model trần, chưa calibrate) ---")


def make_features():
    return FeatureUnion([
        ("word_tfidf", TfidfVectorizer(**TFIDF_WORD)),
        ("char_tfidf", TfidfVectorizer(**TFIDF_CHAR)),
    ])


configs = {
    "LogisticRegression_C1_bal": ("LogisticRegression",
        dict(C=1.0, class_weight="balanced", random_state=SEED, max_iter=1000)),
    "LogisticRegression_C1": ("LogisticRegression",
        dict(C=1.0, class_weight=None, random_state=SEED, max_iter=1000)),
    "LogisticRegression_C0.1_bal": ("LogisticRegression",
        dict(C=0.1, class_weight="balanced", random_state=SEED, max_iter=1000)),
    "LinearSVC_C1_bal": ("LinearSVC",
        dict(C=1.0, class_weight="balanced", random_state=SEED, max_iter=2000)),
    "LinearSVC_C1": ("LinearSVC",
        dict(C=1.0, class_weight=None, random_state=SEED, max_iter=2000)),
    "LinearSVC_C0.1_bal": ("LinearSVC",
        dict(C=0.1, class_weight="balanced", random_state=SEED, max_iter=2000)),
    "LinearSVC_C10_bal": ("LinearSVC",
        dict(C=10.0, class_weight="balanced", random_state=SEED, max_iter=2000)),
    "LinearSVC_C0.05_bal": ("LinearSVC",
        dict(C=0.05, class_weight="balanced", random_state=SEED, max_iter=2000)),
}

CLF_CLASS = {"LogisticRegression": LogisticRegression, "LinearSVC": LinearSVC}

cv = StratifiedKFold(n_splits=N_CV_FOLDS, shuffle=True, random_state=SEED)
rows = []
for name, (clf_type, params) in configs.items():
    pipe = Pipeline([('tfidf', make_features()), ('clf', CLF_CLASS[clf_type](**params))])
    scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="f1_macro", n_jobs=-1)
    rows.append({"model": name, "clf": clf_type,
                 "cv_f1_mean": round(float(scores.mean()), 4),
                 "cv_f1_std": round(float(scores.std()), 4)})
    print(f"[{name}] CV F1-macro (train, {N_CV_FOLDS} folds): {scores.mean():.4f} ± {scores.std():.4f}")

exp = pd.DataFrame(rows).sort_values("cv_f1_mean", ascending=False)
best_model_name = exp.iloc[0]["model"]
best_clf_type, best_params = configs[best_model_name]
print(f"\n=> Mô hình tốt nhất theo CV: {best_model_name}")

print("--- 4. FIT MODEL THẮNG TRÊN FULL TRAIN + CALIBRATE (1 lần duy nhất) ---")
winner = CLF_CLASS[best_clf_type](**best_params)
if best_clf_type == "LinearSVC":
    winner = CalibratedClassifierCV(winner, cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED))
best_model = Pipeline([('tfidf', make_features()), ('clf', winner)])
best_model.fit(X_train, y_train)

val_predictions = best_model.predict(X_val)
f1_val = f1_score(y_val, val_predictions, average="macro")
f1_train = f1_score(y_train, best_model.predict(X_train), average="macro")
print(f"=> F1-macro trên Train: {f1_train:.4f} | trên Val (đánh giá 1 lần): {f1_val:.4f}")

exp["val_f1"] = ""
exp.loc[exp["model"] == best_model_name, "val_f1"] = round(f1_val, 4)
exp["selected"] = exp["model"] == best_model_name
exp.to_csv(BASELINE_EXP_CSV, index=False)
print(f"Đã lưu log thí nghiệm vào: {BASELINE_EXP_CSV}")

with open(BASELINE_PARAMS_JSON, "w", encoding="utf-8") as f:
    json.dump({
        "best_model": best_model_name, "clf": best_clf_type, "params": best_params,
        "cv_f1_mean": float(exp.iloc[0]["cv_f1_mean"]),
        "val_f1": round(float(f1_val), 4), "train_f1": round(float(f1_train), 4),
        "n_train": len(X_train), "n_val": len(X_val),
        "sklearn_version": sklearn.__version__, "seed": SEED,
        "calibrated": best_clf_type == "LinearSVC",
    }, f, ensure_ascii=False, indent=2)

print("--- 5. BÁO CÁO TRÊN TẬP VAL (1 lần duy nhất) ---")
target_names = [f"{ten} ({i})" for i, ten in enumerate(LABEL_NAMES)]
print(classification_report(y_val, val_predictions, target_names=target_names))

cm = confusion_matrix(y_val, val_predictions)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=target_names, yticklabels=target_names)
plt.title(f'Confusion Matrix (Val) - {best_model_name}')
plt.ylabel('Thực tế')
plt.xlabel('Dự đoán')
plt.tight_layout()
plt.savefig(BASELINE_CM_VAL, dpi=300)
print(f"Đã lưu ảnh Confusion Matrix (val) vào: {BASELINE_CM_VAL}")

print("--- 6. LƯU DỰ ĐOÁN VAL (phục vụ phân tích lỗi) ---")
val_results = val.copy()
val_results['pred'] = val_predictions
probs = best_model.predict_proba(X_val)
for i in range(len(LABEL_NAMES)):
    val_results[f'prob_{i}'] = probs[:, i]
val_results.to_csv(BASELINE_VAL_PRED, index=False, encoding=ENCODING)
print(f"Đã lưu dự đoán Val vào: {BASELINE_VAL_PRED}")

print("\n--- LỖI THƯỜNG GẶP (NHÃN OFFENSIVE BỊ ĐOÁN SAI) ---")
misclassified_offensive = val_results[(val_results['label'] == 1) & (val_results['pred'] != 1)]
for idx, row in misclassified_offensive.head(15).iterrows():
    print(f"Thực tế: OFFENSIVE | Dự đoán: {target_names[row['pred']]}")
    print(f"Text: {row['text']}")
    print("-" * 50)

print("\n--- 7. LƯU MÔ HÌNH ---")
with open(BASELINE_PKL, "wb") as f:
    pickle.dump(best_model, f)
print(f"Chưa chạy tập test. Đã lưu mô hình tốt nhất ({best_model_name}) vào: {BASELINE_PKL}")
print(f"Random state được sử dụng: {SEED}")
