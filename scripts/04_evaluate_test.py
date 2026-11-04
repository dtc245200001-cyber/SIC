"""Đánh giá baseline + PhoBERT TRÊN TẬP TEST.

File DUY NHẤT được phép dùng tập test. Load cả hai mô hình và predict
trên CÙNG tập test, join theo sid, lưu kèm metadata tái lập.
Ghi:
  - outputs/phobert_test_predictions.csv (predict từ best checkpoint)
  - outputs/bang_so_sanh_mo_hinh.csv
  - outputs/confusion_matrix_test.png
"""
import datetime
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, f1_score, precision_recall_fscore_support, confusion_matrix,
)
from src.config import (
    SEED, LABEL_NAMES, N_LABELS, ENCODING, TEST_CSV,
    BASELINE_PKL, PHOBERT_BEST, PHOBERT_TEST_PRED, COMPARE_CSV, COMPARE_CM,
)

N_BOOTSTRAP = 1000
rng = np.random.default_rng(SEED)


def bootstrap_ci(y_true, y_pred, n=N_BOOTSTRAP):
    n_test = len(y_true)
    scores = np.empty(n)
    for b in range(n):
        idx = rng.integers(0, n_test, n_test)
        scores[b] = f1_score(y_true[idx], y_pred[idx], average="macro",
                             labels=list(range(N_LABELS)))
    return round(float(np.percentile(scores, 2.5)), 4), round(float(np.percentile(scores, 97.5)), 4)


def mcnemar(a_correct, b_correct):
    n01 = int(((~a_correct) & b_correct).sum())
    n10 = int((a_correct & (~b_correct)).sum())
    from math import erf, sqrt
    n = n01 + n10
    if n == 0:
        return 1.0
    z = abs(n01 - n10) / sqrt(n)
    return round(float(2 * (1 - 0.5 * (1 + erf(z / sqrt(2))))), 4)

os.makedirs(os.path.dirname(COMPARE_CSV), exist_ok=True)

print("--- ĐÁNH GIÁ TRÊN TẬP TEST ---")
test = pd.read_csv(TEST_CSV, keep_default_na=False, encoding=ENCODING)
test = test[test['text_seg'] != ''].copy()
test['label'] = test['label'].astype(int)
assert test['label'].isin(range(N_LABELS)).all(), "Nhãn test ngoài {0,1,2}"
assert test['sid'].is_unique, "sid trong test không duy nhất"

results = {}

print("--- Baseline: predict từ baseline.pkl ---")
import pickle
with open(BASELINE_PKL, "rb") as f:
    baseline = pickle.load(f)
test = test.sort_values('sid').reset_index(drop=True)
y_base = np.asarray(baseline.predict(test['text_seg']), dtype=int)
assert set(np.unique(y_base)) <= set(range(N_LABELS)), "Dự đoán baseline ngoài {0,1,2}"
results["baseline"] = y_base
print(f"Baseline: đã dự đoán {len(y_base)} câu test")

print("--- PhoBERT: predict từ best checkpoint ---")
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained(PHOBERT_BEST)
model = AutoModelForSequenceClassification.from_pretrained(PHOBERT_BEST).to(device)
model.eval()
from src.config import MAX_LEN
probs_list = []
with torch.no_grad():
    for i in range(0, len(test), 32):
        batch = test['text_seg'].iloc[i:i + 32].tolist()
        enc = tokenizer(batch, truncation=True, max_length=MAX_LEN,
                        padding=True, return_tensors="pt").to(device)
        logits = model(**enc).logits
        probs_list.append(torch.nn.functional.softmax(logits, dim=-1).cpu().numpy())
probs = np.concatenate(probs_list)
y_pho = np.asarray(np.argmax(probs, axis=-1), dtype=int)
assert len(y_pho) == len(test), "Số dự đoán PhoBERT lệch số câu test"
assert set(np.unique(y_pho)) <= set(range(N_LABELS)), "Dự đoán PhoBERT ngoài {0,1,2}"
results["phobert"] = y_pho

phobert_out = test[['sid', 'text', 'label']].copy()
phobert_out['pred'] = y_pho
for i in range(N_LABELS):
    phobert_out[f'prob_{i}'] = probs[:, i]
phobert_out.to_csv(PHOBERT_TEST_PRED, index=False, encoding=ENCODING)
print(f"PhoBERT: đã predict {len(y_pho)} câu, lưu vào {PHOBERT_TEST_PRED}")

y_true = test['label'].to_numpy(dtype=int)
assert len(y_base) == len(y_true) == len(y_pho), "Số dự đoán lệch số câu test"

rows = []
f1_ci = {}
for name, y_pred in results.items():
    acc = accuracy_score(y_true, y_pred)
    f1m = f1_score(y_true, y_pred, average="macro", labels=list(range(N_LABELS)))
    f1w = f1_score(y_true, y_pred, average="weighted", labels=list(range(N_LABELS)))
    f1_ci[name] = bootstrap_ci(y_true, y_pred)
    prec, rec, f1c, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(N_LABELS)), zero_division=0,
    )
    row = {"mo_hinh": name, "accuracy": round(acc, 4),
           "f1_macro": round(f1m, 4), "f1_weighted": round(float(f1w), 4),
           "f1_macro_ci95_lo": f1_ci[name][0], "f1_macro_ci95_hi": f1_ci[name][1]}
    for i, ten in enumerate(LABEL_NAMES):
        row[f"precision_{ten}"] = round(float(prec[i]), 4)
        row[f"recall_{ten}"] = round(float(rec[i]), 4)
        row[f"f1_{ten}"] = round(float(f1c[i]), 4)
    rows.append(row)
    print(f"[{name}] accuracy={acc:.4f} f1_macro={f1m:.4f} f1_weighted={f1w:.4f} "
          f"CI95=({f1_ci[name][0]:.4f}, {f1_ci[name][1]:.4f})")

p_mcnemar = mcnemar(y_true == results["baseline"], y_true == results["phobert"])
print(f"McNemar (baseline vs phobert đúng/sai): p={p_mcnemar}")
rows.append({"mo_hinh": "mcnemar_p", "accuracy": p_mcnemar})

compare = pd.DataFrame(rows)
compare.to_csv(COMPARE_CSV, index=False, encoding=ENCODING)
print(f"Đã lưu bảng so sánh vào: {COMPARE_CSV}")

meta = {
    "seed": SEED,
    "n_test": len(y_true),
    "baseline_pkl": BASELINE_PKL,
    "phobert_best": PHOBERT_BEST,
    "device": device,
    "ngay_chay": datetime.date.today().isoformat(),
}
with open(os.path.join(os.path.dirname(COMPARE_CSV), "danh_gia_test_meta.json"), "w", encoding=ENCODING) as f:
    json.dump(meta, f, ensure_ascii=False, indent=2)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
target_names = [f"{ten} ({i})" for i, ten in enumerate(LABEL_NAMES)]
for ax, (name, y_pred) in zip(axes, results.items()):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(N_LABELS)))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=target_names, yticklabels=target_names, ax=ax)
    ax.set_title(f'Confusion Matrix (Test) - {name}')
    ax.set_ylabel('Thực tế')
    ax.set_xlabel('Dự đoán')
plt.tight_layout()
plt.savefig(COMPARE_CM, dpi=300)
print(f"Đã lưu confusion matrix vào: {COMPARE_CM}")
