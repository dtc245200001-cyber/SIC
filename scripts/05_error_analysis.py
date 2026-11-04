"""Phân tích lỗi trên TẬP VAL (không dùng tập test).

Đọc outputs/baseline_val_predictions.csv và outputs/phobert_val_predictions.csv,
xuất:
  - outputs/phan_tich_loi.csv (câu đoán sai + độ dài + độ tự tin + cột nhan_xet trống)
  - outputs/ma_tran_cap_nham_val.csv (cặp nhãn thật -> nhãn đoán)
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from src.config import LABEL_NAMES, ENCODING, BASELINE_VAL_PRED, PHOBERT_VAL_PRED, ERROR_CSV

os.makedirs(os.path.dirname(ERROR_CSV), exist_ok=True)

print("--- PHÂN TÍCH LỖI TRÊN TẬP VAL ---")


def load_errors(path, model_name):
    df = pd.read_csv(path, keep_default_na=False, encoding=ENCODING)
    df['label'] = df['label'].astype(int)
    df['pred'] = df['pred'].astype(int)
    err = df[df['label'] != df['pred']].copy()
    err['mo_hinh'] = model_name
    cols = ['mo_hinh', 'sid', 'text', 'label', 'pred']
    cols += [f'prob_{i}' for i in range(len(LABEL_NAMES)) if f'prob_{i}' in err.columns]
    cols = [c for c in cols if c in err.columns]
    return err[cols]


frames = []
try:
    frames.append(load_errors(BASELINE_VAL_PRED, "baseline"))
    print(f"Baseline: {len(frames[-1])} câu sai (val)")
except FileNotFoundError:
    print(f"Thiếu file {BASELINE_VAL_PRED}, hãy chạy 02_train_baseline.py trước")

try:
    frames.append(load_errors(PHOBERT_VAL_PRED, "phobert"))
    print(f"PhoBERT: {len(frames[-1])} câu sai (val)")
except FileNotFoundError:
    print(f"Thiếu file {PHOBERT_VAL_PRED}, hãy chạy 03_train_phobert.py trước")

if not frames:
    print("Chưa có kết quả, hãy chạy script huấn luyện trước")
    sys.exit(1)

errors = pd.concat(frames, ignore_index=True)
errors['nhan_that'] = errors['label'].map({i: ten for i, ten in enumerate(LABEL_NAMES)})
errors['nhan_doan'] = errors['pred'].map({i: ten for i, ten in enumerate(LABEL_NAMES)})
prob_cols = [f'prob_{i}' for i in range(len(LABEL_NAMES)) if f'prob_{i}' in errors.columns]
if prob_cols:
    errors['do_tu_tin'] = errors[prob_cols].max(axis=1).round(4)
errors['do_dai'] = errors['text'].astype(str).str.len()
errors['nhan_xet'] = ""
errors.to_csv(ERROR_CSV, index=False)
print(f"Đã lưu {len(errors)} câu sai vào: {ERROR_CSV}")

print("--- Ma trận cặp nhầm (nhãn thật -> nhãn đoán) ---")
cap_nham = errors.groupby(['nhan_that', 'nhan_doan', 'mo_hinh']).size().reset_index(name='so_cau')
cap_nham = cap_nham.sort_values(['mo_hinh', 'so_cau'], ascending=[True, False])
print(cap_nham.to_string(index=False))
cap_path = os.path.join(os.path.dirname(ERROR_CSV), "ma_tran_cap_nham_val.csv")
cap_nham.to_csv(cap_path, index=False)
print(f"Đã lưu ma trận cặp nhầm vào: {cap_path}")

print("--- Phân bố độ tự tin của câu sai ---")
if 'do_tu_tin' in errors.columns:
    print(errors.groupby('mo_hinh')['do_tu_tin'].describe().round(4).to_string())
print("--- Phân bố độ dài của câu sai ---")
print(errors.groupby('mo_hinh')['do_dai'].describe().round(1).to_string())
