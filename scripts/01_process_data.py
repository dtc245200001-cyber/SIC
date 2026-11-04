import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unicodedata

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import (
    SEED, LABEL_NAMES, ENCODING, RAW_CSV, TRAIN_CSV, VAL_CSV, TEST_CSV,
    TEST_SIZE, VAL_SIZE,
)
from src.preprocess import lam_sach, tach_tu


def chuan_hoa_nhe(text):
    return unicodedata.normalize('NFC', str(text)).strip().lower()

print("Đọc dữ liệu...")
df = pd.read_csv(RAW_CSV, encoding=ENCODING)
if 'free_text' in df.columns: df.rename(columns={'free_text': 'text'}, inplace=True)
if 'label_id' in df.columns: df.rename(columns={'label_id': 'label'}, inplace=True)
df = df.dropna(subset=['text', 'label'])
df = df[df['text'].astype(str).str.strip().str.lower() != '']
df = df[df['text'].astype(str).str.strip() != '#ERROR!']
df['text'] = df['text'].astype(str)
if df['label'].dtype == object: df['label'] = df['label'].map({ten: i for i, ten in enumerate(LABEL_NAMES)})
df['label'] = df['label'].astype(int)

print("Bỏ trùng theo raw chuẩn hóa nhẹ (strip + NFC + lower)...")
n_truoc = len(df)
df['text_key'] = df['text'].apply(chuan_hoa_nhe)
group_sizes = df.groupby('text_key')['label'].size()
n_nhom_trung = int((group_sizes > 1).sum())
mau_trung = df[df.duplicated(subset=['text_key'], keep=False)]
n_cap_mau_thuan = int((mau_trung.groupby('text_key')['label'].nunique() > 1).sum())
print(f"Số nhóm trùng: {n_nhom_trung}, nhóm nhãn mâu thuẫn: {n_cap_mau_thuan}")
# Nhóm mâu thuẫn: giữ nhãn độc hại cao nhất (HATE=2 > OFFENSIVE=1 > CLEAN=0)
labels = df.groupby('text_key')['label'].max().reset_index()
df_sach = df.drop_duplicates(subset=['text_key'])[['text', 'text_key']].merge(labels, on='text_key')
print(f"Bỏ trùng: {n_truoc} -> {len(df_sach)} mẫu (giữ {n_cap_mau_thuan} nhóm mâu thuẫn theo nhãn nặng nhất)")

print("Làm sạch...")
n_truoc_sach = len(df_sach)
df_sach['text_clean'] = df_sach['text'].apply(lam_sach)
df_sach = df_sach[df_sach['text_clean'] != '']
print(f"Làm sạch: {n_truoc_sach} -> {len(df_sach)} mẫu (loại {n_truoc_sach - len(df_sach)} mẫu rỗng sau làm sạch)")

print("Tách từ...")
df_sach['text_seg'] = df_sach['text_clean'].apply(tach_tu)
df_sach = df_sach[df_sach['text_seg'].str.len() > 0].copy().reset_index(drop=True)
df_sach.insert(0, 'sid', range(len(df_sach)))

print("Chia tập...")
train_val, test = train_test_split(df_sach, test_size=TEST_SIZE, stratify=df_sach['label'], random_state=SEED)
train, val = train_test_split(train_val, test_size=VAL_SIZE, stratify=train_val['label'], random_state=SEED)
train = train.reset_index(drop=True)
val = val.reset_index(drop=True)
test = test.reset_index(drop=True)

os.makedirs(os.path.dirname(TRAIN_CSV), exist_ok=True)
train.to_csv(TRAIN_CSV, index=False, encoding=ENCODING)
val.to_csv(VAL_CSV, index=False, encoding=ENCODING)
test.to_csv(TEST_CSV, index=False, encoding=ENCODING)

print(f"Train: {len(train)}")
print(f"Val: {len(val)}")
print(f"Test: {len(test)}")
print("Tỉ lệ nhãn (train/val/test):")
print(pd.crosstab(df_sach['label'], "n").T.to_string())
for ten, d in [("train", train), ("val", val), ("test", test)]:
    print(f"  {ten}: {d['label'].value_counts(normalize=True).sort_index().round(4).to_dict()}")
