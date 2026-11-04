import os
import json

notebook = {
    "cells": [],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {"name": "ipython", "version": 3},
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.11.0"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

def add_code_cell(source):
    notebook["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\\n" for line in source.split("\\n")]
    })

imports_code = """import os
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings("ignore")

os.makedirs("../models", exist_ok=True)
os.makedirs("../outputs", exist_ok=True)

SEED = 42
"""
add_code_cell(imports_code)

load_data_code = """print("--- 1. ĐỌC DỮ LIỆU ---")
train = pd.read_csv("../data/processed/03_train.csv", keep_default_na=False)
val = pd.read_csv("../data/processed/03_val.csv", keep_default_na=False)
test = pd.read_csv("../data/processed/03_test.csv", keep_default_na=False)

train = train[train['text_seg'] != '']
val = val[val['text_seg'] != '']
test = test[test['text_seg'] != '']

X_train, y_train = train['text_seg'], train['label']
X_val, y_val = val['text_seg'], val['label']
X_test, y_test = test['text_seg'], test['label']

print(f"Train size: {len(X_train)} | Val size: {len(X_val)} | Test size: {len(X_test)}")
"""
add_code_cell(load_data_code)

train_code = """print("--- 2. VECTOR HÓA & 3. HUẤN LUYỆN ---")
tfidf = TfidfVectorizer(
    analyzer='word',
    ngram_range=(1, 2),
    min_df=3,
    max_df=0.9
)
tfidf_char = TfidfVectorizer(
    analyzer='char_wb',
    ngram_range=(2, 5),
    min_df=5,
    max_df=0.9
)

from sklearn.pipeline import FeatureUnion

combined_features = FeatureUnion([
    ("word_tfidf", tfidf),
    ("char_tfidf", tfidf_char)
])

models = {
    "LogisticRegression_C1": LogisticRegression(C=1.0, class_weight="balanced", random_state=SEED, max_iter=1000),
    "LogisticRegression_C0.1": LogisticRegression(C=0.1, class_weight="balanced", random_state=SEED, max_iter=1000),
    "LinearSVC_C1": LinearSVC(C=1.0, class_weight="balanced", random_state=SEED, max_iter=2000),
    "LinearSVC_C0.1": LinearSVC(C=0.1, class_weight="balanced", random_state=SEED, max_iter=2000),
    "LinearSVC_C0.05": LinearSVC(C=0.05, class_weight="balanced", random_state=SEED, max_iter=2000),
    "LinearSVC_C0.03": LinearSVC(C=0.03, class_weight="balanced", random_state=SEED, max_iter=2000),
}

best_f1 = 0
best_model_name = ""
best_model = None
val_predictions = None

print("Đang huấn luyện và đánh giá trên tập Validation...")
for name, clf in models.items():
    pipeline = Pipeline([
        ('tfidf', combined_features),
        ('clf', clf)
    ])
    
    pipeline.fit(X_train, y_train)
    y_pred_val = pipeline.predict(X_val)
    f1_macro = f1_score(y_val, y_pred_val, average="macro")
    
    print(f"[{name}] F1-macro (Val): {f1_macro:.4f}")
    
    if f1_macro > best_f1:
        best_f1 = f1_macro
        best_model_name = name
        best_model = pipeline
        val_predictions = y_pred_val

print(f"\\n=> Mô hình tốt nhất: {best_model_name} với F1-macro = {best_f1:.4f}")

# Đánh giá độ overfit
y_pred_train = best_model.predict(X_train)
f1_macro_train = f1_score(y_train, y_pred_train, average="macro")
print(f"=> F1-macro trên tập Train của mô hình tốt nhất: {f1_macro_train:.4f}")
"""
add_code_cell(train_code)

eval_code = """print("--- 5. BÁO CÁO MÔ HÌNH TỐT NHẤT TRÊN TẬP VAL ---")
target_names = ["CLEAN (0)", "OFFENSIVE (1)", "HATE (2)"]
print(classification_report(y_val, val_predictions, target_names=target_names))

cm = confusion_matrix(y_val, val_predictions)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=target_names, yticklabels=target_names)
plt.title(f'Confusion Matrix (Val) - {best_model_name}')
plt.ylabel('Thực tế')
plt.xlabel('Dự đoán')
plt.tight_layout()

cm_path = f"../outputs/confusion_matrix_{best_model_name}_val.png"
plt.savefig(cm_path, dpi=300)
print(f"Đã lưu ảnh Confusion Matrix vào: {cm_path}")
plt.show()

print("\\n--- LỖI THƯỜNG GẶP (NHÃN OFFENSIVE BỊ ĐOÁN SAI) ---")
val_results = val.copy()
val_results['pred'] = val_predictions
# Lọc nhãn OFFENSIVE (1) nhưng bị đoán sai
misclassified_offensive = val_results[(val_results['label'] == 1) & (val_results['pred'] != 1)]
for idx, row in misclassified_offensive.head(15).iterrows():
    print(f"Thực tế: OFFENSIVE | Dự đoán: {target_names[row['pred']]}")
    print(f"Text: {row['text']}")
    print("-" * 50)

print("\\n--- 6. LƯU MÔ HÌNH ---")
model_path = f"../models/baseline_{best_model_name}.pkl"
with open(model_path, "wb") as f:
    pickle.dump(best_model, f)
print(f"Chưa chạy tập test. Đã lưu mô hình tốt nhất vào: {model_path}")
print(f"Random state được sử dụng: {SEED}")
"""
add_code_cell(eval_code)

os.makedirs('notebooks', exist_ok=True)
with open('notebooks/02_baseline.ipynb', 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)
print("Tạo notebook notebooks/02_baseline.ipynb thành công!")
