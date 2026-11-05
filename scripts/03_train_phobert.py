import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer, DataCollatorWithPadding, set_seed
from transformers import EarlyStoppingCallback
from datasets import Dataset

from src.config import (
    SEED, LABEL_NAMES, N_LABELS, ENCODING, TRAIN_CSV, VAL_CSV,
    PHOBERT_DIR, PHOBERT_VAL_PRED,
    MAX_LEN, PHOBERT_MODEL_NAME, PHOBERT_EPOCHS,
    PHOBERT_TRAIN_BS, PHOBERT_EVAL_BS, PHOBERT_LR,
)

parser = argparse.ArgumentParser()
parser.add_argument('--smoke', action='store_true', help='Chạy thử nhanh với ít dữ liệu')
args = parser.parse_args()

os.environ.pop("WANDB_DISABLED", None)
model_dir = PHOBERT_DIR + "_smoke" if args.smoke else PHOBERT_DIR
out_dir = os.path.join(os.path.dirname(PHOBERT_VAL_PRED), "smoke") if args.smoke else os.path.dirname(PHOBERT_VAL_PRED)
val_pred_path = os.path.join(out_dir, "phobert_val_predictions.csv")
os.makedirs(model_dir, exist_ok=True)
os.makedirs(out_dir, exist_ok=True)

set_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = "cuda" if torch.cuda.is_available() else "cpu"

print("--- 1. ĐỌC DỮ LIỆU (train + val, KHÔNG dùng test) ---")
train_df = pd.read_csv(TRAIN_CSV, keep_default_na=False, encoding=ENCODING)
val_df = pd.read_csv(VAL_CSV, keep_default_na=False, encoding=ENCODING)

train_df = train_df[train_df['text_seg'] != '']
val_df = val_df[val_df['text_seg'] != '']

if args.smoke:
    train_df = train_df.head(300)
    val_df = val_df.head(100)

print(f"Train: {len(train_df)} | Val: {len(val_df)}")

classes = np.arange(N_LABELS)
class_weights = compute_class_weight(class_weight='balanced', classes=classes, y=train_df['label'])
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)

print(f"Class weights: {class_weights}")

print("--- 2. TOKENIZATION ---")
id2label = {i: ten for i, ten in enumerate(LABEL_NAMES)}
label2id = {ten: i for i, ten in enumerate(LABEL_NAMES)}
tokenizer = AutoTokenizer.from_pretrained(PHOBERT_MODEL_NAME)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

train_ds = Dataset.from_pandas(train_df[['text_seg', 'label']])
val_ds = Dataset.from_pandas(val_df[['text_seg', 'label']])


def tokenize_function(examples):
    return tokenizer(examples["text_seg"], truncation=True, max_length=MAX_LEN)


def drop_cols(ds):
    cols = ["text_seg"]
    if "__index_level_0__" in ds.column_names:
        cols.append("__index_level_0__")
    return cols


train_encoded = train_ds.map(tokenize_function, batched=True, remove_columns=drop_cols(train_ds))
val_encoded = val_ds.map(tokenize_function, batched=True, remove_columns=drop_cols(val_ds))

train_encoded.set_format("torch")
val_encoded.set_format("torch")

print("--- 3. HUẤN LUYỆN PHOBERT ---")


class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fct = torch.nn.CrossEntropyLoss(weight=class_weights_tensor.to(logits.device))
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {"f1_macro": f1_score(labels, predictions, average="macro")}


lengths = tokenizer(train_df['text_seg'].tolist(), truncation=False)['input_ids']
lengths = [len(x) for x in lengths]
p99 = int(np.percentile(lengths, 99))
print(f"Độ dài token: p50={int(np.median(lengths))} p95={int(np.percentile(lengths, 95))} "
      f"p99={p99} max={max(lengths)} | MAX_LEN={MAX_LEN} "
      f"(tỉ lệ bị cắt: {np.mean([x > MAX_LEN for x in lengths]):.2%})")

model = AutoModelForSequenceClassification.from_pretrained(
    PHOBERT_MODEL_NAME, num_labels=N_LABELS, id2label=id2label, label2id=label2id,
    use_safetensors=True,
)

import inspect

use_cuda = torch.cuda.is_available()
try:
    from transformers import is_torch_bf16_gpu_available
    use_bf16 = is_torch_bf16_gpu_available()
except Exception:
    use_bf16 = False

training_args_kwargs = dict(
    output_dir=model_dir,
    save_strategy="epoch",
    learning_rate=PHOBERT_LR,
    per_device_train_batch_size=PHOBERT_TRAIN_BS,
    gradient_accumulation_steps=2,
    per_device_eval_batch_size=PHOBERT_EVAL_BS,
    num_train_epochs=1 if args.smoke else PHOBERT_EPOCHS,
    weight_decay=0.01,
    fp16=use_cuda and not use_bf16,
    bf16=use_bf16,
    load_best_model_at_end=True,
    metric_for_best_model="f1_macro",
    greater_is_better=True,
    seed=SEED,
    data_seed=SEED,
    logging_steps=100,
    save_total_limit=1,
    report_to="none",
)

valid_params = set(inspect.signature(TrainingArguments.__init__).parameters)
if "warmup_ratio" in valid_params:
    training_args_kwargs["warmup_ratio"] = 0.1
elif "warmup_steps" in valid_params:
    training_args_kwargs["warmup_steps"] = 100
if "eval_strategy" in valid_params:
    training_args_kwargs["eval_strategy"] = "epoch"
elif "evaluation_strategy" in valid_params:
    training_args_kwargs["evaluation_strategy"] = "epoch"

training_args = TrainingArguments(**training_args_kwargs)

trainer = WeightedTrainer(
    model=model,
    args=training_args,
    train_dataset=train_encoded,
    eval_dataset=val_encoded,
    compute_metrics=compute_metrics,
    data_collator=data_collator,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
)

trainer.train()
trainer.save_model(f"{model_dir}/best")
tokenizer.save_pretrained(f"{model_dir}/best")

print("--- 4. ĐÁNH GIÁ TRÊN TẬP VAL ---")
predictions = trainer.predict(val_encoded)
probs = torch.nn.functional.softmax(torch.tensor(predictions.predictions), dim=-1).numpy()
preds = np.argmax(probs, axis=-1)

val_results = val_df.copy()
val_results['pred'] = preds
for i in range(N_LABELS):
    val_results[f'prob_{i}'] = probs[:, i]
val_results.to_csv(val_pred_path, index=False)

target_names = [f"{ten} ({i})" for i, ten in enumerate(LABEL_NAMES)]
cm = confusion_matrix(val_df['label'], preds)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
plt.title('Confusion Matrix (PhoBERT) - Val')
plt.ylabel('Thực tế')
plt.xlabel('Dự đoán')
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "confusion_matrix_phobert_val.png"), dpi=300)

print(f"Đã lưu mô hình tốt nhất vào {model_dir}/best\n"
      f"Đã lưu dự đoán Val vào {val_pred_path}\n"
      f"Tập test CHƯA dùng. Chạy scripts/04_evaluate_test.py để predict test từ {model_dir}/best\n"
      f"Classification Report (Val):\n{classification_report(val_df['label'], preds, target_names=target_names)}")
