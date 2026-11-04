# Phát hiện bình luận độc hại tiếng Việt

Phân loại bình luận tiếng Việt thành 3 nhãn: **CLEAN (0)** — bình thường,
**OFFENSIVE (1)** — công kích, **HATE (2)** — thù ghét. So sánh baseline
TF-IDF với PhoBERT fine-tune, chọn mô hình bằng F1-macro trên tập val.

## Nguồn dữ liệu

Bộ **ViHSD** (Vietnamese Hate Speech Detection): file `data/raw/comments.csv`
với cột văn bản (`free_text`) và nhãn (`label_id`).

## Cấu trúc thư mục

```text
├── CLAUDE.md            # quy tắc dự án cho AI agent
├── README.md
├── requirements.txt
├── data/raw/            # comments.csv (không sửa)
├── data/processed/      # 03_train/val/test.csv (sinh từ script 01)
├── src/                 # preprocess.py, config.py
├── scripts/             # 01_process_data → 05_error_analysis
├── notebooks/           # ket_qua.ipynb (chỉ đọc outputs/)
├── models/              # file mô hình (git bỏ qua)
├── outputs/             # bảng so sánh, confusion matrix, phân tích lỗi
├── app/                 # tạo sau, hiện để trống
└── archive/             # file cũ
```

## Cách chạy (từ thư mục gốc, theo thứ tự)

Máy local (CPU): 01, 02, 04, 05. Colab/Kaggle (GPU): chỉ 03.
Xem chi tiết cài đặt trong `requirements.txt`.

```bash
pip install -r requirements.txt
python scripts/01_process_data.py      # xử lý dữ liệu, chia train/val/test
python scripts/02_train_baseline.py    # TF-IDF + LR/LinearSVC, lưu models/baseline.pkl
# Trên Colab (GPU): upload data/processed/*.csv + scripts/03_train_phobert.py + src/config.py
python scripts/03_train_phobert.py     # fine-tune PhoBERT, lưu models/phobert/best/
# Tải models/phobert/best/ từ Colab về máy local rồi chạy tiếp:
python scripts/04_evaluate_test.py     # predict cả hai trên test, so sánh
python scripts/05_error_analysis.py    # phân tích lỗi trên tập val
```

Mở `notebooks/ket_qua.ipynb` để xem bảng và biểu đồ từ `outputs/`.

## Bảng kết quả (tập test)

Chưa chạy — chạy `scripts/04_evaluate_test.py` để sinh `outputs/bang_so_sanh_mo_hinh.csv`.

| Mô hình  | Accuracy | F1-macro | Precision (CLEAN/OFF/HATE) | Recall (CLEAN/OFF/HATE) |
|----------|----------|----------|----------------------------|--------------------------|
| Baseline | chưa chạy | chưa chạy | chưa chạy                 | chưa chạy                |
| PhoBERT  | chưa chạy | chưa chạy | chưa chạy                 | chưa chạy                |
