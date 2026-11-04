# SIC — Phát hiện bình luận độc hại tiếng Việt

Dự án phân loại bình luận tiếng Việt thành 3 nhãn: CLEAN (0), OFFENSIVE (1), HATE (2).
So sánh baseline TF-IDF + LinearSVC/LogisticRegression với PhoBERT fine-tune.

## Sơ đồ thư mục

- `data/raw/` — `comments.csv` (không sửa).
- `data/processed/` — `03_train.csv`, `03_val.csv`, `03_test.csv` (sinh ra từ script 01, bị git bỏ qua).
- `src/preprocess.py` — nguồn DUY NHẤT cho làm sạch (`lam_sach`) và tách từ (`tach_tu`).
- `src/config.py` — seed, đường dẫn, tên nhãn, MAX_LEN, siêu tham số PhoBERT.
- `scripts/` — pipeline đánh số thứ tự: 01 xử lý dữ liệu, 02 baseline, 03 PhoBERT, 04 đánh giá test, 05 phân tích lỗi.
- `notebooks/ket_qua.ipynb` — chỉ đọc file trong `outputs/` để xem bảng và biểu đồ.
- `models/` — file mô hình (bị git bỏ qua).
- `outputs/` — bảng so sánh, confusion matrix, dự đoán, phân tích lỗi.
- `app/` — tạo sau, hiện để trống.
- `archive/` — file cũ, giữ nguyên để tham khảo, không dùng.

## Quy tắc

- `src/preprocess.py` là nguồn duy nhất cho làm sạch + tách từ; không sao chép logic này sang file khác.
- Tập test chỉ dùng ở `04_evaluate_test.py`. 04 load cả baseline.pkl và
  best checkpoint PhoBERT rồi predict lại trên cùng tập test (join theo `sid`).
  Các script 02/03 chỉ train + chọn mô hình trên tập val.
- Chọn mô hình bằng F1-macro trên tập val.
- Không tự điền số liệu kết quả giả; chưa chạy thì ghi "chưa chạy".
- Không chạy lại 01 khi chưa báo trước, vì nó ghi đè `data/processed/`.
- Sửa `preprocess.py` thì phải chạy lại 01, 02, 03 theo thứ tự.
- Không xóa file, chỉ chuyển vào `archive/`.
- PhoBERT cần GPU — chạy trên Colab hoặc Kaggle, không chạy trên máy cá nhân không có GPU.

## Các lệnh thường dùng (chạy từ thư mục gốc, theo thứ tự)

```bash
pip install -r requirements.txt
python scripts/01_process_data.py
python scripts/02_train_baseline.py
python scripts/03_train_phobert.py        # cần GPU (Colab/Kaggle)
python scripts/04_evaluate_test.py
python scripts/05_error_analysis.py
python scripts/03_train_phobert.py --smoke  # chạy thử nhanh
```
