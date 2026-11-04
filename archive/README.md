# Hướng dẫn chạy dự án

Dự án này bao gồm các kịch bản (scripts) để tiền xử lý dữ liệu và huấn luyện mô hình. Bạn nên chạy từ thư mục gốc của dự án.

## Cài đặt thư viện
```bash
pip install -r requirements.txt
```

## Các bước thực hiện

1. **Kiểm tra dữ liệu:**
```bash
python scripts/check_data.py
```

2. **Tiền xử lý dữ liệu:**
*(Lưu ý: Script `process_data_fixed.py` sẽ chạy lại toàn bộ quy trình tách từ và ghi đè dữ liệu trong thư mục `data/processed/`, nên chỉ chạy lại khi thật sự cần thiết).*
```bash
python scripts/process_data_fixed.py
```

3. **Huấn luyện Baseline (ví dụ Logistic Regression, SVM, v.v.):**
- `scripts/create_baseline.py`: Chỉ dùng để khởi tạo ra file notebook `notebooks/02_baseline.ipynb` (không thực hiện huấn luyện).
- `scripts/run_baseline.py`: Chạy kịch bản huấn luyện mô hình Baseline (TF-IDF + SVM/Logistic Regression).
```bash
python scripts/run_baseline.py
```

4. **Huấn luyện PhoBERT:**
- `scripts/create_phobert.py`: Chỉ dùng để khởi tạo ra file notebook `notebooks/03_phobert.ipynb` (không thực hiện huấn luyện).
- `scripts/run_phobert.py`: Chạy kịch bản huấn luyện mô hình PhoBERT. 
*(Lưu ý: PhoBERT cần cấu hình phần cứng mạnh, bắt buộc phải có GPU để huấn luyện. Khuyến nghị sử dụng Google Colab hoặc Kaggle thay vì chạy trên máy tính cá nhân không có GPU).*
```bash
python scripts/run_phobert.py
```
