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

## Bảng kết quả (tập test, 3054 câu)

Sinh từ `scripts/04_evaluate_test.py` → `outputs/bang_so_sanh_mo_hinh.csv`.
Khoảng tin cậy 95% (bootstrap 1000 lần) không chồng nhau, kiểm định McNemar
p=0.0003 — khác biệt giữa hai mô hình có ý nghĩa thống kê, không do ngẫu nhiên.

| Mô hình  | Accuracy | F1-macro | F1-macro CI95 | Precision (CLEAN/OFF/HATE) | Recall (CLEAN/OFF/HATE) |
|----------|----------|----------|----------------|------------------------------|---------------------------|
| Baseline | 0.8782   | 0.6313   | (0.6016, 0.6592) | 0.8994 / 0.6282 / 0.7262   | 0.9792 / 0.2311 / 0.5377   |
| PhoBERT  | 0.8533   | **0.6834** | (0.6587, 0.7092) | 0.9604 / 0.3918 / 0.5950  | 0.8909 / 0.5377 / 0.7735   |

*(số liệu đầy đủ, kèm F1 từng lớp và F1-weighted, xem `outputs/bang_so_sanh_mo_hinh.csv`)*

**Kết luận:** PhoBERT thắng về F1-macro (metric chính vì dữ liệu lệch nhãn
nặng: CLEAN 82%, OFFENSIVE 7%, HATE 11%). Baseline có accuracy cao hơn nhưng
chủ yếu nhờ đoán đúng lớp CLEAN đa số; PhoBERT đánh đổi một phần precision
CLEAN để tăng mạnh recall OFFENSIVE (0.23→0.54) và HATE (0.54→0.77) — tức
bỏ sót ít nội dung độc hại hơn, đúng mục tiêu bài toán kiểm duyệt. Chọn
**PhoBERT** làm mô hình chính thức.

Phân tích lỗi chi tiết (ma trận cặp nhầm, độ tự tin câu sai) xem tại
`outputs/phan_tich_loi.csv` và `outputs/ma_tran_cap_nham_val.csv`.

## Hạn chế đã biết

### 1. Bias theo từ (lexical bias)

Mô hình học một số từ thành tín hiệu độc hại mà không xét ngữ cảnh. Ví dụ
khi thử trên web demo với bản PhoBERT gốc:

- "Anh Trần Văn Ngu là nhân viên ưu tú nhất của công ty" → HATE 91.8%
  ("Ngu" là tên riêng có thật ở Việt Nam)
- "Bọn trẻ đang chơi đùa ngoài sân" → HATE 95.1%
- "Con bò ăn cỏ trên đồng" → HATE 82.2%
- "Lũ kiến đang ăn vụng bánh ngọt" → HATE 80.3%

Nguyên nhân: trong ViHSD các từ này xuất hiện chủ yếu trong câu độc hại
(số mẫu chứa từ, đếm trên `comments.csv`):

| Từ    | Số mẫu | Tỉ lệ HATE |
|-------|--------|------------|
| ngu   | 377    | 67.4%      |
| lũ    | 161    | 72.7%      |
| bọn   | 387    | 58.1%      |
| chó   | 354    | 64.4%      |
| bò    | 119    | 63.9%      |

Cách xử lý: thêm câu đối chứng (counter-examples) vào tập train, file
`data/raw/counter_examples.csv`. Mỗi từ có cả câu trung tính dùng đúng nghĩa
gốc (tên người, con vật, nhóm người) và câu thật sự xúc phạm, để mô hình
học phân biệt theo ngữ cảnh. Script `01_process_data.py` chỉ thêm các câu
này vào train, val và test giữ nguyên dữ liệu gốc. Từ nào xuất hiện nhiều
trong dữ liệu gốc thì cần nhiều câu đối chứng hơn: với "ngu" (377 mẫu),
7 câu đối chứng ban đầu không đủ để sửa, phải tăng lên 27 câu.

Các câu đối chứng do người viết, không lấy từ bình luận thật, nên chỉ
giảm bias với các mẫu câu tương tự, không chứng minh mô hình đã hết bias.
Số câu đối chứng cho mỗi từ trong `data/raw/counter_examples.csv`
(sau nhiều lần điều chỉnh theo kết quả kiểm thử thực tế):

| Từ  | Số câu đối chứng | CLEAN | OFFENSIVE | HATE |
|-----|-------------------|-------|-----------|------|
| ngu | 39                | 31    | 8         | 0    |
| lũ  | 11                | 9     | 0         | 2    |
| bọn | 7                 | 5     | 0         | 2    |
| chó | 6                 | 4     | 0         | 2    |
| bò  | 7                 | 4     | 3         | 0    |

Với "ngu" (377 mẫu gốc, tỉ lệ HATE cao nhất), 8 câu đối chứng ban đầu
không đủ để sửa bias, phải tăng lên 39 câu mới thấy cải thiện rõ trên
ví dụ thực tế.

### 2. Bias chính trị và tôn giáo trong dữ liệu gốc

Một số từ chính trị và dân tộc có tỉ lệ HATE cao trong ViHSD:

| Từ    | Số mẫu | Tỉ lệ HATE |
|-------|--------|------------|
| csvn  | 15     | 73.3%      |
| cs    | 93     | 67.7%      |
| dlv   | 62     | 64.5%      |
| tàu   | 86     | 58.1%      |
| khựa  | 27     | 55.6%      |
| đảng  | 160    | 44.4%      |

Các từ như "cộng", "đỏ", "đạo" có xu hướng tương tự. Đây là cách người gán
nhãn ViHSD đánh giá bình luận chính trị và tôn giáo, không phải lỗi từ vựng
đơn giản như mục 1. Dự án **không** sửa nhóm này bằng câu đối chứng, vì
quyết định một bình luận chính trị có phải thù ghét hay không là vấn đề
tiêu chí gán nhãn, cần người có chuyên môn xem lại dữ liệu chứ không nên
do người làm mô hình tự quyết. Kết quả của mô hình với bình luận chính trị
và tôn giáo cần được xem là kém tin cậy.
