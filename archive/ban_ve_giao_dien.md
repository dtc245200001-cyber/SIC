# 🧩 Bản vẽ giao diện web demo — Phát hiện bình luận độc hại tiếng Việt

Tài liệu này để **dán cho Antigravity** (hoặc bất kỳ công cụ AI viết code nào) dựng web demo hoàn chỉnh.
Giao diện lấy **bố cục và phong cách** từ 7 ảnh bạn gửi (Sentinels AI), nhưng **thu nhỏ cho vừa sức dự án môn học** và **chỉ hiển thị số liệu thật** do notebook xuất ra.

---

## 0. Nguyên tắc (đọc trước)

1. **Không có số liệu giả.** Mọi con số (độ chính xác, số bình luận, F1...) phải đọc từ file do notebook tạo ra (`outputs/*.csv`, `models/config.json`) hoặc từ kết quả mô hình chạy thật. Nếu thiếu file → hiện thông báo "Chưa có kết quả, hãy chạy notebook trước", **không điền số mẫu**.
2. **Không làm** những thứ trong ảnh mà đề bài không cần: cụm GPU, luồng 1.400 bình luận/phút, nút "Freeze Stream", "Deploy Rule", webhook, khóa tài khoản... Đây là hệ thống công ty, không phải dự án môn học.
3. Nếu `models/config.json` có `"is_demo_data": true` → hiện **băng đỏ trên mọi trang**: "Đang dùng dữ liệu mẫu, không phải kết quả thật."
4. Cuối mỗi kết quả phân loại luôn có dòng nhắc: *"Mô hình có thể sai. Kết quả chỉ mang tính hỗ trợ, không thay thế người kiểm duyệt."*
5. Ngôn ngữ giao diện: **tiếng Việt**, từ ngữ đơn giản.

## 1. Công nghệ đề xuất

| Thành phần | Chọn | Lý do |
|---|---|---|
| Web | **Streamlit** (nhiều trang) | Viết bằng Python, cùng ngôn ngữ với notebook, dễ chạy lúc thi |
| Mô hình | Đọc từ `models/` | Không huấn luyện lại trong web |
| Biểu đồ | `st.bar_chart`, `st.image`, hoặc Plotly | Đơn giản |
| YouTube | `requests` gọi YouTube Data API v3 | Không cần thư viện nặng |

Cài đặt: `streamlit pandas scikit-learn pyvi requests plotly` và, nếu dùng BiLSTM/PhoBERT, thêm `torch transformers`.

## 2. Cấu trúc thư mục

```
du_an/
├── app/
│   ├── Home.py                    # Trang 1: Tổng quan
│   ├── pages/
│   │   ├── 1_Thu_nghiem.py       # Trang 2: Nhập câu → kết quả
│   │   ├── 2_Tien_xu_ly.py       # Trang 3: Trước / sau xử lý
│   │   ├── 3_Danh_gia_mo_hinh.py # Trang 4: So sánh + ma trận + phân tích lỗi
│   │   ├── 4_YouTube.py          # Trang 5: Phân loại bình luận YouTube
│   │   └── 5_Han_che_dao_duc.py  # Trang 6: Hạn chế & đạo đức
│   └── ui.py                      # Màu, CSS, hàm vẽ thẻ dùng chung
├── src/
│   ├── preprocess.py             # COPY NGUYÊN VĂN từ mục 2 của notebook
│   ├── inference.py              # Nạp mô hình + hàm phân loại
│   └── youtube.py                # COPY từ mục 9 của notebook
├── models/                        # từ notebook: config.json, tfidf_lr.pkl, bilstm.pt, phobert/
├── outputs/                       # từ notebook: các CSV và PNG kết quả
└── requirements.txt
```

> ⚠️ `src/preprocess.py` phải **giống hệt** phần làm sạch + tách từ trong notebook (từ điển teencode, các regex, `lam_sach`, `tach_tu`). Nếu khác, mô hình sẽ đoán sai vì dữ liệu đầu vào khác lúc huấn luyện.

## 3. Phong cách chung (lấy từ ảnh của bạn)

- Nền xám rất nhạt `#F7F8FA`, thẻ (card) nền trắng, bo góc 12px, viền mảnh `#E5E7EB`.
- Màu chính xanh dương `#2563EB`. Ý nghĩa màu nhãn: **Bình thường = xanh lá `#16A34A`**, **Trung lập/Xúc phạm nhẹ = cam `#D97706`**, **Toxic/Thù ghét = đỏ `#DC2626`**.
- Font chữ Inter; nhãn nhỏ (tag) dùng font đơn cách (monospace) như trong ảnh.
- Thanh bên trái: tên dự án + 6 mục trang. Trang đang chọn nổi bật.
- Mỗi kết quả là một **thẻ** có viền trái màu theo nhãn (như thẻ bình luận trong ảnh 1 và 2).

## 4. Chi tiết từng trang

### Trang 1 — Tổng quan (`Home.py`)
**Mục đích:** giới thiệu bài toán và cho thấy kết quả chính trong 10 giây.

```
┌──────────────────────────────────────────────────────────────┐
│ Phát hiện bình luận độc hại tiếng Việt                       │
│ (1–2 câu mô tả bài toán)                                     │
├───────────┬───────────┬───────────┬──────────────────────────┤
│ Số bình   │ Số nhãn   │ Mô hình   │ F1-macro mô hình tốt     │
│ luận      │           │ tốt nhất  │ nhất                     │
├───────────┴───────────┴───────────┴──────────────────────────┤
│ Quy trình: Dữ liệu → Làm sạch → Tách từ → Mô hình → Kết quả  │
├──────────────────────────────┬───────────────────────────────┤
│ Biểu đồ phân bố nhãn         │ Nút đi nhanh: Thử nghiệm,     │
│ (outputs/phan_bo_nhan.png)   │ Đánh giá, YouTube             │
└──────────────────────────────┴───────────────────────────────┘
```
**Dữ liệu:** `models/config.json` (mô hình tốt nhất, tên nhãn), `outputs/bang_so_sanh_mo_hinh.csv` (F1), `data/processed/01_du_lieu_goc.csv` (đếm số bình luận, nếu có), `outputs/phan_bo_nhan.png`.

### Trang 2 — Thử nghiệm: nhập câu → ra kết quả
**Mục đích:** phần demo chính. Tham khảo ảnh 7.

```
┌─ Nhập bình luận ───────────────────────┐ ┌─ Kết quả ──────────────────────┐
│ [Mẫu 1: teencode] [Mẫu 2: mỉa mai]     │ │  ● TOXIC            96.8%      │
│ [Mẫu 3: khen]     [Mẫu 4: spam]        │ │  Điểm độc hại ▓▓▓▓▓▓▓▓▓░       │
│ ┌────────────────────────────────────┐ │ │  Xác suất từng nhãn:           │
│ │  (ô nhập, tối đa 500 ký tự)        │ │ │   Bình thường ▓░░░░  3%        │
│ └────────────────────────────────────┘ │ │   Xúc phạm   ▓▓▓▓▓▓░ 70%       │
│ Mô hình: [PhoBERT ▾]  [Phân tích]      │ │   Thù ghét   ▓▓░░░░ 27%        │
└────────────────────────────────────────┘ │  Sau xử lý: "thằng ad ngu ..." │
                                           │  Từ gây độc: thằng, ngu (tô đỏ)│
                                           └────────────────────────────────┘
        ⚠ Mô hình có thể sai. Kết quả chỉ mang tính hỗ trợ.
```
**Hành vi:**
- Gọi `phan_loai(text, model_name)` (xem mục 5). Thẻ kết quả có **viền trái và nhãn màu** theo kết quả.
- Có ô chọn mô hình lấy từ `config.json → models_available`.
- Câu rỗng hoặc quá dài → báo lỗi thân thiện.
- **Từ gây độc (mức nâng cao, tùy chọn):** bỏ lần lượt từng từ rồi tính lại điểm; từ nào làm điểm Toxic giảm nhiều nhất thì tô đỏ. Không cần thư viện giải thích phức tạp.
- 4 nút mẫu chỉ **điền sẵn câu** vào ô, người dùng bấm "Phân tích" để chạy mô hình thật (không hiện kết quả viết sẵn).

### Trang 3 — Tiền xử lý: trước và sau
**Mục đích:** cho thấy văn bản được "dọn" thế nào. Tham khảo ảnh 4.

```
┌ Nhập câu thô ─────────────────────────────────────────────┐
│ hàng như l** ý mn đừng mua lừa đảo đấy k dùng dc đâu ...  │  [Chạy chuẩn hóa]
└───────────────────────────────────────────────────────────┘
┌ TRƯỚC (raw) ──────────────┐ ┌ SAU làm sạch + tách từ ─────────────┐
│ câu gốc                    │ │ hàng như l * * ý mọi_người đừng ... │
│ Phát hiện: 1 link,         │ │ Từng từ hiện thành chip riêng       │
│ 3 teencode, ...            │ │ (từ ghép nối bằng "_")              │
└────────────────────────────┘ └─────────────────────────────────────┘
┌ Ví dụ có sẵn ──────────────┐ ┌ Thống kê toàn bộ dữ liệu ────────────┐
│ bảng từ truoc_sau_xu_ly.csv│ │ ảnh thong_ke_dau_hieu_ban.png        │
└────────────────────────────┘ └──────────────────────────────────────┘
```
**Dữ liệu:** hàm `lam_sach`, `tach_tu`, `phat_hien` trong `src/preprocess.py`; file `outputs/truoc_sau_xu_ly.csv`, `outputs/thong_ke_dau_hieu_ban.png`. Mục "Đã phát hiện" hiện dạng huy hiệu nhỏ (link, teencode, chữ kéo dài, chèn ký tự, emoji).

### Trang 4 — Đánh giá mô hình
**Mục đích:** phần "ăn điểm" khi bảo vệ. Tham khảo ảnh 5 và 6.

Có 3 tab:
1. **So sánh mô hình:** bảng từ `outputs/bang_so_sanh_mo_hinh.csv` (Accuracy, F1-macro, Toxic Precision, Toxic Recall, độ trễ, dung lượng) + ảnh `outputs/so_sanh_mo_hinh.png`. Dòng mô hình tốt nhất tô nền xanh nhạt. Có nút tải CSV.
2. **Ma trận nhầm lẫn:** chọn mô hình bằng ô chọn → hiện ảnh `outputs/ma_tran_nham_lan_*.png` tương ứng. Dưới ảnh có 1–2 câu hướng dẫn đọc (hàng = nhãn thật, cột = mô hình đoán).
3. **Phân tích lỗi:** đọc `outputs/phan_tich_loi.csv`, tách 2 cột:
   - 🟠 *Báo nhầm (False Positive)*: bình thường nhưng bị báo độc hại.
   - 🔴 *Bỏ sót (False Negative)*: độc hại nhưng bị bỏ sót.
   Mỗi lỗi là một thẻ nhỏ: câu bình luận, nhãn thật → nhãn đoán, **gợi ý nguyên nhân**, và **nhận xét của nhóm** (cột `nhận_xét_của_nhóm` nếu đã điền, nếu trống hiện "Chưa có nhận xét").

### Trang 5 — YouTube: phân loại hàng loạt
**Mục đích:** áp dụng mô hình vào bình luận thật. Chỉ dùng video **công khai**.

```
┌ Link video YouTube: [____________________]                    ┐
│ API key: [•••••••••••]  (không lưu)    Số bình luận: [100 ───] │
│ [Lấy và phân loại]                                            │
├───────────────────────────────────────────────────────────────┤
│ Đã phân loại N bình luận │ % Bình thường │ % Xúc phạm │ % ... │
│ Biểu đồ tỷ lệ các nhóm                                         │
│ Top 5 bình luận có điểm độc hại cao nhất (bảng)               │
│ [Tải kết quả CSV]                                              │
└───────────────────────────────────────────────────────────────┘
```
**Quy tắc:**
- API key nhập qua ô mật khẩu hoặc `st.secrets`, **không ghi ra file, không in ra màn hình, không đưa lên GitHub**.
- Chỉ lưu nội dung, số like, thời gian. **Không lưu tên người bình luận.**
- Lỗi thường gặp phải có thông báo dễ hiểu: video tắt bình luận (`commentsDisabled`), hết hạn mức (`quotaExceeded`), sai key (`keyInvalid`), link sai.
- Có hiện vòng quay tiến trình khi đang chạy; giới hạn số bình luận tối đa (ví dụ 500) để tiết kiệm hạn mức.

### Trang 6 — Hạn chế & đạo đức
Trang chữ, ngắn gọn: nguồn dữ liệu và ai gán nhãn, những trường hợp mô hình dễ sai (mỉa mai, nói giảm nói tránh, thiếu ngữ cảnh, né bộ lọc), nguy cơ **chặn nhầm** làm ảnh hưởng quyền tự do ngôn luận, thiên lệch (bias) có thể có, lời khẳng định "chỉ hỗ trợ người kiểm duyệt", và hướng phát triển. **Phần này nhóm tự viết nội dung** từ kết quả phân tích lỗi thật.

## 5. Hợp đồng hàm `src/inference.py` (để các trang gọi)

```python
def doc_cau_hinh() -> dict
    # đọc models/config.json: label_names, toxic_ids, best_model, models_available, is_demo_data

def tai_mo_hinh(ten_mo_hinh: str)        # dùng st.cache_resource để chỉ nạp 1 lần
    # "TF-IDF + Logistic Regression" -> models/tfidf_lr.pkl
    # "BiLSTM"                       -> models/bilstm.pt
    # "PhoBERT (fine-tune)"          -> models/phobert/

def phan_loai(binh_luan: str, ten_mo_hinh: str | None = None) -> dict
    # trả về: {"binh_luan", "sau_xu_ly", "nhan", "do_tin_cay", "diem_toxic",
    #          "xac_suat_tung_nhan": {ten_nhan: xac_suat}, "mo_hinh"}

def phan_loai_hang_loat(danh_sach: list[str], ten_mo_hinh=None) -> "DataFrame"
    # thêm cột: nhan, diem_toxic
```
Quy trình bên trong luôn là: `lam_sach → tach_tu → mô hình → softmax`. Hàm tương ứng trong notebook (mục 8 và 9) là bản mẫu để chép.

---

## 6. Các prompt để dán cho Antigravity (làm lần lượt từng bước)

**Chuẩn bị:** đặt vào thư mục dự án: notebook, thư mục `models/` và `outputs/` tải từ Colab, và file bản vẽ này. Gắn kèm 7 ảnh giao diện làm tham khảo bố cục.

### Prompt 1 — Dựng khung và lõi suy luận
```
Mình đang làm web demo cho dự án môn Trí tuệ nhân tạo: "Phát hiện bình luận độc hại tiếng Việt".
Hãy đọc file ban_ve_giao_dien.md và notebook toxic_comment_SIC.ipynb trong thư mục dự án.
Giải thích bằng tiếng Việt đơn giản.

Việc cần làm bước này (CHƯA làm các trang):
1. Tạo cấu trúc thư mục đúng như mục 2 của bản vẽ.
2. Tạo src/preprocess.py: sao chép NGUYÊN VĂN từ mục 2 của notebook (TEENCODE, các regex,
   phat_hien, lam_sach, tach_tu). Không được sửa logic.
3. Tạo src/inference.py đúng "hợp đồng hàm" ở mục 5 của bản vẽ. Đọc models/config.json.
   Chỉ nạp loại mô hình nào có file tương ứng trong models/; thiếu file thì bỏ qua và không báo lỗi.
4. Tạo src/youtube.py từ mục 9 của notebook (lay_video_id, lay_binh_luan_youtube).
5. Tạo requirements.txt và app/ui.py (màu và CSS theo mục 3 của bản vẽ).
6. Viết một script thử nhỏ chạy phan_loai() với 3 câu và in kết quả ra màn hình để kiểm tra.
Tuyệt đối không điền số liệu giả ở bất kỳ đâu.
```

### Prompt 2 — Trang 1, 2, 3
```
Dựa trên bản vẽ ở mục 4, hãy tạo app/Home.py, app/pages/1_Thu_nghiem.py, app/pages/2_Tien_xu_ly.py.
- Dùng Streamlit, giao diện tiếng Việt, theo phong cách mục 3 (thẻ trắng, viền trái màu theo nhãn).
- Số liệu chỉ được đọc từ models/config.json và outputs/*.csv. Nếu thiếu file, hiện thông báo
  "Chưa có kết quả, hãy chạy notebook trước" thay vì điền số mẫu.
- Nếu config.json có is_demo_data = true, hiện băng đỏ cảnh báo ở đầu mỗi trang.
- Trang Thử nghiệm: 4 nút mẫu chỉ điền sẵn câu vào ô nhập; kết quả luôn do mô hình chạy thật.
  Cuối kết quả luôn có dòng nhắc "Mô hình có thể sai...".
Sau khi xong, chạy "streamlit run app/Home.py" và báo mình cách mở.
```

### Prompt 3 — Trang 4, 5, 6
```
Tiếp tục theo bản vẽ mục 4: tạo app/pages/3_Danh_gia_mo_hinh.py (3 tab: so sánh mô hình, ma trận nhầm lẫn,
phân tích lỗi), app/pages/4_YouTube.py và app/pages/5_Han_che_dao_duc.py.
- Trang YouTube: API key nhập bằng ô mật khẩu (hoặc st.secrets), KHÔNG ghi ra file, KHÔNG in ra màn hình.
  Chỉ lưu nội dung, like, thời gian; không lưu tên người bình luận. Bắt và giải thích các lỗi
  commentsDisabled, quotaExceeded, keyInvalid, link sai.
- Trang Đánh giá: đọc outputs/bang_so_sanh_mo_hinh.csv, outputs/ma_tran_nham_lan_*.png,
  outputs/phan_tich_loi.csv. Dòng mô hình tốt nhất tô nền xanh nhạt.
- Trang Hạn chế & đạo đức: chỉ tạo khung có tiêu đề và đoạn trống ghi chú "Nhóm tự viết nội dung ở đây".
```

### Prompt 4 — Kiểm tra cuối
```
Hãy rà soát toàn bộ dự án:
1. Tìm mọi chỗ có số liệu bị viết cứng (hard-code) trong app/ và sửa thành đọc từ file.
2. Kiểm tra không có API key nào bị ghi vào file hoặc log.
3. Chạy thử từng trang với: (a) đủ file, (b) thiếu thư mục outputs/, (c) thiếu models/. Trang nào bị lỗi thì sửa
   để hiện thông báo thân thiện thay vì báo lỗi đỏ.
4. Viết README.md ngắn bằng tiếng Việt: cách cài đặt, cách chạy, cấu trúc thư mục.
```

---

## 7. Danh sách kiểm tra trước khi đi thi

- [ ] Web chạy được trên máy sẽ dùng khi thi (mang theo cả thư mục `models/`).
- [ ] Mọi số trên web khớp với bảng so sánh trong báo cáo.
- [ ] Có sẵn 3–4 câu ví dụ hay (đúng, sai, mỉa mai, teencode) để demo.
- [ ] Phần YouTube có phương án dự phòng nếu mất mạng: dùng sẵn file `outputs/ket_qua_youtube.csv` đã chạy trước.
- [ ] API key không nằm trong code, trong ảnh chụp màn hình hay trên GitHub.
