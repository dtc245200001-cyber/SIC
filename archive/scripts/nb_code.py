import os, re, time, random, pickle, unicodedata, warnings, copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix

warnings.filterwarnings("ignore")
SEED = 42
random.seed(SEED); np.random.seed(SEED)

# ===================== CẤU HÌNH (CHỈ SỬA Ở ĐÂY) =====================
DATA_PATH    = "data/raw/comments.csv"          # đường dẫn file CSV của bạn
TEXT_COL     = "free_text"                      # tên cột chứa bình luận
LABEL_COL    = "label_id"                       # tên cột chứa nhãn (số 0,1,2... hoặc tên nhãn)
LABEL_NAMES  = ["CLEAN", "OFFENSIVE", "HATE"]   # tên nhãn theo thứ tự 0,1,2
TOXIC_IDS    = [1, 2]                           # nhãn nào được coi là độc hại

USE_DEMO_DATA = False    # True = chạy thử bằng dữ liệu mẫu nhỏ (KHÔNG dùng để báo cáo)
RUN_LSTM      = True     # False nếu muốn bỏ qua BiLSTM
RUN_PHOBERT   = True     # False nếu chưa có GPU / muốn chạy nhanh

MAX_LEN        = 128               # độ dài tối đa của 1 bình luận (số từ / token)
LSTM_EPOCHS    = 6
PHOBERT_NAME   = "vinai/phobert-base"
PHOBERT_EPOCHS = 3
PHOBERT_BATCH  = 32                # nếu hết bộ nhớ GPU thì giảm xuống 16
# ======================================================================

for thu_muc in ["data/raw", "data/processed", "outputs"]:
    os.makedirs(thu_muc, exist_ok=True)
print("Đã sẵn sàng. Thư mục kết quả: outputs/ | Dữ liệu trung gian: data/processed/")

def tao_du_lieu_mau(n_moi_lop=300):
    """Dữ liệu mẫu CỰC NHỎ chỉ để thử chạy notebook. KHÔNG phải dữ liệu thật."""
    sach = ["video này hay quá, cảm ơn bạn nhiều nha", "sản phẩm dùng tốt, giao hàng nhanh",
            "mình thấy shop tư vấn rất nhiệt tình", "bài giảng dễ hiểu, mong ra thêm video",
            "quán này đồ ăn ngon, giá hợp lý", "hôm nay trời đẹp quá mọi người ơi",
            "cho mình hỏi cách đặt hàng với", "ko ngờ dc giao sớm vậy, hài lòng lắm",
            "nội dung bổ ích, chúc kênh phát triển", "đội nhà đá hay thật sự, tự hào quá"]
    doc = ["mày ngu như bò, câm miệng đi", "đồ lừa đảo, cút khỏi đây",
           "thằng này nhìn là thấy rác rưởi", "bọn bán hàng như l**, đừng mua",
           "đm mày làm ăn kiểu gì vậy", "b.ắ.c k.ỳ toàn lũ ăn hại",
           "cái đồ vô học, biến đi cho khuất mắt", "sản phẩm như rác, shop lừa đảo ko dc tin",
           "ngu thì im đi, đừng có phát biểu", "lũ óc chó này chỉ giỏi chửi bậy"]
    them = ["", "haha", "thật sự", "mn ơi", "ad ơi", "😂", "https://abc.vn/xyz", "quááááá"]
    rows = []
    for nhan, goc in [(0, sach), (1, doc)]:
        for _ in range(n_moi_lop):
            rows.append((random.choice(them) + " " + random.choice(goc) + " " + random.choice(them), nhan))
    return pd.DataFrame(rows, columns=["text", "label"])

if USE_DEMO_DATA:
    TEXT_COL, LABEL_COL = "text", "label"
    LABEL_NAMES, TOXIC_IDS = ["Bình thường", "Toxic"], [1]
    df_goc = tao_du_lieu_mau()
    print("⚠️⚠️ ĐANG DÙNG DỮ LIỆU MẪU NHỎ — KẾT QUẢ CHỈ ĐỂ THỬ CHẠY, KHÔNG ĐƯA VÀO BÁO CÁO ⚠️⚠️")
else:
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Không thấy file '{DATA_PATH}'. Hãy tải file CSV lên và sửa DATA_PATH ở ô CẤU HÌNH.")
    df_goc = pd.read_csv(DATA_PATH)

df = df_goc[[TEXT_COL, LABEL_COL]].copy()
df.columns = ["text", "label"]
df = df.dropna().drop_duplicates(subset="text").copy()
df["text"] = df["text"].astype(str)

# Nếu nhãn là chữ (vd "CLEAN") thì đổi sang số theo thứ tự LABEL_NAMES
if df["label"].dtype == object:
    ten_sang_so = {ten: i for i, ten in enumerate(LABEL_NAMES)}
    df["label"] = df["label"].map(ten_sang_so)
    if df["label"].isna().any():
        raise ValueError("Có nhãn dạng chữ không khớp LABEL_NAMES. Hãy sửa LABEL_NAMES cho đúng tên nhãn trong file.")
df["label"] = df["label"].astype(int)
N_LOP = len(LABEL_NAMES)

df_goc_luu = df.copy()
df_goc_luu.to_csv("data/processed/01_du_lieu_goc.csv", index=False)

bang_nhan = df["label"].value_counts().sort_index().rename("Số bình luận").to_frame()
bang_nhan["Tên nhãn"] = [LABEL_NAMES[i] for i in bang_nhan.index]
bang_nhan["Tỷ lệ %"] = (bang_nhan["Số bình luận"] / len(df) * 100).round(2)
print(f"Tổng số bình luận sau khi bỏ trùng và bỏ dòng trống: {len(df):,}")
display(bang_nhan)

plt.figure(figsize=(6, 3.5))
sns.barplot(x=bang_nhan["Tên nhãn"], y=bang_nhan["Số bình luận"], palette="Set2")
plt.title("Phân bố nhãn trong dữ liệu"); plt.xlabel(""); plt.tight_layout()
plt.savefig("outputs/phan_bo_nhan.png", dpi=150); plt.show()

from pyvi import ViTokenizer

TEENCODE = {
    "ko": "không", "k": "không", "kh": "không", "hok": "không", "khong": "không", "hông": "không",
    "dc": "được", "đc": "được", "dk": "được", "duoc": "được",
    "vs": "với", "j": "gì", "ntn": "như thế nào", "mn": "mọi người", "mk": "mình", "mik": "mình",
    "bn": "bao nhiêu", "bt": "bình thường", "đt": "điện thoại", "sp": "sản phẩm", "ae": "anh em",
    "ad": "admin", "cmt": "bình luận", "trc": "trước", "r": "rồi", "z": "vậy", "wa": "quá", "qa": "quá",
    "thik": "thích", "iu": "yêu", "bit": "biết", "ok": "ổn", "oke": "ổn", "dm": "đm", "dcm": "đcm",
    "đg": "đang", "lm": "làm", "ng": "người", "nc": "nói chuyện", "ib": "nhắn tin", "tks": "cảm ơn", "thanks": "cảm ơn",
}
RE_URL      = re.compile(r"(https?://\S+|www\.\S+|\bt\.me/\S+)", re.I)
RE_HTML     = re.compile(r"<[^>]+>")
RE_MENTION  = re.compile(r"@\w+")
RE_KEO_DAI  = re.compile(r"(\w)\1{2,}")
RE_CHEN_KT  = re.compile(r"\b\w(?:[\.\-_\*]\w)+\b")   # b.ắ.c  k.ỳ  đ.ị.t
RE_EMOJI    = re.compile(r"[\U0001F300-\U0001FAFF\u2600-\u27BF]")
RE_TU       = re.compile(r"\b(\w+)\b")

def phat_hien(raw):
    """Đếm các 'dấu hiệu bẩn' trong bình luận gốc (dùng cho khung TRƯỚC/SAU và phân tích lỗi)."""
    thap = unicodedata.normalize("NFC", raw).lower()
    return {
        "so_url": len(RE_URL.findall(thap)),
        "so_teencode": sum(1 for t in RE_TU.findall(thap) if t in TEENCODE),
        "so_keo_dai": len(RE_KEO_DAI.findall(thap)),
        "so_chen_ky_tu": len(RE_CHEN_KT.findall(thap)),
        "co_emoji": bool(RE_EMOJI.search(thap)),
    }

def lam_sach(text):
    t = unicodedata.normalize("NFC", str(text)).lower()
    t = RE_HTML.sub(" ", t)
    t = RE_URL.sub(" ", t)
    t = RE_MENTION.sub(" ", t)
    t = RE_CHEN_KT.sub(lambda m: re.sub(r"[\.\-_\*]", "", m.group()), t)
    t = RE_KEO_DAI.sub(r"\1", t)
    t = RE_TU.sub(lambda m: TEENCODE.get(m.group(1), m.group(1)), t)
    return re.sub(r"\s+", " ", t).strip()

def tach_tu(text_sach):
    return ViTokenizer.tokenize(text_sach)

t0 = time.time()
df_sach = df.copy()
df_sach["text_clean"] = df_sach["text"].apply(lam_sach)
df_sach["text_seg"] = df_sach["text_clean"].apply(tach_tu)
df_sach = df_sach[df_sach["text_seg"].str.len() > 0].copy().reset_index(drop=True)
df_sach.to_csv("data/processed/02_du_lieu_sach.csv", index=False)
print(f"Xong làm sạch + tách từ {len(df_sach):,} bình luận trong {time.time()-t0:.1f} giây. Đã lưu data/processed/02_du_lieu_sach.csv")

cau_vi_du = [
    "hàng như l** ý mn đừng mua lừa đảo đấy k dùng dc đâu https://shopee.vn/fake-shop",
    "Bọn b.ắ.c k.ỳ ăn rau muống phá hoại văn hóa...",
    "Video bổ ích và chất lượng quááááá 😍 chúc kênh phát triển hơn nữa nhé!",
]
cau_ngau_nhien = df_sach["text"].sample(min(3, len(df_sach)), random_state=SEED).tolist()

def so_sanh_truoc_sau(danh_sach_cau):
    hang = []
    for raw in danh_sach_cau:
        sach = lam_sach(raw); seg = tach_tu(sach); dh = phat_hien(raw)
        ghi_chu = []
        if dh["so_url"]: ghi_chu.append(f"{dh['so_url']} link")
        if dh["so_teencode"]: ghi_chu.append(f"{dh['so_teencode']} teencode/viết tắt")
        if dh["so_keo_dai"]: ghi_chu.append("chữ kéo dài")
        if dh["so_chen_ky_tu"]: ghi_chu.append("chèn ký tự để né lọc")
        if dh["co_emoji"]: ghi_chu.append("có emoji")
        hang.append({"TRƯỚC (raw)": raw, "SAU làm sạch": sach, "SAU tách từ (vào mô hình)": seg,
                     "Đã phát hiện": ", ".join(ghi_chu) if ghi_chu else "—",
                     "Số ký tự trước → sau": f"{len(raw)} → {len(seg)}"})
    return pd.DataFrame(hang)

df_truoc_sau = so_sanh_truoc_sau(cau_vi_du + cau_ngau_nhien)
df_truoc_sau.to_csv("outputs/truoc_sau_xu_ly.csv", index=False)
pd.set_option("display.max_colwidth", 120)
display(df_truoc_sau)

# Thống kê trên toàn bộ dữ liệu: bao nhiêu % bình luận "bẩn" theo từng loại
dau_hieu = df_sach["text"].apply(phat_hien).apply(pd.Series)
thong_ke = pd.Series({
    "Có link": (dau_hieu["so_url"] > 0).mean() * 100,
    "Có teencode / viết tắt": (dau_hieu["so_teencode"] > 0).mean() * 100,
    "Chữ kéo dài": (dau_hieu["so_keo_dai"] > 0).mean() * 100,
    "Chèn ký tự né lọc": (dau_hieu["so_chen_ky_tu"] > 0).mean() * 100,
    "Có emoji": dau_hieu["co_emoji"].mean() * 100,
}).round(2)
print("Tỷ lệ % bình luận chứa từng dấu hiệu cần xử lý (toàn bộ dữ liệu):")
display(thong_ke.to_frame("Tỷ lệ %"))
plt.figure(figsize=(7, 3.5)); thong_ke.sort_values().plot(kind="barh", color="#4c78a8")
plt.xlabel("% bình luận"); plt.title("Mức độ 'bẩn' của dữ liệu thô"); plt.tight_layout()
plt.savefig("outputs/thong_ke_dau_hieu_ban.png", dpi=150); plt.show()

train_val, test = train_test_split(df_sach, test_size=0.10, stratify=df_sach["label"], random_state=SEED)
train, val = train_test_split(train_val, test_size=1/9, stratify=train_val["label"], random_state=SEED)
train, val, test = [d.copy().reset_index(drop=True) for d in (train, val, test)]
train.to_csv("data/processed/03_train.csv", index=False)
val.to_csv("data/processed/03_val.csv", index=False)
test.to_csv("data/processed/03_test.csv", index=False)
print(f"train = {len(train):,} | val = {len(val):,} | test = {len(test):,}. Đã lưu 3 file CSV trong data/processed/")

y_train, y_val, y_test = train["label"].values, val["label"].values, test["label"].values
X_train, X_val, X_test = train["text_seg"].tolist(), val["text_seg"].tolist(), test["text_seg"].tolist()