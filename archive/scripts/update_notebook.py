import os, re, json, unicodedata
import pandas as pd
from pyvi import ViTokenizer
from sklearn.model_selection import train_test_split

print("1. Kiểm tra ViTokenizer:")
sample_text = "Tôi là sinh viên đại học"
print(f"Gốc: {sample_text}")
print(f"Segmented: {ViTokenizer.tokenize(sample_text)}")

# Đọc dữ liệu
print("\n--- XỬ LÝ DỮ LIỆU ---")
DATA_PATH = "data/raw/comments.csv" # Assuming it was concatenated here or we use df_goc
if not os.path.exists(DATA_PATH):
    # Try the previously saved raw data if comments.csv doesn't exist
    DATA_PATH = "data/processed/01_du_lieu_goc.csv"
df = pd.read_csv(DATA_PATH)

# Rename to standard if needed
if 'free_text' in df.columns:
    df.rename(columns={'free_text': 'text'}, inplace=True)
if 'label_id' in df.columns:
    df.rename(columns={'label_id': 'label'}, inplace=True)

df["text"] = df["text"].astype(str)
# Xóa "#ERROR!"
n_before = len(df)
df = df[df["text"] != "#ERROR!"]
print(f"Đã xóa {n_before - len(df)} dòng '#ERROR!'")

# Định nghĩa các regex
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
RE_CHEN_KT  = re.compile(r"\b\w(?:[\.\-_]\w)+\b")   # removed \* from here to preserve l**
RE_EMOJI    = re.compile(r"[\U0001F300-\U0001FAFF\u2600-\u27BF]")
RE_TU       = re.compile(r"\b(\w+)\b")

def fix_keo_dai(match):
    word = match.group(0)
    if word.isupper(): return word # Keep acronyms like CCCĐ
    return re.sub(r"(.)\1{2,}", r"\1", word)

def lam_sach(text):
    t = unicodedata.normalize("NFC", str(text))
    t = RE_HTML.sub(" ", t)
    t = RE_URL.sub(" ", t)
    t = RE_MENTION.sub(" ", t)
    t = RE_EMOJI.sub(" ", t) # Remove emoji
    
    # Fix keo dai without breaking acronyms
    t = re.sub(r"\b\w+\b", fix_keo_dai, t)
    
    # Fix chen ky tu
    t = RE_CHEN_KT.sub(lambda m: re.sub(r"[\.\-_]", "", m.group()), t)
    
    # Lowercase at the end
    t = t.lower()
    
    # Substitute teencode
    t = RE_TU.sub(lambda m: TEENCODE.get(m.group(1), m.group(1)), t)
    return re.sub(r"\s+", " ", t).strip()

def tach_tu(text_sach):
    if not isinstance(text_sach, str): return ""
    # ViTokenizer adds '_' between syllables of a word.
    # To handle 'l**' or 'b**i', PyVi might split it because of '*'.
    # We replace '*' with a placeholder before segmentation to preserve it.
    t = text_sach.replace("*", "X_STAR_X")
    t = ViTokenizer.tokenize(t)
    return t.replace("X_STAR_X", "*")

print("Đang làm sạch và tách từ...")
df["text_clean"] = df["text"].apply(lam_sach)
df = df[df["text_clean"] != ""]

# Bỏ trùng
def resolve_duplicates(group):
    modes = group['label'].mode()
    if len(modes) == 1:
        return group.iloc[0:1].assign(label=modes.iloc[0])
    return pd.DataFrame(columns=group.columns) # drop if tie

n_before_dedup = len(df)
dup_mask = df.duplicated(subset=['text_clean'], keep=False)
df_unique = df[~dup_mask]
df_dups = df[dup_mask]

groups_processed = df_dups['text_clean'].nunique()
resolved = df_dups.groupby('text_clean', group_keys=False).apply(resolve_duplicates)

df_final = pd.concat([df_unique, resolved]).reset_index(drop=True)
deleted_rows = n_before_dedup - len(df_final)

print(f"\n2. Đã xử lý {groups_processed} nhóm trùng lặp (trùng sau khi làm sạch).")
print(f"Số dòng bị xóa trong bước này (do hòa nhãn hoặc bỏ bớt các bản sao): {deleted_rows}")

# Tách từ
df_final["text_seg"] = df_final["text_clean"].apply(tach_tu)

print("\n10 ví dụ CLEAN và SEGMENT:")
for _, row in df_final.sample(10, random_state=42).iterrows():
    print(f"CLEAN: {row['text_clean']}")
    print(f"SEGMENT: {row['text_seg']}")
    print("-")

# Split
train_val, test = train_test_split(df_final, test_size=0.10, stratify=df_final["label"], random_state=42)
train, val = train_test_split(train_val, test_size=1/9, stratify=train_val["label"], random_state=42)

set_train = set(train['text_clean'])
set_val = set(val['text_clean'])
set_test = set(test['text_clean'])

print("\n3. Kiểm tra trùng lặp:")
print(f"Trùng train-val: {len(set_train.intersection(set_val))}")
print(f"Trùng train-test: {len(set_train.intersection(set_test))}")
print(f"Trùng val-test: {len(set_val.intersection(set_test))}")

print("\n5. Số dòng cuối và tỉ lệ:")
print(f"Train: {len(train)}")
print(train['label'].value_counts(normalize=True)*100)
print(f"Val: {len(val)}")
print(val['label'].value_counts(normalize=True)*100)
print(f"Test: {len(test)}")
print(test['label'].value_counts(normalize=True)*100)

# Cập nhật Notebook
notebook_path = "notebooks/01_du_lieu_va_tien_xu_ly.ipynb"
with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Tạo các cell mã mới
cell_lam_sach = """def fix_keo_dai(match):
    word = match.group(0)
    if word.isupper(): return word
    return re.sub(r"(.)\\1{2,}", r"\\1", word)

def lam_sach(text):
    t = unicodedata.normalize("NFC", str(text))
    t = RE_HTML.sub(" ", t)
    t = RE_URL.sub(" ", t)
    t = RE_MENTION.sub(" ", t)
    t = RE_EMOJI.sub(" ", t) # Loại bỏ emoji
    
    t = re.sub(r"\\b\\w+\\b", fix_keo_dai, t)
    t = RE_CHEN_KT.sub(lambda m: re.sub(r"[\\.\\-_]", "", m.group()), t)
    t = t.lower()
    t = RE_TU.sub(lambda m: TEENCODE.get(m.group(1), m.group(1)), t)
    return re.sub(r"\\s+", " ", t).strip()

def tach_tu(text_sach):
    if not isinstance(text_sach, str): return ""
    t = text_sach.replace("*", "X_STAR_X")
    t = ViTokenizer.tokenize(t)
    return t.replace("X_STAR_X", "*")"""

cell_process = """# Đọc dữ liệu
if USE_DEMO_DATA:
    df = tao_du_lieu_mau()
else:
    df = pd.read_csv(DATA_PATH)

if 'free_text' in df.columns: df.rename(columns={'free_text': 'text'}, inplace=True)
if 'label_id' in df.columns: df.rename(columns={'label_id': 'label'}, inplace=True)

df["text"] = df["text"].astype(str)
if df["label"].dtype == object:
    df["label"] = df["label"].map({ten: i for i, ten in enumerate(LABEL_NAMES)})
df["label"] = df["label"].astype(int)

# Xóa ERROR
df = df[df["text"] != "#ERROR!"]

# Làm sạch trước
df["text_clean"] = df["text"].apply(lam_sach)
df = df[df["text_clean"] != ""]

# Bỏ trùng dựa trên text_clean
def resolve_duplicates(group):
    modes = group['label'].mode()
    if len(modes) == 1:
        return group.iloc[0:1].assign(label=modes.iloc[0])
    return pd.DataFrame(columns=group.columns)

dup_mask = df.duplicated(subset=['text_clean'], keep=False)
df_unique = df[~dup_mask]
df_dups = df[dup_mask]

resolved = df_dups.groupby('text_clean', group_keys=False).apply(resolve_duplicates)
df_sach = pd.concat([df_unique, resolved]).reset_index(drop=True)

# Tách từ
df_sach["text_seg"] = df_sach["text_clean"].apply(tach_tu)
df_sach = df_sach[df_sach["text_seg"].str.len() > 0].copy().reset_index(drop=True)
df_sach.to_csv("data/processed/02_du_lieu_sach.csv", index=False)"""

cell_split = """train_val, test = train_test_split(df_sach, test_size=0.10, stratify=df_sach["label"], random_state=SEED)
train, val = train_test_split(train_val, test_size=1/9, stratify=train_val["label"], random_state=SEED)
train, val, test = [d.copy().reset_index(drop=True) for d in (train, val, test)]
train.to_csv("data/processed/03_train.csv", index=False)
val.to_csv("data/processed/03_val.csv", index=False)
test.to_csv("data/processed/03_test.csv", index=False)
print(f"train = {len(train):,} | val = {len(val):,} | test = {len(test):,}. Đã lưu 3 file CSV trong data/processed/")"""

# Update cells based on content heuristics
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = "".join(cell['source'])
        if "def lam_sach" in source:
            cell['source'] = [line + "\\n" for line in cell_lam_sach.split("\\n")]
            cell['source'][-1] = cell['source'][-1].replace("\\n", "")
        elif "drop_duplicates" in source and "01_du_lieu_goc" in source:
            cell['source'] = [line + "\\n" for line in cell_process.split("\\n")]
            cell['source'][-1] = cell['source'][-1].replace("\\n", "")
        elif "train_test_split" in source and "03_train.csv" in source:
            cell['source'] = [line + "\\n" for line in cell_split.split("\\n")]
            cell['source'][-1] = cell['source'][-1].replace("\\n", "")
        elif "RE_CHEN_KT" in source and "RE_KEO_DAI" in source:
             # Update regexes cell
             new_source = source.replace('r"\\b\\w(?:[\\.\\-_\\*]\\w)+\\b"', 'r"\\b\\w(?:[\\.\\-_]\\w)+\\b"')
             cell['source'] = [line + "\\n" for line in new_source.split("\\n")]
             cell['source'][-1] = cell['source'][-1].replace("\\n", "")

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print("Đã cập nhật notebook thành công!")
