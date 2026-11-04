import pandas as pd
import numpy as np
import re

print("=== 1. Dòng bị mất ===")
try:
    df_goc = pd.read_csv('data/processed/01_du_lieu_goc.csv')
    df_sach = pd.read_csv('data/processed/02_du_lieu_sach.csv')
    
    # 01_du_lieu_goc.csv has raw texts, let's check column name. Assume 'free_text'
    text_col = 'free_text' if 'free_text' in df_goc.columns else 'text'
    
    # Dòng bị mất là dòng có trong df_goc nhưng không có trong df_sach (ở cột raw tương ứng)
    # df_sach có thể giữ lại cột gốc với tên gì? 'free_text' hoặc 'raw'?
    sach_raw_col = 'free_text' if 'free_text' in df_sach.columns else ('text' if 'text' in df_sach.columns else 'raw_text')
    # If not found, just use the first column of df_sach as it might be raw text
    if sach_raw_col not in df_sach.columns:
        sach_raw_col = df_sach.columns[0]
        
    diff = df_goc[~df_goc[text_col].isin(df_sach[sach_raw_col])]
    print(f"Số dòng bị mất: {len(diff)}")
    for idx, row in diff.iterrows():
        print(f"- Text: '{row[text_col]}'")
        print(f"  Nhãn: {row['label']}")
        print("  Lý do bị loại: Khả năng cao do sau khi làm sạch (xóa emoji, URL, khoảng trắng...) thì văn bản trở thành rỗng (empty).")
except Exception as e:
    print(f"Error: {e}")

print("\n=== 2. Trùng lặp khác nhãn ===")
try:
    df_train = pd.read_csv('data/raw/train.csv')
    df_dev = pd.read_csv('data/raw/dev.csv')
    df_test = pd.read_csv('data/raw/test.csv')
    df_all_raw = pd.concat([df_train, df_dev, df_test], ignore_index=True)
    
    text_col_raw = 'free_text' if 'free_text' in df_all_raw.columns else 'text'
    label_col_raw = 'label_id' if 'label_id' in df_all_raw.columns else 'label'
    
    dup_diff_label = df_all_raw.groupby(text_col_raw)[label_col_raw].nunique()
    dup_texts = dup_diff_label[dup_diff_label > 1].index
    print(f"Số văn bản xuất hiện nhiều lần với nhãn KHÁC NHAU: {len(dup_texts)}")
    print("10 ví dụ và nhãn được giữ lại:")
    
    count = 0
    for text in dup_texts:
        if count >= 10: break
        labels_in_raw = df_all_raw[df_all_raw[text_col_raw] == text][label_col_raw].tolist()
        kept_label = df_goc[df_goc[text_col] == text]['label'].tolist()
        if not kept_label:
            kept_label = df_sach[df_sach[sach_raw_col] == text]['label'].tolist()
        print(f"- Text: '{str(text)[:60]}...' | Nhãn ban đầu: {labels_in_raw} | Nhãn được giữ: {kept_label}")
        count += 1
except Exception as e:
    print(f"Error: {e}")

print("\n=== 3. Tỷ lệ nhãn trong tập train/val/test mới ===")
try:
    df_t = pd.read_csv('data/processed/03_train.csv')
    df_v = pd.read_csv('data/processed/03_val.csv')
    df_te = pd.read_csv('data/processed/03_test.csv')
    
    print("Việc chia split có stratified theo label (stratify=df_sach['label']) và random_state=42.")
    print("Tỷ lệ % train:")
    print(df_t['label'].value_counts(normalize=True)*100)
    print("Tỷ lệ % val:")
    print(df_v['label'].value_counts(normalize=True)*100)
    print("Tỷ lệ % test:")
    print(df_te['label'].value_counts(normalize=True)*100)
except Exception as e:
    print(f"Error: {e}")

print("\n=== 4. Kiểm tra trùng lặp giữa train/val/test ===")
try:
    # df_sach has clean text. We can check overlap in clean text or segment text.
    # We will just check if there's any exact overlap between the text_clean column of the splits
    clean_col = 'text_clean' if 'text_clean' in df_t.columns else ('text_segment' if 'text_segment' in df_t.columns else df_t.columns[1])
    set_train = set(df_t[clean_col].dropna())
    set_val = set(df_v[clean_col].dropna())
    set_test = set(df_te[clean_col].dropna())

    print(f"Trùng giữa train & val: {len(set_train.intersection(set_val))}")
    print(f"Trùng giữa train & test: {len(set_train.intersection(set_test))}")
    print(f"Trùng giữa val & test: {len(set_val.intersection(set_test))}")
except Exception as e:
    print(f"Error: {e}")

print("\n=== 5. 15 ví dụ nhãn 1 hoặc 2 ===")
try:
    sample_df = df_sach[df_sach['label'].isin([1, 2])].copy()
    
    clean_col_name = 'text_clean' if 'text_clean' in df_sach.columns else df_sach.columns[1]
    seg_col_name = 'text_segment' if 'text_segment' in df_sach.columns else df_sach.columns[2]
    
    def has_weird_chars(text):
        # Heuristic for teencode and filter evasion: numbers in words, asterisks, dots between letters, repeated chars
        t = str(text)
        if re.search(r'\w\*\*\w|\w\.\w\.\w', t): return 10
        if re.search(r'[*@#$\\/]', t): return 5
        return 0

    sample_df['priority'] = sample_df[sach_raw_col].apply(has_weird_chars)
    sample_df = sample_df.sort_values(by='priority', ascending=False)
    
    for idx, row in sample_df.head(15).iterrows():
        print(f"RAW: {row[sach_raw_col]}")
        print(f"CLEAN: {row.get(clean_col_name, 'N/A')}")
        print(f"SEGMENT: {row.get(seg_col_name, 'N/A')}")
        print("-" * 30)
except Exception as e:
    print(f"Error: {e}")
