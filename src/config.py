import os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SEED = 42

ENCODING = "utf-8-sig"

LABEL_NAMES = ["CLEAN", "OFFENSIVE", "HATE"]
TOXIC_IDS = [1, 2]
N_LABELS = len(LABEL_NAMES)

RAW_CSV = os.path.join(ROOT_DIR, "data", "raw", "comments.csv")
# Dữ liệu đối chứng (counter-examples): câu có tên riêng chứa âm tiết nhạy cảm
# (Ngu/Dốt/Khùng...) dùng trong ngữ cảnh trung tính/tích cực, gán CLEAN.
# Mục đích: giảm lexical bias - model học "thấy từ X -> độc hại" mà không
# xét ngữ cảnh. Chỉ thêm vào TRAIN, không thêm vào val/test (xem 01_process_data.py).
COUNTER_EXAMPLES_CSV = os.path.join(ROOT_DIR, "data", "raw", "counter_examples.csv")
TRAIN_CSV = os.path.join(ROOT_DIR, "data", "processed", "03_train.csv")
VAL_CSV = os.path.join(ROOT_DIR, "data", "processed", "03_val.csv")
TEST_CSV = os.path.join(ROOT_DIR, "data", "processed", "03_test.csv")

TEST_SIZE = 0.10
VAL_SIZE = 1 / 9  # val chiếm 10% tổng (train 80 / val 10 / test 10)

BASELINE_PKL = os.path.join(ROOT_DIR, "models", "baseline.pkl")
BASELINE_VAL_PRED = os.path.join(ROOT_DIR, "outputs", "baseline_val_predictions.csv")
BASELINE_EXP_CSV = os.path.join(ROOT_DIR, "outputs", "baseline_experiments.csv")
BASELINE_PARAMS_JSON = os.path.join(ROOT_DIR, "models", "baseline_params.json")
BASELINE_CM_VAL = os.path.join(ROOT_DIR, "outputs", "confusion_matrix_baseline_val.png")

N_CV_FOLDS = 3

TFIDF_WORD = dict(analyzer='word', ngram_range=(1, 2), min_df=3,
                  max_df=0.8, max_features=50000, sublinear_tf=True)
TFIDF_CHAR = dict(analyzer='char_wb', ngram_range=(2, 5), min_df=5,
                  max_df=0.8, max_features=30000, sublinear_tf=True)

PHOBERT_DIR = os.path.join(ROOT_DIR, "models", "phobert")
PHOBERT_BEST = os.path.join(PHOBERT_DIR, "best")
PHOBERT_VAL_PRED = os.path.join(ROOT_DIR, "outputs", "phobert_val_predictions.csv")
PHOBERT_TEST_PRED = os.path.join(ROOT_DIR, "outputs", "phobert_test_predictions.csv")

MAX_LEN = 128
PHOBERT_MODEL_NAME = "vinai/phobert-base"
PHOBERT_EPOCHS = 30
PHOBERT_TRAIN_BS = 16
PHOBERT_EVAL_BS = 32
PHOBERT_LR = 2e-5

COMPARE_CSV = os.path.join(ROOT_DIR, "outputs", "bang_so_sanh_mo_hinh.csv")
COMPARE_CM = os.path.join(ROOT_DIR, "outputs", "confusion_matrix_test.png")
ERROR_CSV = os.path.join(ROOT_DIR, "outputs", "phan_tich_loi.csv")

# Web demo
FEEDBACK_CSV = os.path.join(ROOT_DIR, "outputs", "feedback.csv")
WEB_MAX_CHARS = 500
