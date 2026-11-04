import re, unicodedata
from pyvi import ViTokenizer

TEENCODE = {'ko': 'không', 'k': 'không', 'kh': 'không', 'hok': 'không', 'khong': 'không', 'hông': 'không', 'dc': 'được', 'đc': 'được', 'dk': 'được', 'duoc': 'được', 'vs': 'với', 'j': 'gì', 'ntn': 'như thế nào', 'mn': 'mọi người', 'mk': 'mình', 'mik': 'mình', 'bn': 'bao nhiêu', 'bt': 'bình thường', 'đt': 'điện thoại', 'sp': 'sản phẩm', 'ae': 'anh em', 'ad': 'admin', 'cmt': 'bình luận', 'trc': 'trước', 'r': 'rồi', 'z': 'vậy', 'wa': 'quá', 'qa': 'quá', 'thik': 'thích', 'iu': 'yêu', 'bit': 'biết', 'ok': 'ổn', 'oke': 'ổn', 'dm': 'đm', 'dcm': 'đcm', 'đg': 'đang', 'lm': 'làm', 'ng': 'người', 'nc': 'nói chuyện', 'ib': 'nhắn tin', 'tks': 'cảm ơn', 'thanks': 'cảm ơn'}
RE_HTML = re.compile(r'<[^>]+>')
RE_URL = re.compile(r'(https?://\S+|www\.\S+|\bt\.me/\S+)', re.I)
RE_MENTION = re.compile(r'@[A-Za-z0-9_.]+')
RE_EMOJI = re.compile(
    r'[\U0001F300-\U0001FAFF\u2600-\u27BF\U0001F1E6-\U0001F1FF\uFE0F\u200D\uFF00-\uFFEF]'
)
RE_CHEN_KT = re.compile(r'\b\w(?:[\.\-_]\w)+\b')
RE_TU = re.compile(r'\b(\w+)\b')

def fix_keo_dai(match):
    word = match.group(0)
    if word.isupper(): return word
    return re.sub(r'(.)\1{2,}', r'\1', word)

def lam_sach(text):
    t = unicodedata.normalize('NFC', str(text))
    t = RE_HTML.sub(' ', t)
    t = RE_URL.sub(' <URL> ', t)
    t = RE_MENTION.sub(' <MENTION> ', t)
    t = RE_EMOJI.sub(' <EMOJI> ', t)
    t = re.sub(r'\b\w+\b', fix_keo_dai, t)
    t = RE_CHEN_KT.sub(lambda m: re.sub(r'[\.\-_]', '', m.group()), t)
    t = t.lower()
    t = RE_TU.sub(lambda m: TEENCODE.get(m.group(1), m.group(1)), t)
    return re.sub(r'\s+', ' ', t).strip()

def tach_tu(text_sach):
    t = text_sach.replace('*', 'X_STAR_X')
    t = ViTokenizer.tokenize(t)
    return t.replace('X_STAR_X', '*')

def phat_hien(raw):
    raw = str(raw)
    counts = {
        'link': len(RE_URL.findall(raw)),
        'emoji': len(RE_EMOJI.findall(raw)),
        'chen_kt': len(RE_CHEN_KT.findall(raw)),
        'teencode': 0,
        'keo_dai': 0
    }
    t = raw.lower()
    for m in RE_TU.finditer(t):
        w = m.group(1)
        if w in TEENCODE:
            counts['teencode'] += 1
        if re.search(r'(.)\1{2,}', w) and not w.isupper():
            counts['keo_dai'] += 1
    return counts
