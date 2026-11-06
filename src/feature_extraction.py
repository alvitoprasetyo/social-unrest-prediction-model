import html
import re
import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer

MODEL_NAME = "indobenchmark/indobert-base-p2"
_tokenizer = None
_model = None


def split_camel_case(text: str) -> str:
    """Memecah CamelCase, Akronim Kapital, dan Angka sesuai Colab."""
    if not text:
        return ""
    s = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", str(text))
    s = re.sub(r"(?<=[A-Z])(?=[A-Z][a-z])", " ", s)
    s = re.sub(r"(?<=[a-zA-Z])(?=[0-9])|(?<=[0-9])(?=[a-zA-Z])", " ", s)
    return s


def clean_text(text: str) -> str:
    """Pembersihan teks presisi 1:1 sesuai fungsi cleaning_text di Colab."""
    if pd.isna(text) or not str(text).strip():
        return ""

    s = str(text)

    # 1. Unescape HTML Entities (&amp; -> &)
    s = html.unescape(s)

    # 2. Hapus URL
    s = re.sub(r"https?://\S+|www\.\S+", "", s)

    # 3. Ubah mention (@username) menjadi username murni
    s = re.sub(r"@(\w+)", r"\1", s)

    # 4. Ubah hashtag (#Topic) menjadi kata terpecah (CamelCase split)
    s = re.sub(r"#(\w+)", lambda m: split_camel_case(m.group(1)), s)

    # 5. Hapus karakter khusus (simpan alfanumerik & spasi)
    s = re.sub(r"[^a-zA-Z0-9\s]", " ", s)

    # 6. Case folding & normalisasi spasi
    s = re.sub(r"\s+", " ", s).strip().lower()

    return s


def is_garbage_token(word: str) -> bool:
    """Menyeleksi token bising/cacat hasil OCR berdasarkan struktur kata."""
    if len(word) < 2 and not word.isdigit():
        return True
    if len(word) > 3 and not re.search(r"[aiueo]", word):
        return True
    if re.search(r"[bcdfghjklmnpqrstvwxyz]{4,}", word):
        return True
    return False


def clean_ocr_text(text: str) -> str:
    """Pembersihan khusus teks OCR dengan filter garbage token sesuai Colab."""
    if pd.isna(text) or not str(text).strip():
        return ""
    s = clean_text(text)
    words = s.split()
    valid_words = [w for w in words if not is_garbage_token(w)]
    return " ".join(valid_words).strip()


def early_fusion(text_clean: str, ocr_clean: str) -> str:
    """Menggabungkan teks tweet dan OCR menggunakan tag [POSTER:] sesuai Colab."""
    t_str = "" if pd.isna(text_clean) else str(text_clean).strip()
    o_str = "" if pd.isna(ocr_clean) else str(ocr_clean).strip()

    if o_str and o_str != "nan":
        return f"{t_str} [POSTER:] {o_str}".strip()
    return t_str


def get_indobert_model():
    global _tokenizer, _model
    if _tokenizer is None or _model is None:
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        _model = AutoModel.from_pretrained(MODEL_NAME)
        _model.eval()
    return _tokenizer, _model


def extract_author_features(df: pd.DataFrame) -> np.ndarray:
    """Mengekstrak 4 fitur metadata author selaras dengan Colab."""
    if df.empty:
        return np.zeros((1, 4), dtype=np.float32)

    row = df.iloc[0]

    followers = row.get("author.followers", 0) or 0

    following = (
        row.get("author.following", 0)
        if "author.following" in row
        else row.get("author.friendsCount", 0)
    ) or 0
    ffratio = float(followers / (following + 1))

    age_dayss = 0.0

    is_blue = float(1.0 if row.get("author.isBlueVerified", False) else 0.0)

    statuses_count = float(
        row.get("author.statusesCount", 0)
        or row.get("author.tweetsCount", 0)
        or 0
    )

    return np.array(
        [[ffratio, age_dayss, is_blue, statuses_count]], dtype=np.float32
    )


def get_full_features(text_raw: str, df_scraped: pd.DataFrame) -> np.ndarray:
    """Proses utama ekstraksi 772 fitur (768 IndoBERT + 4 Metadata)."""
    tokenizer, model = get_indobert_model()

    inputs = tokenizer(
        text_raw,
        return_tensors="pt",
        max_length=128,
        truncation=True,
        padding=True,
    )

    with torch.no_grad():
        outputs = model(**inputs)

    cls_embedding = outputs.last_hidden_state[:, 0, :].numpy()  # [1, 768]
    author_features = extract_author_features(df_scraped)  # [1, 4]

    return np.hstack([cls_embedding, author_features])


def filter_top_engagement(df_raw: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
    """Menyaring data mentah batch ke Top N berbasis Engagement Score."""
    if df_raw.empty:
        return df_raw

    metrics = ["retweetCount", "likeCount", "replyCount", "quoteCount"]
    for col in metrics:
        if col in df_raw.columns:
            df_raw[col] = pd.to_numeric(df_raw[col], errors="coerce").fillna(0)
        else:
            df_raw[col] = 0

    df_raw["engagement_score"] = (
        (df_raw["retweetCount"] * 2)
        + df_raw["likeCount"]
        + df_raw["replyCount"]
        + (df_raw["quoteCount"] * 1.5)
    )

    df_sorted = df_raw.sort_values(
        by="engagement_score", ascending=False
    ).reset_index(drop=True)
    return df_sorted.head(top_n).copy()


def extract_action_topic(text_raw: str, ocr_clean: str = "") -> str:
    """Mengekstrak topik/isu aksi dari Hashtag atau keyphrase secara rule-based."""
    full_str = f"{text_raw} {ocr_clean}".strip()

    # 1. Cari Hashtag (Prioritas Utama)
    hashtags = re.findall(r"#(\w+)", full_str)
    if hashtags:
        extracted = [split_camel_case(ht) for ht in hashtags[:2]]
        return " | ".join([f"#{ht}" for ht in extracted])

    # 2. Keyphrase Parser (Jika Tanpa Hashtag)
    pattern = r"(?:tolak|kawal|seruan|aksi|demo|kembalikan|usut)\s+([a-zA-Z0-9\s]{3,30})"
    match = re.search(pattern, full_str, re.IGNORECASE)
    if match:
        return f"Isu: {match.group(0).title()}"

    return "Aksi/Isu Umum"