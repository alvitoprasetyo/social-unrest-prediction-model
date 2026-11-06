import easyocr
import requests

# Load model EasyOCR secara lazy loading/singleton agar memori hemat
_reader = None


def get_ocr_reader():
    """Memuat engine EasyOCR (Bahasa Indonesia & Inggris)"""
    global _reader
    if _reader is None:
        # gpu=False disesuaikan untuk lingkungan CPU Streamlit
        _reader = easyocr.Reader(["id", "en"], gpu=False)
    return _reader


def extract_text_from_url(image_url: str) -> str:
    """Mengekstrak teks dari URL gambar (dipakai di Tab Link Twitter)"""
    try:
        reader = get_ocr_reader()
        response = requests.get(image_url, timeout=10)
        if response.status_code == 200:
            results = reader.readtext(response.content, detail=0)
            return " ".join(results)
        return ""
    except Exception as e:
        print(f"[OCR ERROR - URL] Gagal membaca {image_url}: {e}")
        return ""


def extract_text_from_bytes(file_bytes: bytes) -> str:
    """Mengekstrak teks dari byte gambar (dipakai di Tab Input Manual)"""
    try:
        reader = get_ocr_reader()
        results = reader.readtext(file_bytes, detail=0)
        return " ".join(results)
    except Exception as e:
        print(f"[OCR ERROR - BYTES] Gagal membaca file unggahan: {e}")
        return ""


def combine_text_and_ocr(main_text: str, ocr_text_list: list) -> str:
    """Menggabungkan Teks Utama Tweet + Hasil Ekstraksi OCR Seluruh Gambar"""
    clean_main = main_text.strip() if main_text else ""
    valid_ocr = [t.strip() for t in ocr_text_list if t and t.strip()]

    if valid_ocr:
        ocr_combined = " ".join(valid_ocr)
        if clean_main:
            return f"{clean_main} {ocr_combined}"
        return ocr_combined

    return clean_main