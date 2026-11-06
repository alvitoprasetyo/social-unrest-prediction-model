import os
import re
import pandas as pd
from apify_client import ApifyClient
from dotenv import load_dotenv

load_dotenv()

# Token API Apify
APIFY_TOKEN = os.getenv("APIFY_API_TOKEN")


def extract_tweet_id(url: str) -> str:
    """Ekstrak ID angka dari URL Twitter/X secara otomatis."""
    if not url:
        return None

    clean_url = str(url).strip()

    # Cari pola angka setelah /status/
    match = re.search(r"status/(\d+)", clean_url)
    if match:
        return match.group(1)

    # Jika input berupa angka ID langsung
    if clean_url.isdigit():
        return clean_url

    return None


def scrape_single_tweet(url: str, token: str = APIFY_TOKEN) -> pd.DataFrame:
    """Mengambil 1 tweet dari Apify untuk Mode 1 via ScrapeBadger."""
    tweet_id = extract_tweet_id(url)

    if not tweet_id:
        print(
            "❌ URL tidak valid. Pastikan formatnya seperti: https://x.com/user/status/123456789"
        )
        return pd.DataFrame()

    client = ApifyClient(token)
    run_input = {"mode": "Get Tweet by ID", "id": tweet_id}

    try:
        run = client.actor("scrape.badger/twitter-tweets-scraper").call(
            run_input=run_input
        )

        items = []
        for item in client.dataset(run.default_dataset_id).iterate_items():
            if isinstance(item, dict):
                items.append(item)

        if not items:
            return pd.DataFrame()

        df_normalized = pd.json_normalize(items)

        # ----------------------------------------------------------------------
        # MAPPING PRESISI 42 KOLOM SCRAPEBADGER KE FORMAT INTERNAL
        # ----------------------------------------------------------------------
        column_mapping = {
            "full_text": "text",
            "created_at": "createdAt",
            "user_followers_count": "author.followers",
            "user_following_count": "author.following",
            "user_tweet_count": "author.tweetsCount",
            "user_is_blue_verified": "author.isBlueVerified",
            "user_created_at": "author.createdAt",
            "user_name": "author.name",
            "retweet_count": "retweetCount",
        }

        for src_col, target_col in column_mapping.items():
            if src_col in df_normalized.columns and target_col not in df_normalized.columns:
                df_normalized[target_col] = df_normalized[src_col]

        # Memastikan 'text' selalu ada
        if "text" not in df_normalized.columns and "full_text" in df_normalized.columns:
            df_normalized["text"] = df_normalized["full_text"]

        essential_cols = [
            "text",
            "full_text",
            "media",
            "author.followers",
            "author.following",
            "author.createdAt",
            "author.isBlueVerified",
            "author.tweetsCount",
            "id",
            "url",
            "createdAt",
            "author.name",
            "retweetCount",
        ]

        available_cols = [col for col in essential_cols if col in df_normalized.columns]
        return df_normalized[available_cols]

    except Exception as e:
        print(f"❌ [SINGLE SCRAPER ERROR]: {e}")
        return pd.DataFrame()


def scrape_batch_tweets(
    keyword: str,
    start_date,
    end_date,
    max_items: int = 20,
    token: str = APIFY_TOKEN,
) -> pd.DataFrame:
    """Mengambil sekumpulan tweet berbasis Keyword & Tanggal via Aktor Kaito (kaitoeasyapi)."""
    if not keyword or not keyword.strip():
        return pd.DataFrame()

    if not token or "apify_api" not in token:
        print("[BATCH SCRAPER ERROR] Token Apify belum diisi atau tidak valid.")
        return pd.DataFrame()

    client = ApifyClient(token)

    # 1. Pastikan tanggal berformat string "YYYY-MM-DD" (kebal datetime.date)
    str_start = (
        start_date.strftime("%Y-%m-%d")
        if hasattr(start_date, "strftime")
        else str(start_date).strip()
    )
    str_end = (
        end_date.strftime("%Y-%m-%d")
        if hasattr(end_date, "strftime")
        else str(end_date).strip()
    )

    # 2. Formatting Keyword Fleksibel
    clean_kw = keyword.strip()
    if "," in clean_kw:
        formatted_kw = " OR ".join(
            [k.strip() for k in clean_kw.split(",") if k.strip()]
        )
    else:
        formatted_kw = clean_kw

    # 3. Query Tanggal Twitter
    search_query = f"{formatted_kw} since:{str_start} until:{str_end}"

    # 4. Payload Aktor Kaito
    run_input = {
        "searchTerms": [search_query],
        "maxItems": max_items,
        "sort": "Latest",
    }

    try:
        run = client.actor(
            "kaitoeasyapi/twitter-x-data-tweet-scraper-pay-per-result-cheapest"
        ).call(run_input=run_input)

        # 5. Penanganan dataset_id aman untuk semua versi SDK Apify
        dataset_id = None
        if isinstance(run, dict):
            dataset_id = run.get("defaultDatasetId") or run.get("default_dataset_id")
        else:
            dataset_id = (
                getattr(run, "default_dataset_id", None)
                or getattr(run, "defaultDatasetId", None)
                or (run.get("defaultDatasetId") if hasattr(run, "get") else None)
            )

        if not dataset_id:
            print("[BATCH SCRAPER ERROR] Dataset ID dari Apify bernilai None.")
            return pd.DataFrame()

        raw_items = list(client.dataset(dataset_id).iterate_items())

        items = []
        for item in raw_items:
            if isinstance(item, dict):
                tweet_txt = (
                    item.get("text")
                    or item.get("full_text")
                    or item.get("fullText")
                    or item.get("tweet_text")
                    or ""
                )
                if isinstance(tweet_txt, str) and tweet_txt.strip():
                    items.append(item)

        if not items:
            print("[BATCH SCRAPER] Tidak ada tweet yang sesuai kriteria.")
            return pd.DataFrame()

        df_normalized = pd.json_normalize(items)

        # 6. Pastikan kolom 'text' utama selalu ada untuk pipeline downstream
        if "text" not in df_normalized.columns:
            for alt_col in ["full_text", "fullText", "tweet_text"]:
                if alt_col in df_normalized.columns:
                    df_normalized["text"] = df_normalized[alt_col]
                    break

        # 7. Deduplikasi: Hapus tweet ganda berbasis teks
        if "text" in df_normalized.columns:
            df_normalized = df_normalized.drop_duplicates(subset=["text"]).reset_index(drop=True)

        return df_normalized

    except Exception as e:
        print(f"[BATCH SCRAPER ERROR] Gagal mengambil data: {e}")
        return pd.DataFrame()