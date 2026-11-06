import warnings

warnings.filterwarnings("ignore")

import datetime
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from src.feature_extraction import (
    clean_ocr_text,
    clean_text,
    early_fusion,
    extract_action_topic,
    filter_top_engagement,
    get_full_features,
)
from src.ocr import extract_text_from_url
from src.scrappers import scrape_batch_tweets, scrape_single_tweet

# THRESHOLD DECISION RISK INTERNAL
INTERNAL_THRESHOLD = 0.3

# TITLE & CONFIG STREAMLIT
st.set_page_config(page_title="Twitter Risk Classifier", layout="wide")
st.title("Prototype of X Social Unrest Risk Classifier")


# SCALER & MODELS LOAD
@st.cache_resource
def load_models_and_scaler():
  """Memuat scaler dan model trained dari folder models/"""
  scaler = joblib.load("models/scaler.pkl")
  svm_model = joblib.load("models/model_svm.pkl")
  rf_model = joblib.load("models/model_rf.pkl")
  xgb_model = joblib.load("models/model_xgb.pkl")

  return scaler, svm_model, xgb_model, rf_model


scaler, svm_model, xgb_model, rf_model = load_models_and_scaler()


# HELPER 1: MAPPING PROBABILITAS (THRESHOLD 0.3 -> SKALA DISPLAY 50%)
def scale_prob_for_display(raw_prob, internal_thresh=INTERNAL_THRESHOLD):
  """Memetakan nilai probabilitas asli sehingga internal threshold (0.3)

  tampil persis sebagai 0.5 (50%) pada antarmuka.
  """
  if raw_prob < internal_thresh:
    return (raw_prob / internal_thresh) * 0.499
  else:
    return 0.5 + ((raw_prob - internal_thresh) / (1.0 - internal_thresh)) * 0.5


# HELPER 2: SMOOTHING PROBABILITAS XGBOOST
def smooth_xgb_prob(raw_prob, power=0.4):
  """Mengangkat probabilitas XGBoost yang tertekan ekstrem mendekati 0

  menggunakan power scaling (akar halus) agar distribusinya lebih seimbang.
  """
  return float(np.power(raw_prob, power))


# HELPER UTAMA: EXTRACTION OCR DARI ROW MEDIA
def extract_ocr_from_media(media_col_value) -> list:
  """Helper untuk mengambil URL gambar dan mengekstrak OCR dari kolom media."""
  ocr_texts = []
  image_urls = []

  if isinstance(media_col_value, list):
    for item in media_col_value:
      if isinstance(item, dict):
        url = (
            item.get("media_url_https")
            or item.get("url")
            or item.get("display_url")
            or ""
        )
        if (
            item.get("type") == "photo"
            or "jpg" in url
            or "png" in url
            or "jpeg" in url
            or "webp" in url
        ):
          if url:
            image_urls.append(url)
      elif isinstance(item, str) and item.startswith("http"):
        image_urls.append(item)

  elif isinstance(media_col_value, str) and media_col_value.startswith("http"):
    image_urls.append(media_col_value)

  for img_url in image_urls:
    ocr_res = extract_text_from_url(img_url)
    if ocr_res and ocr_res.strip():
      ocr_texts.append(ocr_res)

  return image_urls, ocr_texts


# MODELS PREDICTION WITH PROBABILITY RESCALING & XGB SMOOTHING
def predict_risk_scores(features_772, threshold=INTERNAL_THRESHOLD):
  """Scaling 772 Fitur -> Prediksi Klasifikasi Risiko & Mapping Probabilitas Display"""
  scaled_features = scaler.transform(features_772)

  svm_raw = float(svm_model.predict_proba(scaled_features)[0][1])
  xgb_raw = float(xgb_model.predict_proba(scaled_features)[0][1])
  rf_raw = float(rf_model.predict_proba(scaled_features)[0][1])

  xgb_adjusted = smooth_xgb_prob(xgb_raw, power=0.4)

  svm_disp = scale_prob_for_display(svm_raw, threshold)
  xgb_disp = scale_prob_for_display(xgb_adjusted, threshold)
  rf_disp = scale_prob_for_display(rf_raw, threshold)

  return {
      "SVM": {
          "risk": "Beresiko" if svm_raw >= threshold else "Tidak Beresiko",
          "score": svm_disp,
      },
      "XGBoost": {
          "risk": "Beresiko" if xgb_adjusted >= threshold else "Tidak Beresiko",
          "score": xgb_disp,
      },
      "Random Forest": {
          "risk": "Beresiko" if rf_raw >= threshold else "Tidak Beresiko",
          "score": rf_disp,
      },
  }


def display_model_results(results, topic_label=""):
  """Fungsi tampilan kesimpulan standar dengan Persentase Probabilitas Terkalibrasi (Base 50%)"""
  st.subheader("Hasil Prediksi Risiko Model", anchor=False)
  col1, col2, col3 = st.columns(3)

  models = [("SVM", col1), ("XGBoost", col2), ("Random Forest", col3)]
  is_any_risky = False

  for name, col in models:
    res = results[name]
    score_pct = res.get("score", 0.0) * 100

    with col:
      if res["risk"] == "Beresiko":
        is_any_risky = True
        st.error(f"**Model {name}**")
        st.markdown(
            f"<p style='font-size:22px; font-weight:bold; color:#ff4b4b;"
            f" margin:0;'>Beresiko ({score_pct:.1f}%)</p>",
            unsafe_allow_html=True,
        )
      else:
        st.success(f"**Model {name}**")
        st.markdown(
            f"<p style='font-size:22px; font-weight:bold; color:#21c354;"
            f" margin:0;'>Tidak Beresiko ({score_pct:.1f}%)</p>",
            unsafe_allow_html=True,
        )

  if is_any_risky and topic_label:
    st.markdown("---")
    st.info(f"**Topik:** `{topic_label}`")


def display_ensemble_summary(results, is_batch=False, total_tweets=1):
  """Menampilkan narasi kesimpulan berdasarkan konsensus mayoritas (Ensemble Voting)

  yang disesuaikan secara otomatis untuk analisis Single maupun Batch.
  """
  risky_models = [m for m, res in results.items() if res["risk"] == "Beresiko"]
  safe_models = [
      m for m, res in results.items() if res["risk"] == "Tidak Beresiko"
  ]

  count_risky = len(risky_models)
  scores = [res["score"] * 100 for res in results.values()]
  avg_score = sum(scores) / len(scores)

  st.markdown("---")
  st.subheader("Kesimpulan")

  if count_risky >= 2:
    status = "BERESIKO"
    model_str = " dan ".join(risky_models)

    if count_risky == 3:
      detail_txt = (
        f"Ketiga model (**XGBoost, Random Forest, dan SVM**) secara konsisten"
        f" menyatakan status **Beresiko** dengan rata-rata tingkat keyakinan"
        f" sebesar **{avg_score:.1f}%**."
      )
    else:
      risky_scores = [results[m]["score"] * 100 for m in risky_models]
      min_confidence = min(risky_scores)
      detail_txt = (
          f"Dua dari tiga model (**{model_str}**) sama-sama mengindikasikan"
          f" kondisi beresiko dengan tingkat keyakinan di atas"
          f" **{min_confidence:.1f}%**."
      )

    if is_batch:
      summary_msg = (
          f"**Secara mayoritas hasil analisis batch menyatakan"
          f" status {status}.** {detail_txt} pada tingkat sampel agregat,"
          f" ditemukan potensi indikasi provokasi atau mobilisasi massa yang"
          f" perlu diwaspadai."
      )
    else:
      summary_msg = (
          f"**Secara mayoritas, hasil prediksi cenderung menyatakan"
          f" status {status}.** {detail_txt}"
      )

    st.error(summary_msg)

  else:
    status = "TIDAK BERESIKO"
    safe_str = " dan ".join(safe_models)

    if len(safe_models) == 3:
      detail_txt = (
          f"Ketiga model (**XGBoost, Random Forest, dan SVM**) secara konsisten"
          f" mengategorikan konten sebagai **Tidak Beresiko** dengan rata-rata"
          f" tingkat keyakinan aman sebesar **{100 - avg_score:.1f}%**."
      )
    else:
      detail_txt = (
          f"Dua dari tiga model (**{safe_str}**) mengategorikan konten sebagai"
          f" aman / tidak beresiko."
      )

    if is_batch:
      summary_msg = (
          f"**Secara mayoritas (ensemble), hasil analisis batch menunjukkan"
          f" kondisi {status}.** {detail_txt} Seluruh atau sebagian besar"
          f" sampel tweet pada rentang waktu ini bersifat netral atau berupa"
          f" tanggapan publik umum."
      )
    else:
      summary_msg = (
          f"**Secara mayoritas (ensemble), hasil prediksi menunjukkan kondisi"
          f" {status}.** {detail_txt}"
      )

    st.success(summary_msg)


def display_compact_summary(
    df_results, keyword, model_max_scores, detected_topics
):
  """Menampilkan ringkasan hasil analisis batch format ringkas dan intuitif."""
  total_tweets = len(df_results)
  if total_tweets == 0:
    return

  risky_tweets_count = sum(
      1
      for _, row in df_results.iterrows()
      if "Beresiko" in str(row["SVM"])
      or "Beresiko" in str(row["XGBoost"])
      or "Beresiko" in str(row["Random Forest"])
  )

  risky_pct = (
      (risky_tweets_count / total_tweets) * 100 if total_tweets > 0 else 0
  )
  overall_max_score = max(model_max_scores.values()) * 100
  is_overall_risky = overall_max_score >= 50.0

  hashtags = [
      t for t in detected_topics if isinstance(t, str) and t.startswith("#")
  ]
  top_hashtags = (
      ", ".join(sorted(list(set(hashtags)))[:3])
      if hashtags
      else "Tidak ada tagar khusus"
  )

  status_label = (
      f"**BERESIKO (Max: {overall_max_score:.1f}%)**"
      if is_overall_risky
      else f"**TIDAK BERESIKO (Max: {overall_max_score:.1f}%)**"
  )

  st.markdown("---")
  st.markdown("### **Summary**")
  st.markdown(f"- **Keyword:** '{keyword}' ({total_tweets} Tweet)")
  st.markdown(f"- **Status Agregat Peak:** {status_label}")
  st.markdown(
      f"- **Rasio Deteksi:** {risky_tweets_count} dari {total_tweets} tweet"
      f" ({risky_pct:.0f}%) terindikasi provokasi aksi."
  )
  st.markdown(f"- **Tagar Terkait:** {top_hashtags}")

  st.markdown("")
  if is_overall_risky:
    st.info(
        "**Sistem menggunakan metode"
        " Worst-Case Scenario untuk pencegahan dini.**"
    )
  else:
    st.success(
        "**Sistem menandai kondisi aman karena tidak terdeteksi indikasi"
        " provokasi aksi pada sampel data.**"
    )


# ==============================================================================
# TABS INTERFACE
# ==============================================================================
tab1, tab2 = st.tabs(["Single URL", "Keyword"])

# ------------------------------------------------------------------------------
# TAB 1: SINGLE TWEET URL
# ------------------------------------------------------------------------------
with tab1:
  url_input = st.text_input(
      label="Masukkan Link:",
      placeholder="https://x.com/username/status/...",
      key="url_input",
  )

  if st.button("Analisis Tweet", key="btn_scrape_link"):
    if not url_input.strip():
      st.warning("Silakan masukkan URL tweet terlebih dahulu.")
    else:
      with st.spinner(
          "Memproses tweet, ekstraksi OCR & menganalisis risiko..."
      ):
        try:
          df = scrape_single_tweet(url_input)

          if not df.empty:
            tweet_text = df["text"].iloc[0] if "text" in df.columns else ""

            st.subheader("Pratinjau:")
            st.info(tweet_text if tweet_text else "*(Tidak ada teks)*")

            # 1. Pipeline OCR
            media_val = df["media"].iloc[0] if "media" in df.columns else None
            image_urls, ocr_texts = extract_ocr_from_media(media_val)

            if image_urls:
              st.markdown(f"**Gambar Terlampir ({len(image_urls)}):**")
              cols = st.columns(
                  [1] * len(image_urls) + [max(1, 4 - len(image_urls))]
              )
              for idx, img_url in enumerate(image_urls):
                cols[idx].image(img_url, caption=f"Gambar {idx+1}", width=200)

            # 2. Pipeline Cleaning + Garbage Filter OCR + Early Fusion
            text_clean = clean_text(tweet_text)
            ocr_clean = clean_ocr_text(" ".join(ocr_texts))
            final_text = early_fusion(text_clean, ocr_clean)

            # 3. IndoBERT Embedding + Metadata (772 Fitur)
            features_772 = get_full_features(final_text, df)

            st.divider()

            # 4. Modeling & Result
            predictions = predict_risk_scores(features_772)
            topic_info = extract_action_topic(tweet_text, ocr_clean)

            # Tampilan Hasil Model & Kesimpulan Ensemble
            display_model_results(predictions, topic_label=topic_info)
            display_ensemble_summary(
                predictions, is_batch=False, total_tweets=1
            )

          else:
            st.warning("Tweet tidak ditemukan, terhapus, atau privat.")
        except Exception as e:
          st.error(f"Terjadi kesalahan: {e}")

# ------------------------------------------------------------------------------
# TAB 2: BATCH KEYWORD & DATE RANGE
# ------------------------------------------------------------------------------
with tab2:
  st.subheader("Analisis Batch berbasis Kata Kunci & Rentang Tanggal")

  col1, col2, col3 = st.columns([2, 2, 1])
  with col1:
    keyword = st.text_input(
        "Kata Kunci:",
        placeholder="Masukkan Keywords",
        key="batch_keyword",
    )
  with col2:
    today = datetime.date.today()
    default_start = today - datetime.timedelta(days=7)
    date_range = st.date_input(
        "Rentang Tanggal:", value=(default_start, today), key="batch_date"
    )
  with col3:
    max_tweets = st.selectbox(
        "Jumlah Twit:",
        options=[10, 20, 30, 40, 50],
        index=1,
        key="batch_max_items",
    )

  if st.button("Analisis", key="btn_batch"):
    if not keyword:
      st.warning("Harap masukkan kata kunci terlebih dahulu.")
    elif len(date_range) < 2:
      st.warning("Harap pilih rentang tanggal lengkap (Mulai dan Selesai).")
    else:
      start_d, end_d = date_range[0], date_range[1]
      days_diff = (end_d - start_d).days

      if days_diff > 7:
        st.error(
            f"Rentang tanggal yang Anda pilih adalah **{days_diff} hari**."
            " Maksimal rentang tanggal disarankan **7 hari**."
        )
      else:
        with st.spinner("Mengambil data..."):
          df_raw = scrape_batch_tweets(
              keyword, start_d, end_d, max_items=max_tweets
          )

        if df_raw.empty:
          st.warning(
              "Tidak ditemukan tweet untuk kata kunci dan rentang tanggal"
              " tersebut."
          )
        else:
          df_top = filter_top_engagement(df_raw, top_n=max_tweets)

          results_list = []
          progress_bar = st.progress(0)
          total_items = len(df_top)

          detected_topics = set()
          model_max_scores = {"SVM": 0.0, "XGBoost": 0.0, "Random Forest": 0.0}

          for idx, row in df_top.iterrows():
            try:
              t_text = (
                  row.get("text")
                  or row.get("full_text")
                  or row.get("twitterContent")
                  or ""
              )

              if not isinstance(t_text, str) or not t_text.strip():
                progress_bar.progress((idx + 1) / total_items)
                continue

              row_df = pd.DataFrame([row])

              # 1. OCR Ekstraksi dari Gambar
              media_val = row.get("media", None)
              image_urls, ocr_texts = extract_ocr_from_media(media_val)

              # 2. Cleaning & Early Fusion
              t_clean = clean_text(t_text)
              ocr_clean = clean_ocr_text(" ".join(ocr_texts))
              final_fused_text = early_fusion(t_clean, ocr_clean)

              # 3. Fitur IndoBERT (772) & Predict Model ML
              f_772 = get_full_features(final_fused_text, row_df)
              preds = predict_risk_scores(f_772)
              topic = extract_action_topic(t_text, ocr_clean)

              is_risky = False
              for m_name in ["SVM", "XGBoost", "Random Forest"]:
                score = preds[m_name]["score"]
                if score > model_max_scores[m_name]:
                  model_max_scores[m_name] = score

                if preds[m_name]["risk"] == "Beresiko":
                  is_risky = True

              if is_risky and topic and topic != "Aman":
                detected_topics.add(topic)

              # PARSING TANGGAL & JAM (YYYY-MM-DD HH:MM)
              raw_date = row.get("createdAt", "") or row.get("created_at", "")
              try:
                dt_obj = pd.to_datetime(raw_date)
                formatted_datetime = dt_obj.strftime("%Y-%m-%d %H:%M")
              except Exception:
                formatted_datetime = (
                    str(raw_date)[:16] if raw_date else "1970-01-01 00:00"
                )

              results_list.append({
                  "Waktu": formatted_datetime,
                  "Teks Tweet": t_text,
                  "Gambar": image_urls if image_urls else None,
                  "SVM": (
                      "Beresiko"
                      if preds["SVM"]["risk"] == "Beresiko"
                      else "Tidak Beresiko"
                  ),
                  "XGBoost": (
                      "Beresiko"
                      if preds["XGBoost"]["risk"] == "Beresiko"
                      else "Tidak Beresiko"
                  ),
                  "Random Forest": (
                      "Beresiko"
                      if preds["Random Forest"]["risk"] == "Beresiko"
                      else "Tidak Beresiko"
                  ),
              })
            except Exception as err:
              print(f"[INFERENCE ERROR] Gagal memproses baris {idx}: {err}")

            progress_bar.progress((idx + 1) / total_items)

          if results_list:
            df_final_results = pd.DataFrame(results_list)

            # URUTKAN TWEET BERDESARKAN WAKTU TERBARU (DESCENDING)
            df_final_results = df_final_results.sort_values(
                by="Waktu", ascending=False
            ).reset_index(drop=True)

            # ------------------------------------------------------------------
            # TABEL RINGKASAN BATCH
            # ------------------------------------------------------------------
            st.divider()
            st.subheader("Tabel Ringkasan Analisis Risiko Batch")

            st.dataframe(
                df_final_results,
                column_config={
                    "Waktu": st.column_config.TextColumn(
                        "Waktu", width="medium"
                    ),
                    "Teks Tweet": st.column_config.TextColumn(
                        "Teks Tweet", width="large"
                    ),
                    "Gambar": st.column_config.ListColumn(
                        "Gambar",
                        width="medium",
                        help="Daftar URL gambar terlampir",
                    ),
                    "SVM": st.column_config.TextColumn("SVM", width="small"),
                    "XGBoost": st.column_config.TextColumn(
                        "XGBoost", width="small"
                    ),
                    "Random Forest": st.column_config.TextColumn(
                        "Random Forest", width="small"
                    ),
                },
                use_container_width=True,
                hide_index=True,
            )

            # ------------------------------------------------------------------
            # KESIMPULAN AGREGAT MODEL (KARTU STATISTIK, ENSEMBLE & INDIKATOR)
            # ------------------------------------------------------------------
            st.markdown("---")

            batch_predictions = {
                "SVM": {
                    "risk": (
                        "Beresiko"
                        if model_max_scores["SVM"] >= 0.5
                        else "Tidak Beresiko"
                    ),
                    "score": model_max_scores["SVM"],
                },
                "XGBoost": {
                    "risk": (
                        "Beresiko"
                        if model_max_scores["XGBoost"] >= 0.5
                        else "Tidak Beresiko"
                    ),
                    "score": model_max_scores["XGBoost"],
                },
                "Random Forest": {
                    "risk": (
                        "Beresiko"
                        if model_max_scores["Random Forest"] >= 0.5
                        else "Tidak Beresiko"
                    ),
                    "score": model_max_scores["Random Forest"],
                },
            }

            hashtags_only = [
                t
                for t in detected_topics
                if isinstance(t, str) and t.startswith("#")
            ]

            if hashtags_only:
              top_hashtags = " | ".join(sorted(list(set(hashtags_only)))[:3])
              batch_topic_label = f"{keyword} ({top_hashtags})"
            else:
              batch_topic_label = f"{keyword}"

            display_model_results(
                batch_predictions, topic_label=batch_topic_label
            )

            # Kesimpulan Mayoritas Ensemble
            display_ensemble_summary(
                batch_predictions, is_batch=True, total_tweets=len(df_final_results)
            )

            # Indikator Ringkas
            display_compact_summary(
                df_final_results, keyword, model_max_scores, detected_topics
            )

          else:
            st.error("Gagal memproses inferensi model pada dataset terambil.")