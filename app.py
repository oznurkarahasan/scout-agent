"""
app.py — Scout Agent Streamlit UI
Çalıştır: streamlit run app.py
"""

from __future__ import annotations

import os
import sys

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.data.loader import load_and_normalize
from src.fuzzy.engine import score_all_ads, _norm_text
from src.llm import UserPreferences, score_ads_with_llm

RAW_DATA_PATH        = "data/ads.json"
NORMALIZED_DATA_PATH = "data/normalized_ads.json"

# ── Sayfa ayarları ────────────────────────────────────────────────────
st.set_page_config(
    page_title="Scout Agent",
    page_icon="🏠",
    layout="wide",
)

# ── Veriyi bir kez yükle ─────────────────────────────────────────────
@st.cache_data
def load_ads():
    return load_and_normalize(RAW_DATA_PATH, NORMALIZED_DATA_PATH, save=True)

ads = load_ads()
ads_map = {a["id"]: a for a in ads}
cities = sorted({a.get("city", "") for a in ads if a.get("city")})


# ══════════════════════════════════════════════════════════════════════
# SIDEBAR — Kullanıcı Girdileri
# ══════════════════════════════════════════════════════════════════════

st.sidebar.title("🔍 Arama Kriterleri")

city = st.sidebar.selectbox("Hedef Şehir", cities, index=cities.index("İstanbul") if "İstanbul" in cities else 0)

budget = st.sidebar.number_input(
    "Aylık Bütçe (TL)",
    min_value=1000,
    max_value=100_000,
    value=10_000,
    step=500,
)

prefs = st.sidebar.text_area(
    "Tercihleriniz",
    value="balkon, eşyalı, metro",
    help="Virgülle ayırın. Örn: 3+1, balkon, eşyalı, metro yakını",
)

only_city = st.sidebar.checkbox("Sadece bu şehirdeki ilanlar", value=True)

use_llm = st.sidebar.checkbox(
    "LLM Analizi",
    value=True,
    help="Groq API key varsa gerçek analiz, yoksa heuristic kullanılır.",
)

top_n = st.sidebar.slider("Gösterilecek ilan sayısı", 5, 30, 10)

search = st.sidebar.button("🚀 Ara", use_container_width=True)


# ══════════════════════════════════════════════════════════════════════
# ANA ALAN
# ══════════════════════════════════════════════════════════════════════

st.title("🏠 Scout Agent")
st.caption("Hibrit Bulanık Mantık + LLM Emlak Sıralama Sistemi")

if not search:
    st.info("Sol panelden kriterlerinizi girin ve **Ara** butonuna tıklayın.")
    st.stop()

# ── Pipeline ─────────────────────────────────────────────────────────
with st.spinner("İlanlar analiz ediliyor..."):
    target = [a for a in ads if _norm_text(a.get("city", "")) == _norm_text(city)] if only_city else ads

    if not target:
        st.warning(f"'{city}' için ilan bulunamadı. Tüm ilanlar değerlendiriliyor.")
        target = ads

    llm_scores: dict[str, float] = {}
    if use_llm:
        llm_scores = score_ads_with_llm(target, UserPreferences(free_text=prefs), use_remote_llm=None)

    results = score_all_ads(
        target,
        user_budget=budget,
        user_city=city,
        llm_scores=llm_scores or None,
        filter_city=False,
    )[:top_n]

if not results:
    st.error("Sonuç bulunamadı.")
    st.stop()

# ── Özet metrikler ───────────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)
col1.metric("Değerlendirilen İlan", len(target))
col2.metric("En Yüksek Skor", f"{results[0]['score']:.1f}")
col3.metric("Ortalama Skor", f"{sum(r['score'] for r in results) / len(results):.1f}")
groq_on = bool((os.getenv("GROQ_API_KEY") or "").strip())
col4.metric("LLM Modu", "Groq API" if (use_llm and groq_on) else ("Heuristic" if use_llm else "Kapalı"))

st.divider()

# ── Skor tablosu ─────────────────────────────────────────────────────
st.subheader("📋 Sıralama")

rows = []
for i, r in enumerate(results, 1):
    ad  = ads_map.get(r["id"], {})
    b   = r["breakdown"]
    llm = llm_scores.get(r["id"], b.get("llm_score_input", 0.5))
    rows.append({
        "#":        i,
        "Skor":     r["score"],
        "İlan":     r["title"],
        "Şehir":    f"{'✓' if b['city_match'] else '✗'} {ad.get('city', '?')}",
        "Fiyat":    f"{ad.get('price', 0):,} TL",
        "LLM":      round(llm, 2),
        "Yayıncı":  ad.get("publisher_type", "?"),
        "id":       r["id"],
    })

df = pd.DataFrame(rows)

st.dataframe(
    df.drop(columns=["id"]),
    use_container_width=True,
    hide_index=True,
    column_config={
        "Skor": st.column_config.ProgressColumn("Skor", min_value=0, max_value=100, format="%.1f"),
        "LLM":  st.column_config.ProgressColumn("LLM Uyum", min_value=0, max_value=1, format="%.2f"),
    },
)

st.divider()

# ── Detay görünümü ───────────────────────────────────────────────────
st.subheader("🔎 İlan Detayı")

selected_title = st.selectbox(
    "İlan seçin",
    options=[r["title"] for r in results],
    index=0,
)

selected = next(r for r in results if r["title"] == selected_title)
ad       = ads_map.get(selected["id"], {})
b        = selected["breakdown"]

col_left, col_right = st.columns([1, 1])

with col_left:
    st.markdown(f"**{selected['title']}**")
    st.write(f"📍 {ad.get('city', '?')} / {ad.get('district', '?')}")
    st.write(f"💰 {ad.get('price', 0):,} TL/ay")
    st.write(f"📅 {ad.get('days_since_posted', '?')} gün önce yayınlandı")
    st.write(f"📸 {ad.get('image_count', 0)} fotoğraf")
    st.write(f"👤 {ad.get('publisher_type', '?')}")
    st.write(f"📝 {ad.get('description', '')}")

with col_right:
    # Radar chart — breakdown görselleştirmesi
    def top_val(d: dict) -> float:
        return max(d.values())

    categories = ["Fiyat Uygunluğu", "Görsel Kalite", "Tazelik", "LLM Uyumu", "Yayıncı"]
    values = [
        1.0 - top_val({"e": b["price_memberships"].get("expensive", 0)}),  # expensive ne kadar düşükse o kadar iyi
        top_val(b["quality_memberships"]),
        top_val(b["fresh_memberships"]),
        b["llm_score_input"],
        b["publisher_score"],
    ]
    # 0-1 → 0-100
    values_pct = [v * 100 for v in values]

    fig = go.Figure(go.Scatterpolar(
        r=values_pct + [values_pct[0]],
        theta=categories + [categories[0]],
        fill="toself",
        fillcolor="rgba(99, 110, 250, 0.2)",
        line=dict(color="rgb(99, 110, 250)", width=2),
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        showlegend=False,
        margin=dict(l=40, r=40, t=40, b=40),
        height=320,
    )
    st.plotly_chart(fig, use_container_width=True)

    st.metric("Uygunluk Skoru", f"{selected['score']:.1f} / 100")
