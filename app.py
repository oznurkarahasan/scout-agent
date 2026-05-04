import streamlit as st
import json
import re
import os
from src.fuzzy.engine import ScoutFuzzyEngine

# Page Config
st.set_page_config(page_title="Scout Agent: Profesyonel İlan Tarayıcı", layout="wide")

# Custom CSS for Realistic Branding and Card Styling
st.markdown("""
<style>
    .ad-card {
        background-color: #ffffff;
        color: #333;
        padding: 20px;
        border-radius: 12px;
        margin-bottom: 25px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        border-left: 10px solid #ddd;
        transition: transform 0.2s;
    }
    .ad-card:hover {
        transform: translateY(-5px);
    }
    .source-badge {
        padding: 4px 10px;
        border-radius: 4px;
        font-size: 12px;
        font-weight: bold;
        text-transform: uppercase;
        margin-bottom: 8px;
        display: inline-block;
    }
    .source-sahibinden { background-color: #ffe800; color: #000; }
    .source-hepsiemlak { background-color: #e30613; color: #fff; }
    .source-emlakjet { background-color: #0089cf; color: #fff; }
    
    .score-badge {
        font-size: 28px;
        font-weight: 800;
        padding: 10px 20px;
        border-radius: 8px;
    }
    .green-badge { background-color: #d4edda; color: #155724; border: 2px solid #c3e6cb; }
    .red-badge { background-color: #f8d7da; color: #721c24; border: 2px solid #f5c6cb; }
    
    .price-tag { font-size: 22px; font-weight: bold; color: #2c3e50; }
    .meta-info { color: #7f8c8d; font-size: 14px; margin-top: 5px; }
</style>
""", unsafe_allow_html=True)

# 1. Initialize Engine
if 'engine' not in st.session_state or not hasattr(st.session_state.engine, 'prepare'):
    st.session_state.engine = ScoutFuzzyEngine()

if 'search_clicked' not in st.session_state:
    st.session_state.search_clicked = False

engine = st.session_state.engine

# 2. Sidebar - Advanced Search Criteria
st.sidebar.title("🏢 Arama Filtreleri")

if st.sidebar.button("🔍 Veri Setinden Getir", use_container_width=True, type="primary"):
    # Trigger a fresh run
    st.cache_data.clear()
    st.session_state.search_clicked = True

st.sidebar.subheader("🏠 Emlak Bilgileri")
target_listing_type = st.sidebar.radio("İlan Tipi", ["Hepsi", "Kiralık", "Satılık"], horizontal=True)

st.sidebar.subheader("💰 Fiyat Aralığı (TL)")
# Dynamic price ranges based on listing type
if target_listing_type == "Satılık":
    min_price, max_price = st.sidebar.slider("Bütçe Seçimi", 500000, 20000000, (1000000, 5000000), step=100000)
elif target_listing_type == "Kiralık":
    min_price, max_price = st.sidebar.slider("Bütçe Seçimi", 2000, 100000, (10000, 30000), step=500)
else: # Hepsi
    min_price, max_price = st.sidebar.slider("Bütçe Seçimi", 2000, 20000000, (10000, 5000000), step=5000)

st.sidebar.subheader("📐 Büyüklük (m²)")
min_m2, max_m2 = st.sidebar.slider("Metrekare Aralığı", 0, 1000, (75, 200), step=5)

st.sidebar.subheader("🛏️ Oda Sayısı")
room_options = ["Hepsi", "1+1", "2+1", "3+1", "4+1"]
target_rooms = st.sidebar.multiselect("Tercih Edilen Oda Sayısı", room_options, default=["Hepsi"])

st.sidebar.divider()
st.sidebar.title("🎯 Senin Önceliklerin")
w_p = st.sidebar.slider("Fiyat Uyumluluğu", 0.0, 1.0, 0.9)
w_l = st.sidebar.slider("Konum Skoru", 0.0, 1.0, 0.7)
w_s = st.sidebar.slider("m² Uyumu", 0.0, 1.0, 0.6)
w_q = st.sidebar.slider("İlan Görselleri/Kalite", 0.0, 1.0, 0.5)
w_m = st.sidebar.slider("Metin Analizi (LLM)", 0.0, 1.0, 0.4)

priorities = {
    'price': w_p,
    'location': w_l,
    'size': w_s,
    'quality': w_q,
    'llm': w_m
}

# 3. Load Data
@st.cache_data(show_spinner="📂 Veri seti yükleniyor...")
def load_ads(mtime):
    # This is the single source of truth requested by the user
    file_path = 'data/normalized_ads.json'
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return []

def get_file_mtime(path):
    return os.path.getmtime(path) if os.path.exists(path) else 0

# Single Source Loading
ads = load_ads(get_file_mtime('data/normalized_ads.json'))

def get_district_name(raw_district):
    district = (raw_district or "").strip()
    if " - " in district:
        district = district.split(" - ", 1)[0].strip()
    district = re.split(r"\s{2,}", district)[0].strip()
    return district

city_options = sorted({ad.get('city', '').strip() for ad in ads if ad.get('city')})
if not city_options:
    city_options = ["İstanbul", "Ankara", "İzmir", "Bursa", "Antalya"]

target_city = st.sidebar.selectbox("📍 Şehir", city_options)
district_options = sorted({
    get_district_name(ad.get('district', ''))
    for ad in ads
    if ad.get('city') == target_city and ad.get('district')
})
target_district = st.sidebar.selectbox("🔍 İlçe Seç", ["Hepsi"] + district_options)
district_filter = "" if target_district == "Hepsi" else target_district

# Helper for UI display
def calculate_suitability_ratios(ad, min_p, max_p, target_city, target_district, min_m=0, max_m=1000):
    # Price
    if min_p <= ad['price'] <= max_p: p_suit = 100
    elif ad['price'] < min_p: p_suit = 100
    else: p_suit = max(0, 100 - ((ad['price'] - max_p) / max_p) * 200)
    
    # Location
    l_score = 0
    if ad['city'] == target_city:
        l_score = 7
        if target_district and target_district.lower() == get_district_name(ad.get('district', '')).lower():
            l_score = 10
    else: l_score = 2
    
    # Size
    ad_m2 = ad.get('area_m2', 100)
    if min_m <= ad_m2 <= max_m: s_suit = 100
    else:
        dist = min(abs(ad_m2 - min_m), abs(ad_m2 - max_m))
        s_suit = max(0, 100 - (dist / max(1, min_m)) * 100)
        
    return p_suit, l_score, s_suit

# 4. Scoring Logic with Range Support
def calculate_ad_score(ad, min_p, max_p, target_city, target_district, target_rooms, engine):
    # a. Price Suitability (0-100)
    if min_p <= ad['price'] <= max_p:
        # Favor cheaper even within range
        p_suit = 100 - ((ad['price'] - min_p) / (max_p - min_p + 1)) * 30
    elif ad['price'] < min_p:
        p_suit = 100 # Cheaper is still good!
    else:
        # Over max: aggressive penalty
        p_suit = max(0, 70 - ((ad['price'] - max_p) / max_p) * 200)
    
    # b. Location Score (0-10)
    l_score = 8 # Base for matching city
    if target_district and target_district.lower() == get_district_name(ad.get('district', '')).lower():
        l_score = 10
        
    # c. Size Suitability (0-100)
    ad_m2 = ad.get('area_m2', 100)
    if min_m2 <= ad_m2 <= max_m2:
        # Center of range is ideal
        center = (min_m2 + max_m2) / 2
        dist_from_center = abs(ad_m2 - center)
        s_suit = 100 - (dist_from_center / ((max_m2 - min_m2 + 1) / 2)) * 20
    else:
        s_suit = 0

    # d. Quality Score (0-10)
    q_score = min(10, ad['image_count'] * 2 + (1 if len(ad['description']) > 150 else 0))
    
    # e. Room Match
    m_score = 5
    if "Hepsi" not in target_rooms:
        if ad.get('room_count') in target_rooms:
            m_score = 10
        else:
            m_score = 0
    
    inputs = {
        'price_suitability': p_suit,
        'location_score': l_score,
        'listing_quality': q_score,
        'size_suitability': s_suit,
        'llm_alignment': m_score
    }
    
    score, sim = engine.compute(inputs)
    return score, sim, inputs

# 5. Process and Filter
engine.prepare(priorities)

scored_ads = []
for ad in ads:
    # --- Hard Filters (Strict) ---
    
    # 1. City
    if ad.get('city') != target_city:
        continue
        
    # 2. Listing Type
    if target_listing_type != "Hepsi":
        if ad.get('listing_type') != target_listing_type:
            continue
            
    # 3. Room Count (Now using the dedicated field)
    if "Hepsi" not in target_rooms:
        if ad.get('room_count') not in target_rooms:
            continue
    
    # 4. District (Hard Filter if provided)
    if district_filter and district_filter.lower() != get_district_name(ad.get('district', '')).lower():
        continue

    # 5. Price (Hard Range)
    price = ad.get('price', 0)
    if price == 0 or price < min_price or price > max_price:
        continue

    # 6. Area m2 (Hard Range)
    ad_m2 = ad.get('area_m2', 100)
    if ad_m2 < min_m2 or ad_m2 > max_m2:
        continue

    # --- Scoring & Ranking ---
    score, sim, fuzzy_inputs = calculate_ad_score(ad, min_price, max_price, target_city, district_filter, target_rooms, engine)
    
    ad_copy = ad.copy()
    ad_copy['scout_score'] = score
    ad_copy['fuzzy_sim'] = sim
    ad_copy['fuzzy_inputs'] = fuzzy_inputs
    ad_copy['llm_score'] = ad.get('llm_score')
    scored_ads.append(ad_copy)

# Sort Descending by Scout Score
scored_ads.sort(key=lambda x: x['scout_score'], reverse=True)

# 6. UI Rendering
st.title("🏹 Scout Agent: Zeki Emlak Bulucu")
st.write(f"🔍 **{target_city}** bölgesinde **{len(scored_ads)}** uygun ilan bulundu.")
st.caption(f"Filtreler: {target_listing_type} | {min_price:,} - {max_price:,} TL | {', '.join(target_rooms)}")

if not scored_ads:
    st.warning(f"Aradığınız kriterlerde {target_city} şehrinde ilan bulunamadı. Lütfen filtreleri esnetmeyi deneyin.")
else:
    for ad in scored_ads[:20]: # Show top 20
        score = ad['scout_score']
        is_green = score >= 50
        badge_color = "#28a745" if is_green else "#dc3545"
        source_class = f"source-{ad['source'].lower().replace(' ', '')}"
        llm_score = ad.get('llm_score')
        llm_score_text = f"{llm_score}/10" if isinstance(llm_score, (int, float)) else "Henüz yok"
        
        st.markdown(f"""
        <div class="ad-card" style="border-left-color: {badge_color};">
            <div style="display: flex; justify-content: space-between; align-items: flex-start;">
                <div style="flex: 4;">
                    <div class="source-badge {source_class}">{ad['source']}</div>
                    <h2 style="margin: 5px 0; color: #1a1a1a;">{ad['title']}</h2>
                    <div class="price-tag">
                        {ad['price']:,} TL {f'<span style="font-size: 14px; color: #666;">/ ay</span>' if ad.get('listing_type') == 'Kiralık' else ''} | {ad['area_m2']} m²
                    </div>
                    <div class="meta-info">
                        📍 {get_district_name(ad.get('district', ''))}, {ad['city']} | 🛏️ {ad.get('room_count', 'Bilinmiyor')} | 📅 {ad['days_since_posted']} gün önce | 🏷️ {ad['city']}
                    </div>
                    <div style="margin-top: 10px; display: flex; gap: 10px; font-size: 12px;">
                        <span style="background: #f0f2f6; padding: 2px 8px; border-radius: 10px;">💰 Fiyat: %{calculate_suitability_ratios(ad, min_price, max_price, target_city, district_filter)[0]:.0f}</span>
                        <span style="background: #f0f2f6; padding: 2px 8px; border-radius: 10px;">📍 Konum: {calculate_suitability_ratios(ad, min_price, max_price, target_city, district_filter)[1]}/10</span>
                        <span style="background: #f0f2f6; padding: 2px 8px; border-radius: 10px;">📐 Boyut: %{calculate_suitability_ratios(ad, min_price, max_price, target_city, district_filter, min_m2, max_m2)[2]:.0f}</span>
                        <span style="background: #f0f2f6; padding: 2px 8px; border-radius: 10px;">🧠 LLM: {llm_score_text}</span>
                    </div>
                    <p style="margin-top: 15px; font-size: 15px; line-height: 1.5; color: #444;">{ad['description'][:220]}...</p>
                </div>
                <div style="flex: 1; text-align: center; border-left: 1px solid #eee; padding-left: 20px;">
                    <div class="score-badge {'green-badge' if is_green else 'red-badge'}">%{score:.1f}</div>
                    <div style="margin-top: 10px; font-weight: bold; color: {badge_color};">
                        {"TAVSİYE EDİLEN" if score > 80 else "DEĞERLENDİRİLEBİLİR" if is_green else "DÜŞÜK UYUM"}
                    </div>
                    <div style="margin-top: 20px;">
                        <a href="{ad['url']}" target="_blank" style="text-decoration: none;">
                            <button style="width: 100%; padding: 10px; border: none; border-radius: 5px; background-color: #333; color: #fff; cursor: pointer; font-weight: bold;">
                                İlana Git ↗
                            </button>
                        </a>
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        with st.expander("Scout Mantığı: Bu Puan Nasıl Hesaplandı?"):
            st.caption("Bu skor basit bir ortalama değil; Mamdani bulanık mantık kuralları tüm girdileri birlikte değerlendirir.")

            st.markdown(f"**Fiyat yüzdesi nasıl hesaplandı?  %{ad['fuzzy_inputs']['price_suitability']:.0f}**")
            st.write(f"İlan fiyatı {ad['price']:,} TL olarak alındı.")
            st.write(f"Seçtiğin aralık {min_price:,} - {max_price:,} TL idi.")
            if min_price <= ad['price'] <= max_price:
                st.write("Fiyat aralığın içindeyse yüzde doğrudan yüksek tutuluyor.")
                st.write("Aralığın içinde ama daha düşük fiyatlı ilanlar biraz daha avantajlı sayılıyor.")
            elif ad['price'] < min_price:
                st.write("Fiyat alt sınırın altındaysa uygun kabul ediliyor ve yüzde 100'e çekiliyor.")
            else:
                st.write("Fiyat üst sınırın üstündeyse, üst limite ne kadar uzaksa yüzde o kadar düşüyor.")

            st.markdown(f"**Konum skoru nasıl hesaplandı?  {ad['fuzzy_inputs']['location_score']}/10**")
            st.write(f"İlanın şehri: {ad['city']}.")
            st.write(f"Seçtiğin şehir: {target_city}.")
            if target_district and target_district != "Hepsi":
                st.write(f"Seçtiğin ilçe: {target_district}.")
            if ad['city'] == target_city:
                st.write("Aynı şehirdeyse konum skoru yüksek başlar.")
                if target_district and target_district.lower() == get_district_name(ad.get('district', '')).lower():
                    st.write("İlçe de aynıysa skor en yüksek seviyeye çıkar.")
                else:
                    st.write("İlçe farklıysa şehir eşleşmesi korunur ama tam puan verilmez.")
            else:
                st.write("Şehir farklıysa konum skoru düşük kalır.")

            st.markdown(f"**Boyut yüzdesi nasıl hesaplandı?  %{ad['fuzzy_inputs']['size_suitability']:.0f}**")
            st.write(f"İlanın büyüklüğü: {ad.get('area_m2', 'Bilinmiyor')} m².")
            st.write(f"Seçtiğin aralık: {min_m2} - {max_m2} m².")
            if min_m2 <= ad.get('area_m2', 0) <= max_m2:
                st.write("Metrekare aralık içindeyse yüzde yüksek tutuluyor.")
                st.write("Aralık merkezine yakın ilanlar biraz daha avantajlı görünüyor.")
            else:
                st.write("Aralık dışındaysa, sınırdan uzaklaştıkça yüzde düşüyor.")

            st.markdown(f"**Diğer girdiler nasıl hesaba katıldı?  Kalite {ad['fuzzy_inputs']['listing_quality']}/10, Metin {ad['fuzzy_inputs']['llm_alignment']}/10**")
            st.write(f"Kalite skoru {ad['fuzzy_inputs']['listing_quality']}/10 olarak hesaplandı.")
            st.write(f"Metin uyumu {ad['fuzzy_inputs']['llm_alignment']}/10 olarak hesaplandı.")
            if isinstance(llm_score, (int, float)):
                st.write(f"Gerçek LLM çıktısı olarak llm_score {llm_score}/10 kaydedilmiş.")
            else:
                st.write("Bu kartta kaydedilmiş bir llm_score yok; LLM skoru üretme aşaması çalışmamış ya da veri setine yazılmamış olabilir.")
            st.write("Bu değerler de fiyat ve konum gibi kurallara giriyor ve son skoru etkiliyor.")

            st.markdown("**Nihai skor nasıl oluştu?**")
            st.write("Buradaki sonuç tek tek yüzdelerin toplanması değil.")
            st.write("Mamdani bulanık mantık kuralları tüm girdileri birlikte değerlendirip tek bir son puan üretiyor.")
            highest_priority = max(priorities, key=priorities.get)
            priority_names = {"price": "Fiyat", "location": "Konum", "size": "m²", "quality": "Kalite", "llm": "Metin"}
            st.info(f"Senin ayarlarda baskın öncelik: {priority_names[highest_priority]}")
            st.success(f"Sonuç: %{score:.1f}")
            st.progress(score/100)

st.sidebar.info(f"💡 {len(ads)} ilan arasından en uyumlu olanlar Mamdani Bulanık Mantık motoru ile seçilmiştir.")
