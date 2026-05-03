import streamlit as st
import json
import re
import sys
import skfuzzy as fuzz
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

engine = st.session_state.engine

# 2. Sidebar - Advanced Search Criteria
st.sidebar.title("Arama Filtreleri")

if st.sidebar.button("Veri Setini İşle", use_container_width=True):
    with st.spinner("Yerel veri seti işleniyor..."):
        import subprocess
        result = subprocess.run([sys.executable, "src/data/processor.py"], capture_output=True, text=True)
        if result.returncode != 0:
            st.error(f"Hata:\n{result.stderr}")
        else:
            st.cache_data.clear()
            st.success("Veri seti güncellendi!")
           # st.rerun()

st.sidebar.subheader("Emlak Bilgileri")
target_listing_type = st.sidebar.radio("İlan Tipi", ["Hepsi", "Kiralık", "Satılık"], horizontal=True)

st.sidebar.subheader("Fiyat Aralığı (TL)")
# Slider aralıkları veri setindeki gerçek min/max'a göre ayarlandı.
# Kiralık: 5k-25k TL (medyan ~9.8k), Satılık: 2.8M-11.2M TL (medyan ~5.8M)
if target_listing_type == "Satılık":
    min_price, max_price = st.sidebar.slider("Bütçe Seçimi", 2_800_000, 11_500_000, (3_000_000, 8_000_000), step=100_000)
elif target_listing_type == "Kiralık":
    min_price, max_price = st.sidebar.slider("Bütçe Seçimi", 5_000, 25_000, (7_000, 15_000), step=500)
else:  # Hepsi — kiralık ve satılık birlikte gösteriliyorsa kiralık aralığı baz alınır
    min_price, max_price = st.sidebar.slider("Bütçe Seçimi", 5_000, 25_000, (7_000, 15_000), step=500)

st.sidebar.subheader("Büyüklük (m²)")
min_m2, max_m2 = st.sidebar.slider("Metrekare Aralığı", 0, 1000, (75, 200), step=5)

target_city = st.sidebar.selectbox("Şehir", ["İstanbul", "Ankara", "İzmir", "Bursa", "Antalya"])
target_district = st.sidebar.text_input("İlçe Ara", "")

st.sidebar.subheader("Oda Sayısı")
room_options = ["Hepsi", "1+1", "2+1", "3+1", "4+1"]
target_rooms = st.sidebar.multiselect("Tercih Edilen Oda Sayısı", room_options, default=["Hepsi"])

st.sidebar.divider()
st.sidebar.title("Senin Önceliklerin")
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
@st.cache_data
def load_ads():
    with open('data/normalized_ads.json', 'r', encoding='utf-8') as f:
        return json.load(f)

ads = load_ads()

# Helper for UI display
def calculate_suitability_ratios(ad, min_p, max_p, target_city, target_district, min_m=0, max_m=1000):
    # Price
    if min_p <= ad['price'] <= max_p: p_suit = 100
    elif ad['price'] < min_p: p_suit = max(0, 100 - ((min_p - ad['price']) / min_p) * 150)
    else: p_suit = max(0, 100 - ((ad['price'] - max_p) / max_p) * 200)
    
    # Location
    l_score = 0
    if ad['city'] == target_city:
        l_score = 7
        if target_district and target_district.lower() in ad['district'].lower(): l_score = 10
    else: l_score = 2
    
    # Size
    ad_m2 = ad.get('area_m2', 100)
    if min_m <= ad_m2 <= max_m: s_suit = 100
    else:
        dist = min(abs(ad_m2 - min_m), abs(ad_m2 - max_m))
        s_suit = max(0, 100 - (dist / max(1, min_m)) * 100)
        
    return p_suit, l_score, s_suit

def calculate_ad_score(ad, min_p, max_p, target_city, target_district, target_rooms, engine):
    # a. Price Suitability
    # Aralık içi: tam uyum. Aralık dışı (her iki yönde) yumuşak ceza.
    if min_p <= ad['price'] <= max_p:
        p_suit = 100
    elif ad['price'] < min_p:
        # Çok ucuz = aralığın altında; minimum'dan uzaklaştıkça ceza artar.
        # Bu sayede 5k ile 9k aynı skoru almaz (ikisi de "ucuz" değil).
        p_suit = max(0, 100 - ((min_p - ad['price']) / min_p) * 150)
    else:
        # Maksimum üstü: sert ceza
        p_suit = max(0, 100 - ((ad['price'] - max_p) / max_p) * 200)

    # b. Location Score (0-10)
    l_score = 0
    if ad['city'] == target_city:
        l_score = 7
        if target_district and target_district.lower() in ad['district'].lower():
            l_score = 10
    else:
        l_score = 2

    # c. Size Suitability (0-100)
    ad_m2 = ad.get('area_m2', 100)
    if min_m2 <= ad_m2 <= max_m2:
        s_suit = 100
    else:
        dist = min(abs(ad_m2 - min_m2), abs(ad_m2 - max_m2))
        s_suit = max(0, 100 - (dist / max(1, min_m2)) * 100)

    # d. Quality Score (0-10)
    q_score = min(10, ad['image_count'] * 2 + (1 if len(ad['description']) > 150 else 0))

    # e. LLM + Room Match
    llm_score = ad.get('llm_score', 5)
    room_bonus = 0
    if "Hepsi" not in target_rooms:
        found_rooms = re.findall(r'\d\+\d', ad['title'])
        if found_rooms and found_rooms[0] in target_rooms:
            room_bonus = 3
        elif found_rooms:
            room_bonus = -3
    m_score = min(10, max(0, llm_score + room_bonus))

    # Fuzzy Compute
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
# PREPARE ENGINE ONCE (This is the performance fix!)
engine.prepare(priorities)

scored_ads = []
for ad in ads:
    # --- Hard Filters ---
    # 1. Listing Type (Kiralık/Satılık)
    if target_listing_type != "Hepsi":
        if ad.get('listing_type') != target_listing_type:
            continue
            
    # 2. Room Count (Strict Filter)
    if "Hepsi" not in target_rooms:
        found_rooms = re.findall(r'\d\+\d', ad['title'])
        if not found_rooms or found_rooms[0] not in target_rooms:
            continue
    
    # --- Scoring ---
    if ad.get('price', 0) == 0:
        continue
        
    score, sim, fuzzy_inputs = calculate_ad_score(ad, min_price, max_price, target_city, target_district, target_rooms, engine)
    ad_copy = ad.copy()
    ad_copy['scout_score'] = score
    ad_copy['fuzzy_sim'] = sim
    ad_copy['fuzzy_inputs'] = fuzzy_inputs
    scored_ads.append(ad_copy)

# Sort Descending
scored_ads.sort(key=lambda x: x['scout_score'], reverse=True)

# 6. UI Rendering
st.title("Scout Agent: Gayrimenkul Asistanı")
st.write(f"**{target_city}** bölgesinde **{min_price:,} - {max_price:,} TL** aralığında en iyi ilanlar taranıyor...")

for ad in scored_ads[:20]: # Show top 20
    score = ad['scout_score']
    is_green = score >= 50
    badge_color = "#28a745" if is_green else "#dc3545"
    source_class = f"source-{ad['source'].lower().replace(' ', '')}"
    
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
                    {ad['district']}, {ad['city']} | {ad['days_since_posted']} gün önce
                </div>
                <div style="margin-top: 10px; display: flex; gap: 10px; font-size: 12px;">
                    <span style="background: #f0f2f6; padding: 2px 8px; border-radius: 10px;">Fiyat: %{calculate_suitability_ratios(ad, min_price, max_price, target_city, target_district)[0]:.0f}</span>
                    <span style="background: #f0f2f6; padding: 2px 8px; border-radius: 10px;">Konum: {calculate_suitability_ratios(ad, min_price, max_price, target_city, target_district)[1]}/10</span>
                    <span style="background: #f0f2f6; padding: 2px 8px; border-radius: 10px;">Boyut: %{calculate_suitability_ratios(ad, min_price, max_price, target_city, target_district, min_m2, max_m2)[2]:.0f}</span>
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
                            İlana Git
                        </button>
                    </a>
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("Scout Mantığı: Bu Puan Nasıl Hesaplandı?"):
        raw = ad['fuzzy_inputs']

        # --- Yardımcı yapılar ---
        _universes  = {'price_suitability': 100, 'location_score': 10,
                       'listing_quality': 10, 'size_suitability': 100, 'llm_alignment': 10}
        _prio_key   = {'price_suitability': 'price', 'location_score': 'location',
                       'listing_quality': 'quality', 'size_suitability': 'size', 'llm_alignment': 'llm'}
        _labels     = {'price_suitability': 'Fiyat', 'location_score': 'Konum',
                       'listing_quality': 'Kalite', 'size_suitability': 'Boyut', 'llm_alignment': 'Metin (LLM)'}
        _mf_objs    = {'price_suitability': (engine.price,    ['pahali','makul','ucuz']),
                       'location_score':    (engine.location, ['uzak','orta','yakin']),
                       'listing_quality':   (engine.quality,  ['zayif','iyi','mukemmel']),
                       'size_suitability':  (engine.size,     ['kucuk','ideal','buyuk']),
                       'llm_alignment':     (engine.llm_match,['uyumsuz','kismi','uyumlu'])}

        # --- Adım 1: Öncelik Ölçekleme ---
        st.markdown("**01 Öncelik Ölçekleme** — önceliği düşük kriter nötre çekilir, yüksek kriter aynen girer")
        cols = st.columns(5)
        scaled_vals = {}
        for i, key in enumerate(raw):
            umax    = _universes[key]
            neutral = umax / 2.0
            p       = priorities[_prio_key[key]]
            s       = neutral + (raw[key] - neutral) * (p ** 1.5)
            scaled_vals[key] = s
            delta   = s - raw[key]
            with cols[i]:
                st.metric(
                    label=f"{_labels[key]} (ö={p:.1f})",
                    value=f"{s:.1f}",
                    delta=f"{delta:+.1f}",
                    delta_color="normal"
                )

        st.divider()

        # --- Adım 2: Bulanıklaştırma ---
        st.markdown("**02 Bulanıklaştırma** — ölçekli değer hangi fuzzy kümeye ne kadar ait?")
        cols2 = st.columns(5)
        for i, key in enumerate(raw):
            antecedent, mf_names = _mf_objs[key]
            sv = scaled_vals[key]
            with cols2[i]:
                st.caption(_labels[key])
                for name in mf_names:
                    act = fuzz.interp_membership(antecedent.universe, antecedent[name].mf, sv)
                    if act > 0.01:
                        st.progress(float(act), text=f"{name}: {act:.2f}")

        st.divider()

        # --- Adım 3: Sonuç Kategorisi ---
        st.markdown("**03 Mamdani Durulama -> Nihai Skor**")
        c1, c2 = st.columns([1, 2])
        with c1:
            st.metric("Uygunluk Skoru", f"%{score:.1f}")
        with c2:
            categories = [('cop','Çöp',(0,25)),('dusuk','Düşük',(15,55)),
                          ('orta','Orta',(45,75)),('yuksek','Yüksek',(65,90)),
                          ('efsane','Efsane',(83,100))]
            dominant = max(
                categories,
                key=lambda c: fuzz.interp_membership(
                    engine.score.universe, engine.score[c[0]].mf, score)
            )
            st.info(f"Skor **{dominant[1]}** kategorisine giriyor "
                    f"(centroid defuzzification, aralık {dominant[2][0]}–{dominant[2][1]})")

st.sidebar.info(f"{len(ads)} ilan arasından en uyumlu olanlar Mamdani Bulanık Mantık motoru ile seçilmiştir.")
