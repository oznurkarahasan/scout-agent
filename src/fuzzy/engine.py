"""
src/fuzzy/engine.py

Scout Agent — Mamdani Bulanık Mantık Motoru
============================================
Bu modül, normalize edilmiş bir ilan verisini alır ve
kullanıcı tercihlerine göre 0-100 arasında bir "Uygunluk Skoru" üretir.

Çalışma akışı:
  1. Fuzzification   : Ham sayısal değerleri bulanık kümelere dönüştür
  2. Rule Evaluation : IF-THEN kurallarını değerlendir
  3. Aggregation     : Tüm kural çıktılarını birleştir
  4. Defuzzification : Centroid yöntemiyle net skoru hesapla

Girdiler (her ilan için):
  - price            : ilanın kira fiyatı (TL)
  - image_count      : ilan fotoğraf sayısı
  - days_since_posted: ilanın kaç gün önce yayınlandığı
  - publisher_type   : "owner" (mal sahibi) | "agency" (emlakçı) | "unknown"
  - llm_score        : LLM'den gelen metin uyum skoru (0.0 - 1.0)

Kullanıcı parametreleri:
  - user_budget      : kullanıcının aylık bütçesi (TL)
  - user_city        : kullanıcının hedef şehri (str)
"""

import numpy as np
import unicodedata


# ══════════════════════════════════════════════════════════════════════
# BÖLÜM 1: ÜYELİK FONKSİYONLARI (Membership Functions)
# ══════════════════════════════════════════════════════════════════════
#
# Üyelik fonksiyonu: Bir değerin belirli bir bulanık kümeye ne kadar
# "ait olduğunu" 0.0 ile 1.0 arasında ölçer.
#
# Kullandığımız şekiller:
#   - trimf (üçgen): hızlı, sezgisel, kural tabanı için yeterli
#   - trapmf (yamuk): uç değerleri %100 üye kabul etmek için
#
#   trimf(x, a, b, c):
#       x=a → 0.0 | x=b → 1.0 | x=c → 0.0
#       [a, b] arasında doğrusal artış, [b, c] arasında doğrusal düşüş
#
#   trapmf(x, a, b, c, d):
#       x<a → 0.0 | [a,b] artış | [b,c] → 1.0 | [c,d] düşüş | x>d → 0.0


def trimf(x: float, a: float, b: float, c: float) -> float:
    """Üçgen üyelik fonksiyonu."""
    if x <= a or x >= c:
        return 0.0
    elif x <= b:
        return (x - a) / (b - a)
    else:
        return (c - x) / (c - b)


def trapmf(x: float, a: float, b: float, c: float, d: float) -> float:
    """Yamuk üyelik fonksiyonu."""
    if x <= a or x >= d:
        return 0.0
    elif x <= b:
        return (x - a) / (b - a)
    elif x <= c:
        return 1.0
    else:
        return (d - x) / (d - c)


def _norm_text(s: str) -> str:
    """
    Metinleri karşılaştırma için normalize eder.

    Not: Türkçe I/İ/ı/i farklılıklarını daha dayanıklı hale getirir.
    """
    s = unicodedata.normalize("NFKC", str(s)).strip().casefold()
    # Turkish-specific dotted/dotless i harmonization
    s = s.replace("ı", "i").replace("i̇", "i")
    return " ".join(s.split())


# ──────────────────────────────────────────────
# 1A. FİYAT ÜYELİK FONKSİYONLARI
# ──────────────────────────────────────────────
# Fiyat değerlendirmesi kullanıcının bütçesine göre görecelidir.
# Bütçe = 10.000 TL ise:
#   cheap  → 10.000 TL'nin altı
#   fair   → bütçeye yakın (±%20)
#   expensive → bütçenin %30 üzeri
#
# Bu sayede aynı kural tabanı her bütçe için çalışır.

def price_membership(price: float, budget: float) -> dict:
    """
    Fiyatın bütçeye göre 3 bulanık kümedeki üyelik derecelerini döndürür.

    Args:
        price  : İlanın kira fiyatı (TL)
        budget : Kullanıcının aylık bütçesi (TL)

    Returns:
        {
          "cheap"    : 0.0 - 1.0  (ucuz)
          "fair"     : 0.0 - 1.0  (uygun)
          "expensive": 0.0 - 1.0  (pahalı)
        }
    """
    if budget <= 0:
        raise ValueError("user_budget must be > 0")
    # Bütçeye göre dinamik sınırlar hesapla
    very_cheap  = budget * 0.60   # bütçenin %60'ı → kesinlikle ucuz
    cheap_peak  = budget * 0.80   # bütçenin %80'i → "ucuz" tepe noktası
    fair_low    = budget * 0.75   # "uygun" başlangıcı
    fair_peak   = budget * 1.00   # bütçeye eşit → "uygun" tepe noktası
    fair_high   = budget * 1.20   # "uygun" bitişi
    exp_start   = budget * 1.10   # "pahalı" başlangıcı
    exp_peak    = budget * 1.40   # bütçenin %140'ı → kesinlikle pahalı
    very_exp    = budget * 1.80   # çok pahalı üst sınır

    return {
        # Bütçenin %80'inden ucuzsa "cheap" — yamuk: sol taraf tamamen üye
        "cheap":     trapmf(price, 0, very_cheap, cheap_peak, budget * 0.95),

        # Bütçeye yakın fiyatlar "fair" — üçgen: bütçeye eşitse tepe
        "fair":      trimf(price, fair_low, fair_peak, fair_high),

        # Bütçenin %10 üzerindeyse "expensive" başlar — yamuk: sağ tamamen üye
        "expensive": trapmf(price, exp_start, exp_peak, very_exp, very_exp + budget),
    }


# ──────────────────────────────────────────────
# 1B. GÖRSEL KALİTE ÜYELİK FONKSİYONLARI
# ──────────────────────────────────────────────
# Fotoğraf sayısı ilan kalitesinin proxy'sidir.
# 0 fotoğraf → çok kötü, 5+ fotoğraf → çok iyi

def image_quality_membership(image_count: int) -> dict:
    """
    Fotoğraf sayısına göre ilan görsel kalitesini ölçer.

    Returns:
        {"poor": ..., "fair": ..., "good": ...}
    """
    x = float(image_count)
    return {
        # 0-1 fotoğraf: kötü kalite
        "poor": trapmf(x, -1, 0, 1, 2),

        # 2-3 fotoğraf: orta kalite
        "fair": trimf(x, 1, 2.5, 4),

        # 4+ fotoğraf: iyi kalite
        "good": trapmf(x, 3, 4, 10, 11),
    }


# ──────────────────────────────────────────────
# 1C. İLAN YENİLİĞİ ÜYELİK FONKSİYONLARI
# ──────────────────────────────────────────────
# Yeni ilanlar = daha az rekabet = daha düşük fırsat maliyeti.
# 0-3 gün: taze fırsat, 7-14 gün: normal, 14+ gün: bayat

def freshness_membership(days: int) -> dict:
    """
    İlanın yayınlanma yaşına göre "tazelik" ölçer.

    Returns:
        {"fresh": ..., "recent": ..., "old": ...}
    """
    x = float(days)
    return {
        # 0-3 gün → taze
        "fresh":  trapmf(x, -1, 0, 2, 5),

        # 4-14 gün → güncel
        "recent": trimf(x, 3, 8, 15),

        # 14+ gün → eski
        "old":    trapmf(x, 12, 18, 60, 61),
    }


# ──────────────────────────────────────────────
# 1D. LLM UYUM SKORU ÜYELİK FONKSİYONLARI
# ──────────────────────────────────────────────
# LLM, ilan metnini okuyarak kullanıcının isteklerine
# (3+1, balkon, eşyalı vb.) ne kadar uyduğunu 0.0-1.0 arası puanlar.

def llm_score_membership(score: float) -> dict:
    """
    LLM'den gelen 0.0-1.0 arası uyum skorunu bulanık kümelere dönüştürür.

    Returns:
        {"low": ..., "medium": ..., "high": ...}
    """
    return {
        "low":    trapmf(score, -0.1, 0.0, 0.3, 0.5),
        "medium": trimf(score, 0.3, 0.55, 0.75),
        "high":   trapmf(score, 0.65, 0.8, 1.0, 1.1),
    }


# ──────────────────────────────────────────────
# 1E. YAYINCI TİPİ BONUSU
# ──────────────────────────────────────────────
# Mal sahibinden kiralama → aracı yok → genellikle daha güvenilir/ucuz.
# Bu binary değeri 0.0-1.0 arası bir skora çeviriyoruz.

def publisher_score(publisher_type: str) -> float:
    """
    Yayıncı tipine göre güven skoru döndürür.
    owner → 1.0 | agency → 0.6 | unknown → 0.4

    """
    scores = {
        "owner":   1.0,   # mal sahibi — en güvenilir
        "agency":  0.6,   # emlakçı — nötr/hafif düşük
        "unknown": 0.4,   # bilinmeyen — düşük güven
    }
    return scores.get(publisher_type, 0.4)


# ══════════════════════════════════════════════════════════════════════
# BÖLÜM 2: ÇIKIŞ DEĞİŞKENİ — "score" (Uygunluk Skoru)
# ══════════════════════════════════════════════════════════════════════
#
# Defuzzification için çıkış evrenini tanımlıyoruz: 0 ile 100 arası.
# Her çıkış kümesi bu evren üzerinde bir bölgeyi temsil eder.
# Mamdani yönteminde: kural ateşlenince çıkış kümesi "kırpılır"
# (clip) ve tüm kırpılmış kümeler birleştirilerek (aggregate)
# Centroid ile tek bir sayıya indirgenir.

# Çıkış evreninin çözünürlüğü — daha fazla nokta = daha hassas sonuç
OUTPUT_UNIVERSE = np.linspace(0, 100, 500)


def output_membership(x_arr: np.ndarray) -> dict:
    """
    Çıkış değişkeni "score" için 5 bulanık küme tanımlar.
    x_arr üzerindeki her nokta için üyelik değerleri döner.

    Kümeler:
        very_low  : 0  - 20
        low       : 10 - 40
        medium    : 30 - 70
        high      : 60 - 90
        very_high : 80 - 100
    """
    def _trimf(arr, a, b, c):
        result = np.zeros_like(arr, dtype=float)
        # Yükselen kenar
        rising = (arr > a) & (arr <= b)
        result[rising] = (arr[rising] - a) / (b - a)
        # Düşen kenar
        falling = (arr > b) & (arr < c)
        result[falling] = (c - arr[falling]) / (c - b)
        # Tepe
        result[arr == b] = 1.0
        return result

    def _trapmf(arr, a, b, c, d):
        result = np.zeros_like(arr, dtype=float)
        result[(arr >= b) & (arr <= c)] = 1.0
        rising = (arr > a) & (arr < b)
        result[rising] = (arr[rising] - a) / (b - a)
        falling = (arr > c) & (arr < d)
        result[falling] = (d - arr[falling]) / (d - c)
        return result

    return {
        "very_low":  _trapmf(x_arr, -1,  0,  15, 25),
        "low":       _trimf(x_arr,  10,  30, 45),
        "medium":    _trimf(x_arr,  35,  50, 65),
        "high":      _trimf(x_arr,  55,  70, 85),
        "very_high": _trapmf(x_arr, 75,  85, 100, 101),
    }


# ══════════════════════════════════════════════════════════════════════
# BÖLÜM 3: KURAL TABANI (Rule Base)
# ══════════════════════════════════════════════════════════════════════
#
# Mamdani sisteminde her kural şu yapıdadır:
#   IF <koşullar> THEN <çıkış>
#
# Koşullar AND (min) veya OR (max) ile birleştirilebilir.
# Kural aktivasyon gücü (firing strength) → çıkış kümesini kırpar.
#
# Kural formatı:
#   (aktivasyon_gücü, çıkış_kümesi_adı)
#
# Toplam 28 kural — insan mantığını temsil eder.


def evaluate_rules(
    price_m: dict,
    quality_m: dict,
    fresh_m: dict,
    llm_m: dict,
    pub_score: float,
    city_match: bool,
) -> list[tuple[float, str]]:
    """
    Tüm IF-THEN kurallarını değerlendirir.

    Args:
        price_m    : price_membership() çıktısı
        quality_m  : image_quality_membership() çıktısı
        fresh_m    : freshness_membership() çıktısı
        llm_m      : llm_score_membership() çıktısı
        pub_score  : publisher_score() çıktısı (0.0-1.0)
        city_match : İlan şehri == kullanıcı şehri mi?

    Returns:
        [(aktivasyon_gücü, çıkış_kümesi), ...] listesi
    """

    # Kısaltmalar — okunabilirlik için
    p  = price_m       # price kümeler: cheap, fair, expensive
    q  = quality_m     # quality kümeler: poor, fair, good
    f  = fresh_m       # freshness kümeler: fresh, recent, old
    l  = llm_m         # llm kümeler: low, medium, high
    ps = pub_score     # 0.0 - 1.0
    cm = 1.0 if city_match else 0.0   # şehir eşleşmesi

    rules = []

    # ── GRUP A: Mükemmel ilanlar ──────────────────────────────────────
    # Ucuz + kaliteli + taze + LLM uyumlu = çok yüksek skor
    rules.append((
        min(p["cheap"], q["good"], f["fresh"], l["high"]),
        "very_high"
    ))

    # Uygun fiyat + iyi kalite + LLM uyumlu + şehir eşleşmesi
    rules.append((
        min(p["fair"], q["good"], l["high"], cm),
        "very_high"
    ))

    # Ucuz + orta kalite + LLM yüksek
    rules.append((
        min(p["cheap"], q["fair"], l["high"]),
        "high"
    ))

    # ── GRUP B: İyi ilanlar ───────────────────────────────────────────
    # Uygun fiyat + iyi kalite + taze
    rules.append((
        min(p["fair"], q["good"], f["fresh"]),
        "high"
    ))

    # Ucuz + iyi kalite + güncel
    rules.append((
        min(p["cheap"], q["good"], f["recent"]),
        "high"
    ))

    # Uygun fiyat + orta kalite + LLM orta + şehir eşleşmesi
    rules.append((
        min(p["fair"], q["fair"], l["medium"], cm),
        "high"
    ))

    # Mal sahibi + ucuz + taze
    rules.append((
        min(ps, p["cheap"], f["fresh"]),
        "high"
    ))

    # Ucuz + LLM yüksek + şehir eşleşmesi
    rules.append((
        min(p["cheap"], l["high"], cm),
        "high"
    ))

    # ── GRUP C: Orta ilanlar ──────────────────────────────────────────
    # Uygun fiyat + orta kalite + güncel
    rules.append((
        min(p["fair"], q["fair"], f["recent"]),
        "medium"
    ))

    # Ucuz + kötü kalite + LLM orta
    rules.append((
        min(p["cheap"], q["poor"], l["medium"]),
        "medium"
    ))

    # Pahalı + iyi kalite + LLM yüksek + taze
    rules.append((
        min(p["expensive"], q["good"], l["high"], f["fresh"]),
        "medium"
    ))

    # Uygun fiyat + taze (kalite belirsiz)
    rules.append((
        min(p["fair"], f["fresh"]),
        "medium"
    ))

    # Şehir eşleşmesi + orta kalite + güncel
    rules.append((
        min(cm, q["fair"], f["recent"]),
        "medium"
    ))

    # Mal sahibi + uygun fiyat + eski ilan
    rules.append((
        min(ps, p["fair"], f["old"]),
        "medium"
    ))

    # Ucuz + kötü görsel + eski (fiyat cazibesiyle kurtarıyor)
    rules.append((
        min(p["cheap"], q["poor"], f["old"]),
        "medium"
    ))

    # ── GRUP D: Düşük skorlu ilanlar ─────────────────────────────────
    # Pahalı + kötü kalite
    rules.append((
        min(p["expensive"], q["poor"]),
        "low"
    ))

    # Pahalı + LLM düşük
    rules.append((
        min(p["expensive"], l["low"]),
        "low"
    ))

    # Eski ilan + kötü kalite + LLM düşük
    rules.append((
        min(f["old"], q["poor"], l["low"]),
        "low"
    ))

    # Pahalı + eski + şehir eşleşmesi yok
    rules.append((
        min(p["expensive"], f["old"], 1.0 - cm),
        "low"
    ))

    # Kötü görsel + LLM düşük + eski
    rules.append((
        min(q["poor"], l["low"], f["old"]),
        "low"
    ))

    # Pahalı + güncel değil + LLM orta
    rules.append((
        min(p["expensive"], f["old"], l["medium"]),
        "low"
    ))

    # ── GRUP E: Çok düşük skorlu ilanlar ─────────────────────────────
    # Pahalı + kötü görsel + LLM düşük + eski
    rules.append((
        min(p["expensive"], q["poor"], l["low"], f["old"]),
        "very_low"
    ))

    # Şehir eşleşmesi yok + pahalı + kötü kalite
    rules.append((
        min(1.0 - cm, p["expensive"], q["poor"]),
        "very_low"
    ))

    # ── GRUP F: Özel durumlar ─────────────────────────────────────────
    # Fotoğraf yok + LLM düşük (ilan kalitesi çok kötü)
    rules.append((
        min(q["poor"], l["low"]),
        "very_low"
    ))

    # Şehir eşleşmesi var + ucuz + LLM yüksek + mal sahibi
    rules.append((
        min(cm, p["cheap"], l["high"], ps),
        "very_high"
    ))

    # Şehir yanlış ama her şey mükemmel (yine de değerlendirilebilir)
    rules.append((
        min(1.0 - cm, p["cheap"], q["good"], l["high"]),
        "high"
    ))

    # Güncel + iyi kalite + LLM orta
    rules.append((
        min(f["fresh"], q["good"], l["medium"]),
        "high"
    ))

    # Uygun fiyat + LLM düşük (kullanıcı isteğiyle örtüşmüyor)
    rules.append((
        min(p["fair"], l["low"]),
        "low"
    ))

    # ── GRUP G: LLM ağırlığı (metin uyumu) ────────────────────────────
    # LLM "high" ise, metin tercihlere çok uyumlu demektir.
    # Bu sinyalin skor üzerindeki etkisini görünür kılmak için
    # birkaç güçlü kural ekliyoruz.
    rules.append((
        min(l["high"], cm),
        "high"
    ))
    rules.append((
        min(l["high"], q["good"]),
        "high"
    ))
    rules.append((
        min(l["high"], q["good"], f["fresh"], cm),
        "very_high"
    ))

    # ── FALLBACK: Güvenlik ağı ────────────────────────────────────────
    # Daha yumuşak/ayrıştırıcı kurallar (tek kriterle de sinyal üret).
    # Not: LLM skorları henüz entegre değilken (çoğu 0.5), bu kurallar
    # 50.0 etrafında yığılmayı azaltmaya yardımcı olur.

    # Sadece fiyat sinyali
    rules.append((p["cheap"], "medium"))
    rules.append((p["fair"], "medium"))
    # Pahalı cezasını biraz yumuşat (LLM high durumunda aşırı baskı kurmasın)
    rules.append((p["expensive"] * 0.7, "low"))

    # Sadece tazelik sinyali
    rules.append((f["fresh"], "high"))
    rules.append((f["recent"], "medium"))
    rules.append((f["old"], "low"))

    # Sadece görsel kalite sinyali
    rules.append((q["good"], "high"))
    rules.append((q["fair"], "medium"))
    rules.append((q["poor"], "low"))

    # Şehir eşleşmesi tek başına küçük bir bonus
    rules.append((cm * 0.6, "medium"))

    # Yayıncı (mal sahibi) bonusu: tek başına küçük pozitif itki
    rules.append((ps * 0.4, "medium"))

    # Hiçbir kural ateşlenmese bile ilan değerlendirilebilir olsun.
    # Bu fallback sadece diğer tüm kurallar 0 ise devreye girer.
    if max((strength for strength, _ in rules), default=0.0) == 0.0:
        rules.append((0.05, "medium"))

    return rules


# ══════════════════════════════════════════════════════════════════════
# BÖLÜM 4: DEFUZZİFİKASYON (Centroid Yöntemi)
# ══════════════════════════════════════════════════════════════════════
#
# Mamdani sistemi şu adımları izler:
#   1. Her kural için aktivasyon gücü hesapla (min → AND)
#   2. Çıkış kümesini aktivasyon gücüyle kırp (clip)
#   3. Tüm kırpılmış kümeleri birleştir (max → OR)
#   4. Centroid (ağırlık merkezi) ile tek sayıya indir
#
# Centroid = Σ(x * μ(x)) / Σ(μ(x))

def defuzzify(rules: list[tuple[float, str]]) -> float:
    """
    Mamdani Centroid defuzzification.

    Args:
        rules: [(aktivasyon_gücü, çıkış_kümesi_adı), ...]

    Returns:
        0.0 - 100.0 arası net uygunluk skoru
    """
    x = OUTPUT_UNIVERSE
    out_mf = output_membership(x)

    # Birleşik çıkış eğrisi (başlangıçta sıfır)
    aggregated = np.zeros_like(x)

    for strength, label in rules:
        if strength <= 0:
            continue   # ateşlenmeyen kuralı atla

        # Çıkış kümesini aktivasyon gücüyle kırp (clip)
        clipped = np.minimum(strength, out_mf[label])

        # Tüm kırpılmış kümeleri birleştir (max/union)
        aggregated = np.maximum(aggregated, clipped)

    # Centroid hesapla
    total_area = np.sum(aggregated)
    if total_area == 0:
        return 0.0   # hiçbir kural ateşlenmediyse 0 döndür

    centroid = np.sum(x * aggregated) / total_area
    return round(float(centroid), 2)


# ══════════════════════════════════════════════════════════════════════
# BÖLÜM 5: ANA MOTOR FONKSİYONU
# ══════════════════════════════════════════════════════════════════════

def score_ad(
    ad: dict,
    user_budget: float,
    user_city: str,
    llm_score: float = 0.5,   # Faz 3'te gerçek LLM skoru gelecek
) -> dict:
    """
    Bir ilanı değerlendirerek uygunluk skoru üretir.

    Args:
        ad          : normalize_ad() çıktısı (dict)
        user_budget : Kullanıcının aylık bütçesi (TL)
        user_city   : Kullanıcının hedef şehri (str)
        llm_score   : LLM uyum skoru (0.0-1.0); Faz 3'e kadar mock

    Returns:
        {
          "id"           : str
          "title"        : str
          "score"        : float  ← Ana çıktı (0-100)
          "breakdown"    : dict   ← Her girdinin bulanık değerleri
        }
    """

    if not isinstance(ad, dict):
        raise TypeError("ad must be a dict")
    required_keys = ("id", "title", "price", "image_count", "days_since_posted", "publisher_type", "city")
    missing = [k for k in required_keys if k not in ad]
    if missing:
        raise KeyError(f"ad is missing required keys: {missing}")

    # ── Adım 1: Fuzzification ─────────────────────────────────────────
    price_m   = price_membership(ad["price"], user_budget)
    quality_m = image_quality_membership(ad["image_count"])
    fresh_m   = freshness_membership(ad["days_since_posted"])
    llm_m     = llm_score_membership(llm_score)
    pub_score = publisher_score(ad["publisher_type"])
    city_match = _norm_text(ad["city"]) == _norm_text(user_city)

    # ── Adım 2: Kural değerlendirmesi ─────────────────────────────────
    rules = evaluate_rules(price_m, quality_m, fresh_m, llm_m, pub_score, city_match)

    # ── Adım 3 & 4: Aggregation + Defuzzification ────────────────────
    final_score = defuzzify(rules)

    # ── Adım 5: Sonucu paketle ───────────────────────────────────────
    return {
        "id":    ad["id"],
        "title": ad["title"],
        "score": final_score,
        "breakdown": {
            "price_memberships":   price_m,
            "quality_memberships": quality_m,
            "fresh_memberships":   fresh_m,
            "llm_memberships":     llm_m,
            "publisher_score":     pub_score,
            "city_match":          city_match,
            "llm_score_input":     llm_score,
        },
    }


def score_all_ads(
    ads: list[dict],
    user_budget: float,
    user_city: str,
    llm_scores: dict[str, float] | None = None,
    filter_city: bool = False,
) -> list[dict]:
    """
    Tüm ilan listesini skorlar ve yüksekten düşüğe sıralar.

    Args:
        ads         : normalize edilmiş ilan listesi
        user_budget : kullanıcı bütçesi
        user_city   : hedef şehir
        llm_scores  : {ilan_id: llm_skoru} — Faz 3'te doldurulacak
        filter_city : True ise sadece user_city ile eşleşen ilanları skorla

    Returns:
        Sıralanmış sonuç listesi
    """
    results = []
    for ad in ads:
        if filter_city and (_norm_text(ad.get("city", "")) != _norm_text(user_city)):
            continue
        # LLM skoru varsa kullan, yoksa mock değer (0.5)
        llm = (llm_scores or {}).get(ad["id"], 0.5)
        result = score_ad(ad, user_budget, user_city, llm_score=llm)
        results.append(result)

    # Skora göre büyükten küçüğe sırala
    results.sort(key=lambda r: r["score"], reverse=True)
    return results


# ══════════════════════════════════════════════════════════════════════
# DOĞRUDAN ÇALIŞTIRMA — Hızlı test --- python3 src/fuzzy/engine.py
# USE_LLM=1 REQUIRE_GROQ=1 USER_PREFS="balkon, eşyalı, metro" python3 src/fuzzy/engine.py
# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import json, os, sys

    # Bu dosya doğrudan çalıştırıldığında `src.*` importları için repo root'u ekle.
    _ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if _ROOT not in sys.path:
        sys.path.insert(0, _ROOT)

    # .env dosyasından GROQ_API_KEY / USE_LLM / USER_PREFS al
    try:
        from dotenv import load_dotenv  # type: ignore

        load_dotenv()
    except Exception:
        pass

    from src.llm import UserPreferences, score_ads_with_llm

    # Normalize veriyi yükle
    data_path = os.path.join(
        os.path.dirname(__file__), "..", "..", "data", "normalized_ads.json"
    )

    with open(data_path, "r", encoding="utf-8") as f:
        ads = json.load(f)
    if not isinstance(ads, list) or len(ads) == 0:
        raise ValueError("normalized_ads.json must be a non-empty list")

    # Kullanıcı tercihleri
    BUDGET    = 10_000   # TL
    CITY      = "İstanbul"
    # LLM entegrasyonu için demo tercih metni (istersen main/app tarafına taşırız)
    USER_PREFS = os.getenv("USER_PREFS", "balkon, eşyalı, ulaşım, metro, yeni")
    USE_LLM = (os.getenv("USE_LLM", "0").strip() == "1")
    REQUIRE_GROQ = (os.getenv("REQUIRE_GROQ", "0").strip() == "1")

    print(f"\n{'='*55}")
    print(f"  Scout Agent — Bulanık Mantık Motoru")
    print(f"  Bütçe: {BUDGET:,} TL  |  Şehir: {CITY}")
    print(f"  LLM: {'AÇIK' if USE_LLM else 'KAPALI'}")
    if USE_LLM:
        print(f"  Groq: {'ZORUNLU' if REQUIRE_GROQ else 'Opsiyonel (fallback var)'}")
    print(f"{'='*55}\n")

    if USE_LLM and REQUIRE_GROQ:
        proxy_vars = ["HTTPS_PROXY", "HTTP_PROXY", "ALL_PROXY", "https_proxy", "http_proxy", "all_proxy"]
        active = {k: os.getenv(k) for k in proxy_vars if os.getenv(k)}
        if active:
            print("[LLM][WARN] Proxy ortam değişkenleri tespit edildi; Groq isteği proxy üzerinden engellenebilir.")
            for k, v in active.items():
                print(f"  - {k}={v}")
            print("")

    # Önce şehir filtresi uygula (performans için)
    filtered_ads = [a for a in ads if _norm_text(a.get("city", "")) == _norm_text(CITY)]

    llm_scores = None
    if USE_LLM:
        prefs = UserPreferences(free_text=USER_PREFS)
        llm_scores = score_ads_with_llm(filtered_ads, prefs, use_remote_llm=True if REQUIRE_GROQ else None)
        print(f"[LLM] {len(llm_scores)} ilan için llm_score üretildi. Örn: {list(llm_scores.items())[:2]}\n")

    results = score_all_ads(
        filtered_ads,
        user_budget=BUDGET,
        user_city=CITY,
        llm_scores=llm_scores,
        filter_city=False,  # zaten yukarıda filtreledik
    )

    print(f"{'Sıra':<5} {'Puan':>6}  {'İlan Başlığı':<45} {'Şehir'}")
    print("-" * 75)
    for i, r in enumerate(results, 1):
        # Breakdown'dan şehri almak için ads'e bak
        ad = next(a for a in ads if a["id"] == r["id"])
        city_tag = "✓" if r["breakdown"]["city_match"] else "✗"
        print(f"  {i:<4} {r['score']:>5.1f}  {r['title'][:44]:<45} {city_tag} {ad['city']}")

    print(f"\n{'='*55}")
    print("  İlk ilanın detaylı breakdown'u:")
    print(f"{'='*55}")
    if not results:
        raise ValueError("No results produced from ads list")
    top = results[0]
    b = top["breakdown"]
    print(f"  Fiyat kümeleri : {top['breakdown']['price_memberships']}")
    print(f"  Kalite kümeleri: {top['breakdown']['quality_memberships']}")
    print(f"  Tazelik kümeleri:{top['breakdown']['fresh_memberships']}")
    print(f"  LLM kümeleri   : {top['breakdown']['llm_memberships']}")
    print(f"  Yayıncı skoru  : {b['publisher_score']}")
    print(f"  Şehir eşleşmesi: {b['city_match']}")
    print(f"  → FİNAL SKOR   : {top['score']}")