"""
src/data/loader.py

Ham ilan verisini (ads.json) normalize ederek işleme katmanına
hazır hale getirir ve normalized_ads.json olarak yazar.

Yapılan dönüşümler:
  - price_raw  → price (int, TL/ay temizle)
  - location_raw → city + district (split)
  - posted_date → days_since_posted (bugünden fark)
  - Eksik alan validation
"""

import json
import re
import os
from datetime import date, datetime


# ──────────────────────────────────────────────
# Yardımcı dönüştürücüler
# ──────────────────────────────────────────────

def parse_price(price_raw: str) -> int | None:
    """
    '12.500 TL/ay', '9.000TL/ay', '6.750 TL / ay' → 12500, 9000, 6750
    Başarısız olursa None döner.
    """
    if not price_raw:
        return None
    # Nokta binlik ayracı kaldır, harf+slash+boşluk temizle
    cleaned = re.sub(r"[^\d]", "", price_raw.replace(".", ""))
    return int(cleaned) if cleaned else None


def parse_location(location_raw: str) -> tuple[str, str]:
    """
    'Kadıköy, İstanbul' → ('İstanbul', 'Kadıköy')
    'Beşiktaş,İstanbul' → ('İstanbul', 'Beşiktaş')  ← boşluksuz format
    Eğer virgül yoksa city=location_raw, district='Bilinmiyor' döner.
    """
    if not location_raw:
        return "Bilinmiyor", "Bilinmiyor"

    parts = [p.strip() for p in location_raw.split(",")]
    if len(parts) >= 2:
        district = parts[0]
        city = parts[1]
    else:
        district = "Bilinmiyor"
        city = parts[0]

    return city, district


def parse_days_since(posted_date_str: str) -> int | None:
    """
    '2025-04-20' → bugünden kaç gün önce (int)
    """
    if not posted_date_str:
        return None
    try:
        posted = datetime.strptime(posted_date_str, "%Y-%m-%d").date()
        delta = date.today() - posted
        return max(delta.days, 0)   # negatif olmasın
    except ValueError:
        return None


# ──────────────────────────────────────────────
# Validasyon
# ──────────────────────────────────────────────

REQUIRED_RAW_FIELDS = [
    "id", "title", "price_raw", "location_raw",
    "description", "image_count", "posted_date",
    "publisher_type"
]

def validate_raw(ad: dict) -> list[str]:
    """Eksik zorunlu alanları listele. Boş liste → geçerli."""
    return [f for f in REQUIRED_RAW_FIELDS if f not in ad or ad[f] is None]


# ──────────────────────────────────────────────
# Ana dönüştürücü
# ──────────────────────────────────────────────

def normalize_ad(raw: dict) -> dict | None:
    """
    Tek bir ham ilanı normalize et.
    Kritik alan eksikse None döner (ilan atlanır).
    """
    errors = validate_raw(raw)
    if errors:
        print(f"  [SKIP] {raw.get('id', '?')} — eksik alanlar: {errors}")
        return None

    price = parse_price(raw["price_raw"])
    if price is None:
        print(f"  [SKIP] {raw['id']} — fiyat parse edilemedi: {raw['price_raw']!r}")
        return None

    city, district = parse_location(raw["location_raw"])
    days = parse_days_since(raw["posted_date"])
    if days is None:
        print(f"  [WARN] {raw['id']} — tarih parse edilemedi, days_since_posted=0 atandı")
        days = 0

    return {
        "id":               raw["id"],
        "title":            raw["title"].strip(),
        "price":            price,
        "city":             city,
        "district":         district,
        "description":      raw["description"].strip(),
        "image_count":      int(raw["image_count"]),
        "days_since_posted": days,
        "publisher_type":   raw["publisher_type"],   # 'owner' | 'agent'
        "is_featured":      bool(raw.get("is_featured", False)),
        "source":           raw.get("source", "unknown"),
        "views":            int(raw.get("views", 0)),
    }


# ──────────────────────────────────────────────
# Pipeline: dosya oku → normalize et → kaydet
# ──────────────────────────────────────────────

def load_and_normalize(
    raw_path: str = "data/ads.json",
    output_path: str = "data/normalized_ads.json",
    save: bool = True,
) -> list[dict]:
    """
    raw_path      : ham ilan JSON dosyası
    output_path   : normalize edilmiş çıktı dosyası
    save          : True ise diske yazar
    Dönüş         : normalize edilmiş ilan listesi
    """
    # 1. Ham veriyi oku
    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"Ham veri bulunamadı: {raw_path}")

    with open(raw_path, "r", encoding="utf-8") as f:
        raw_ads: list[dict] = json.load(f)

    print(f"[LOADER] {len(raw_ads)} ham ilan okundu → {raw_path}")

    # 2. Normalize et
    normalized = []
    for raw in raw_ads:
        result = normalize_ad(raw)
        if result:
            normalized.append(result)

    skipped = len(raw_ads) - len(normalized)
    print(f"[LOADER] {len(normalized)} ilan normalize edildi, {skipped} ilan atlandı.")

    # 3. Diske yaz
    if save:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(normalized, f, ensure_ascii=False, indent=2)
        print(f"[LOADER] Normalize edilmiş veri kaydedildi → {output_path}")

    return normalized


# ──────────────────────────────────────────────
# Doğrudan çalıştırma (test için)
# ──────────────────────────────────────────────

if __name__ == "__main__":
    ads = load_and_normalize(
        raw_path="data/ads.json",
        output_path="data/normalized_ads.json",
    )

    print("\n── İlk 3 normalize ilan (önizleme) ──")
    for ad in ads[:3]:
        print(json.dumps(ad, ensure_ascii=False, indent=2))