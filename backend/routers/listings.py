import json
import os
import re
from typing import Optional
from fastapi import APIRouter, Query

from src.fuzzy.engine import ScoutFuzzyEngine

router = APIRouter()

DATA_PATH = os.path.join(os.path.dirname(__file__), "../../data/normalized_ads.json")

_engine = ScoutFuzzyEngine()
_ads: list[dict] = []


def _load_ads() -> list[dict]:
    global _ads
    if not _ads and os.path.exists(DATA_PATH):
        with open(DATA_PATH, "r", encoding="utf-8") as f:
            _ads = json.load(f)
    return _ads


def _get_district_name(raw_district: str) -> str:
    district = (raw_district or "").strip()
    if " - " in district:
        district = district.split(" - ", 1)[0].strip()
    district = re.split(r"\s{2,}", district)[0].strip()
    return district


def _calculate_ad_score(
    ad: dict,
    min_p: int,
    max_p: int,
    target_district: str,
    target_rooms: list[str],
    min_m2: int,
    max_m2: int,
    priorities: dict,
) -> tuple[float, dict]:
    # Price suitability (0-100) — app.py'den aynen alındı
    if min_p <= ad["price"] <= max_p:
        p_suit = 100 - ((ad["price"] - min_p) / (max_p - min_p + 1)) * 30
    elif ad["price"] < min_p:
        p_suit = 100
    else:
        p_suit = max(0, 70 - ((ad["price"] - max_p) / max_p) * 200)

    # Location score (0-10): yakin=10, nötr=6, uzak=3
    ad_district = _get_district_name(ad.get("district", "")).lower()
    if not target_district:
        l_score = 6  # ilçe filtresi yok: nötr
    elif target_district.lower() == ad_district:
        l_score = 10  # tam eşleşme: yakın
    else:
        l_score = 3   # farklı ilçe: uzak

    # Size suitability (0-100): merkez=100, kenar=0
    ad_m2 = ad.get("area_m2", 100)
    if min_m2 <= ad_m2 <= max_m2:
        center = (min_m2 + max_m2) / 2
        half_range = max((max_m2 - min_m2) / 2, 1)
        s_suit = max(0, 100 - (abs(ad_m2 - center) / half_range) * 100)
    else:
        s_suit = 0

    # Quality score (0-10)
    q_score = min(10, ad["image_count"] * 2 + (1 if len(ad["description"]) > 150 else 0))

    # Room match (0-10)
    m_score = 5
    if "Hepsi" not in target_rooms:
        m_score = 10 if ad.get("room_count") in target_rooms else 0

    inputs = {
        "price_suitability": p_suit,
        "location_score": l_score,
        "listing_quality": q_score,
        "size_suitability": s_suit,
        "llm_alignment": m_score,
    }

    _engine.prepare(priorities)
    score, _ = _engine.compute(inputs)
    return score, inputs


@router.get("/cities")
def get_cities():
    ads = _load_ads()
    cities = sorted({ad.get("city", "").strip() for ad in ads if ad.get("city")})
    return {"cities": cities}


@router.get("/districts")
def get_districts(city: str = Query(...)):
    ads = _load_ads()
    districts = sorted({
        _get_district_name(ad.get("district", ""))
        for ad in ads
        if ad.get("city") == city and ad.get("district")
    })
    return {"districts": districts}


@router.get("/listings")
def get_listings(
    city: str = Query(...),
    district: Optional[str] = Query(default=""),
    listing_type: str = Query(default="Hepsi"),
    min_price: int = Query(default=10000),
    max_price: int = Query(default=30000),
    min_m2: int = Query(default=75),
    max_m2: int = Query(default=200),
    rooms: str = Query(default="Hepsi"),
    priority_price: float = Query(default=0.9),
    priority_location: float = Query(default=0.7),
    priority_size: float = Query(default=0.6),
    priority_quality: float = Query(default=0.5),
    priority_llm: float = Query(default=0.4),
):
    ads = _load_ads()
    target_rooms = rooms.split(",") if rooms != "Hepsi" else ["Hepsi"]
    district_filter = district.strip() if district else ""

    priorities = {
        "price": priority_price,
        "location": priority_location,
        "size": priority_size,
        "quality": priority_quality,
        "llm": priority_llm,
    }

    scored = []
    for ad in ads:
        if ad.get("city") != city:
            continue
        if listing_type != "Hepsi" and ad.get("listing_type") != listing_type:
            continue
        if "Hepsi" not in target_rooms and ad.get("room_count") not in target_rooms:
            continue
        if district_filter and district_filter.lower() != _get_district_name(ad.get("district", "")).lower():
            continue
        price = ad.get("price", 0)
        if price == 0 or price < min_price or price > max_price:
            continue
        ad_m2 = ad.get("area_m2", 100)
        if ad_m2 < min_m2 or ad_m2 > max_m2:
            continue

        score, fuzzy_inputs = _calculate_ad_score(
            ad, min_price, max_price, district_filter,
            target_rooms, min_m2, max_m2, priorities,
        )

        scored.append({
            **ad,
            "scout_score": round(score, 2),
            "fuzzy_inputs": fuzzy_inputs,
        })

    scored.sort(key=lambda x: x["scout_score"], reverse=True)

    return {"count": len(scored), "listings": scored[:20]}
