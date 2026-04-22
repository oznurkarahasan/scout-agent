"""
main.py — Scout Agent Interactive CLI
"""

from __future__ import annotations

import json
import os
import sys

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


def ask(prompt: str, default: str = "") -> str:
    hint = f" [{default}]" if default else ""
    try:
        val = input(f"  {prompt}{hint}: ").strip()
    except (KeyboardInterrupt, EOFError):
        print("\nÇıkılıyor...")
        sys.exit(0)
    return val if val else default


def get_user_input(cities: list[str]) -> dict:
    print(f"\n{'═'*50}")
    print("  Scout Agent — Emlak Asistanı")
    print(f"{'═'*50}\n")

    # Şehir seçimi
    print("  Mevcut şehirler: " + ", ".join(sorted(set(cities))))
    city = ask("Hedef şehir", "İstanbul")

    # Bütçe
    while True:
        raw = ask("Aylık bütçe (TL)", "10000")
        try:
            budget = float(raw.replace(".", "").replace(",", ""))
            break
        except ValueError:
            print("  [!] Geçerli bir sayı girin.")

    # Tercihler
    prefs = ask("Tercihleriniz (virgülle ayırın)", "balkon, eşyalı, metro")

    # Kaç ilan
    while True:
        raw = ask("Kaç ilan listelensin", "10")
        try:
            top_n = int(raw)
            break
        except ValueError:
            print("  [!] Geçerli bir sayı girin.")

    # Sadece seçilen şehir mi?
    only_city = ask("Sadece bu şehirdeki ilanlar mı? (e/h)", "e").lower() in ("e", "evet", "y", "yes")

    # LLM
    use_llm = ask("LLM analizi aktif olsun mu? (e/h)", "e").lower() in ("e", "evet", "y", "yes")

    return dict(city=city, budget=budget, prefs=prefs, top_n=top_n, only_city=only_city, use_llm=use_llm)


def run(city: str, budget: float, prefs: str, top_n: int, only_city: bool, use_llm: bool) -> list[dict]:
    print(f"\n{'─'*50}")

    # 1. Veri
    print("  [1/3] Veri yükleniyor...")
    ads = load_and_normalize(RAW_DATA_PATH, NORMALIZED_DATA_PATH, save=True)

    target = [a for a in ads if _norm_text(a.get("city", "")) == _norm_text(city)] if only_city else ads
    if not target:
        print(f"  [!] '{city}' için ilan bulunamadı, tüm ilanlar değerlendiriliyor.")
        target = ads
    print(f"       {len(target)} ilan değerlendirilecek.")

    # 2. LLM
    llm_scores = None
    if use_llm:
        print("  [2/3] LLM analizi yapılıyor...")
        llm_scores = score_ads_with_llm(target, UserPreferences(free_text=prefs), use_remote_llm=None)
        groq_on = bool((os.getenv("GROQ_API_KEY") or "").strip())
        print(f"       Mod: {'Groq API' if groq_on else 'Heuristic'}")
    else:
        print("  [2/3] LLM analizi atlandı.")

    # 3. Fuzzy engine
    print("  [3/3] Skorlar hesaplanıyor...")
    results = score_all_ads(target, user_budget=budget, user_city=city, llm_scores=llm_scores, filter_city=False)

    return results[:top_n]


def print_results(results: list[dict], ads_map: dict, budget: float, city: str) -> None:
    print(f"\n{'═'*65}")
    print(f"  Sonuçlar  |  Bütçe: {budget:,.0f} TL  |  Şehir: {city}")
    print(f"{'═'*65}")
    print(f"  {'#':<4} {'Skor':>6}  {'Başlık':<38} {'Şehir':<14} {'Fiyat':>10}")
    print(f"  {'─'*4} {'─'*6}  {'─'*38} {'─'*14} {'─'*10}")

    for i, r in enumerate(results, 1):
        ad       = ads_map.get(r["id"], {})
        match    = "✓" if r["breakdown"]["city_match"] else "✗"
        ad_city  = f"{match} {ad.get('city', '?')}"
        price    = f"{ad.get('price', 0):,} TL"
        title    = r["title"][:37]
        print(f"  {i:<4} {r['score']:>5.1f}  {title:<38} {ad_city:<14} {price:>10}")

    print(f"\n  {len(results)} ilan listelendi.")

    # Detay göster?
    try:
        choice = input("\n  Detay görmek istediğiniz ilan numarası (Enter=atla): ").strip()
    except (KeyboardInterrupt, EOFError):
        return

    if choice.isdigit():
        idx = int(choice) - 1
        if 0 <= idx < len(results):
            print_breakdown(results[idx])


def print_breakdown(r: dict) -> None:
    b = r["breakdown"]
    print(f"\n  {'─'*50}")
    print(f"  {r['title']}")
    print(f"  {'─'*50}")

    def fmt(d: dict) -> str:
        return "  ".join(f"{k}={v:.2f}" for k, v in d.items())

    print(f"  Fiyat    : {fmt(b['price_memberships'])}")
    print(f"  Kalite   : {fmt(b['quality_memberships'])}")
    print(f"  Tazelik  : {fmt(b['fresh_memberships'])}")
    print(f"  LLM      : {fmt(b['llm_memberships'])}")
    print(f"  LLM giriş: {b['llm_score_input']:.3f}")
    print(f"  Yayıncı  : {b['publisher_score']:.2f}")
    print(f"  Şehir ✓  : {b['city_match']}")
    print(f"  → SKOR   : {r['score']}")


def main() -> None:
    # Şehir listesini önceden çek
    ads = load_and_normalize(RAW_DATA_PATH, NORMALIZED_DATA_PATH, save=False)
    cities = list({a.get("city", "") for a in ads if a.get("city")})

    while True:
        params  = get_user_input(cities)
        results = run(**params)

        ads_map = {a["id"]: a for a in ads}
        print_results(results, ads_map, params["budget"], params["city"])

        try:
            again = input("\n  Yeni arama yapmak ister misiniz? (e/h): ").strip().lower()
        except (KeyboardInterrupt, EOFError):
            break
        if again not in ("e", "evet", "y", "yes"):
            break

    print("\n  Scout Agent kapatıldı.\n")


if __name__ == "__main__":
    main()
