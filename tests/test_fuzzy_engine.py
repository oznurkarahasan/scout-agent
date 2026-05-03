"""
Fuzzy engine davranış testleri.
3 senaryoyu doğrular:
  1. Orta değerler anlamlı skor üretiyor mu?
  2. Öncelik değişince sıralama değişiyor mu? (input scaling ile)
  3. Uç kombinasyonlar (efsane / çöp) doğru tetikleniyor mu?
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.fuzzy.engine import ScoutFuzzyEngine

engine = ScoutFuzzyEngine()

EQUAL_PRIORITIES = {'price': 0.5, 'location': 0.5, 'size': 0.5, 'quality': 0.5, 'llm': 0.5}
PRICE_FIRST      = {'price': 1.0, 'location': 0.1, 'size': 0.3, 'quality': 0.1, 'llm': 0.1}
LOCATION_FIRST   = {'price': 0.1, 'location': 1.0, 'size': 0.3, 'quality': 0.1, 'llm': 0.1}
ALL_HIGH         = {'price': 0.9, 'location': 0.9, 'size': 0.9, 'quality': 0.9, 'llm': 0.9}

def score(inputs, priorities=EQUAL_PRIORITIES):
    engine.prepare(priorities)
    result, _ = engine.compute(inputs)
    return round(result, 1)

def check(label, condition, got):
    status = "GEÇTI" if condition else "BAŞARISIZ"
    print(f"  [{status}] {label} → {got}")
    return condition

# ─────────────────────────────────────────────
print("\n=== SENARYO 1: Orta değerler anlamlı skor üretmeli ===")
orta_inputs = {
    'price_suitability': 60,
    'location_score':    6,
    'size_suitability':  65,
    'listing_quality':   6,
    'llm_alignment':     6,
}
s = score(orta_inputs)
results = [
    check("Skor 0'dan büyük olmalı",  s > 0,  s),
    check("Skor 30'dan büyük olmalı", s > 30, s),
    check("Skor 75'ten küçük olmalı", s < 75, s),
]
print(f"  Orta ilan skoru: %{s}")

# ─────────────────────────────────────────────
print("\n=== SENARYO 2: Öncelik sıralaması farklılaştırmalı (input scaling) ===")

# Pahalı ama yakın ilan
# → fiyat öncelikliyse: fiyat tam değer (düşük), konum nötre çekilir → skor düşük
# → konum öncelikliyse: konum tam değer (yüksek), fiyat nötre çekilir → skor yüksek
pahali_yakin = {
    'price_suitability': 10,
    'location_score':    10,
    'size_suitability':  65,
    'listing_quality':   6,
    'llm_alignment':     5,
}
s_price_first    = score(pahali_yakin, PRICE_FIRST)
s_location_first = score(pahali_yakin, LOCATION_FIRST)
print(f"  Pahalı+yakın ilan: fiyat_önce=%{s_price_first}  konum_önce=%{s_location_first}")
results += [
    check("Fiyat öncelikliyse skor daha düşük olmalı", s_price_first < s_location_first,
          f"fiyat_önce=%{s_price_first} < konum_önce=%{s_location_first}"),
]

# Ucuz ama uzak ilan
# → fiyat öncelikliyse: fiyat tam değer (yüksek), konum nötre çekilir → skor yüksek
# → konum öncelikliyse: konum tam değer (düşük), fiyat nötre çekilir → skor düşük
ucuz_uzak = {
    'price_suitability': 100,
    'location_score':    2,
    'size_suitability':  65,
    'listing_quality':   6,
    'llm_alignment':     5,
}
s_price_first_2    = score(ucuz_uzak, PRICE_FIRST)
s_location_first_2 = score(ucuz_uzak, LOCATION_FIRST)
print(f"  Ucuz+uzak ilan: fiyat_önce=%{s_price_first_2}  konum_önce=%{s_location_first_2}")
results += [
    check("Ucuz+uzak → fiyat öncelikliyse daha yüksek olmalı", s_price_first_2 > s_location_first_2,
          f"fiyat_önce=%{s_price_first_2} > konum_önce=%{s_location_first_2}"),
]

# ─────────────────────────────────────────────
print("\n=== SENARYO 3: Uç kombinasyonlar (efsane / çöp) ===")
efsane_inputs = {
    'price_suitability': 100,
    'location_score':    10,
    'size_suitability':  65,
    'listing_quality':   9,
    'llm_alignment':     10,
}
cop_inputs = {
    'price_suitability': 5,
    'location_score':    1,
    'size_suitability':  20,
    'listing_quality':   1,
    'llm_alignment':     5,
}
s_efsane = score(efsane_inputs, ALL_HIGH)
s_cop    = score(cop_inputs,    ALL_HIGH)
print(f"  Mükemmel ilan: %{s_efsane}")
print(f"  Kötü ilan:     %{s_cop}")
results += [
    check("Mükemmel ilan %80'den yüksek olmalı", s_efsane > 80, f"%{s_efsane}"),
    check("Kötü ilan %40'dan düşük olmalı",      s_cop < 40,   f"%{s_cop}"),
    check("Mükemmel > Kötü",                     s_efsane > s_cop, f"%{s_efsane} > %{s_cop}"),
]

# ─────────────────────────────────────────────
print("\n=== ÖZET ===")
passed = sum(results)
total  = len(results)
print(f"  {passed}/{total} test geçti\n")
