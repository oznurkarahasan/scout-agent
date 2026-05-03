import json
from collections import defaultdict

with open('data/normalized_ads.json', encoding='utf-8') as f:
    data = json.load(f)

# Kontrol et ne kadar "Bilinmiyor" var
missing_counts = defaultdict(int)
for ad in data:
    for key, val in ad.items():
        if val == 'Bilinmiyor' or val is None or val == '' or val == 0:
            missing_counts[key] += 1

print("Eksik/Boş alan sayıları:")
for k, v in sorted(missing_counts.items(), key=lambda x: -x[1]):
    print(f"  {k}: {v}")

# Şehir ve ilçe kombinasyonlarını göster
print("\nÖrnek eksik şehir ilanları:")
count = 0
for ad in data:
    if ad.get('city') == 'Bilinmiyor' or not ad.get('city'):
        print(f"  District: {ad.get('district')}, Room: {ad.get('room_count')}")
        count += 1
        if count >= 5:
            break

print("\nÖrnek eksik oda sayısı ilanları:")
count = 0
for ad in data:
    if ad.get('room_count') == 'Bilinmiyor' or not ad.get('room_count'):
        print(f"  City: {ad.get('city')}, District: {ad.get('district')}")
        count += 1
        if count >= 5:
            break
