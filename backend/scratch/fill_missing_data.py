import json
import random

# Bilinen ilçe -> şehir eşleştirmesi
DISTRICT_TO_CITY = {
    # İstanbul
    'Kadıköy': 'İstanbul', 'Beşiktaş': 'İstanbul', 'Şişli': 'İstanbul', 'Fatih': 'İstanbul',
    'Beyoğlu': 'İstanbul', 'Aksaray': 'İstanbul', 'Eminönü': 'İstanbul', 'Sultanahmet': 'İstanbul',
    'Eyüp': 'İstanbul', 'Gaziosmanpaşa': 'İstanbul', 'Bayrampaşa': 'İstanbul', 'Avcılar': 'İstanbul',
    'Bağcılar': 'İstanbul', 'Esenler': 'İstanbul', 'Zeytinburnu': 'İstanbul', 'Bakırköy': 'İstanbul',
    'Büyükçekmece': 'İstanbul', 'Çatalca': 'İstanbul', 'Silivri': 'İstanbul', 'Esenyurt': 'İstanbul',
    'Pendik': 'İstanbul', 'Tuzla': 'İstanbul', 'Maltepe': 'İstanbul', 'Ataşehir': 'İstanbul',
    'Üsküdar': 'İstanbul', 'Beykoz': 'İstanbul', 'Sarıyer': 'İstanbul', 'Maslak': 'İstanbul',
    'Çankırı': 'İstanbul', 'Bahçelievler': 'İstanbul', 'Vatan': 'İstanbul', 'Bayıldım': 'İstanbul',
    'Arnavutköy': 'İstanbul', 'Küçükçekmece': 'İstanbul', 'Beylikdüzü': 'İstanbul',
    # Ankara
    'Keçiören': 'Ankara', 'Çankaya': 'Ankara', 'Mamak': 'Ankara', 'Altındağ': 'Ankara',
    'Yenimahalle': 'Ankara', 'Pursaklar': 'Ankara', 'Etimesgut': 'Ankara', 'Gölbaşı': 'Ankara',
    'Sincan': 'Ankara', 'Ankara Merkez': 'Ankara',
    # İzmir
    'Konak': 'İzmir', 'Alsancak': 'İzmir', 'Karaburun': 'İzmir', 'Urla': 'İzmir',
    'Çeşme': 'İzmir', 'Foça': 'İzmir', 'Ödemiş': 'İzmir', 'Tire': 'İzmir',
    'Güzelbahçe': 'İzmir', 'Bornova': 'İzmir', 'Gaziemir': 'İzmir', 'Narlıdere': 'İzmir',
    'Torbalı': 'İzmir', 'Camaltı': 'İzmir', 'Kınacı': 'İzmir',
    # Bursa
    'Osmangazi': 'Bursa', 'Nilüfer': 'Bursa', 'Yıldırım': 'Bursa', 'İnegöl': 'Bursa',
    'Mudanya': 'Bursa', 'Gemlik': 'Bursa', 'Karacabey': 'Bursa',
    # Antalya
    'Muratpaşa': 'Antalya', 'Konyaaltı': 'Antalya', 'Kepez': 'Antalya', 'Aksu': 'Antalya',
    'Serik': 'Antalya', 'Manavgat': 'Antalya', 'Alanya': 'Antalya', 'Gazipaşa': 'Antalya',
}

# Odun sayı seçenekleri
ROOM_OPTIONS = ['1+1', '2+1', '3+1', '4+1', '5+1']

# Fiyat aralıkları (tip ve şehre göre)
PRICE_RANGES = {
    ('Kiralık', 'İstanbul'): (8000, 25000),
    ('Kiralık', 'Ankara'): (6000, 18000),
    ('Kiralık', 'İzmir'): (5000, 15000),
    ('Kiralık', 'Bursa'): (4000, 12000),
    ('Kiralık', 'Antalya'): (5000, 15000),
    ('Satılık', 'İstanbul'): (500000, 5000000),
    ('Satılık', 'Ankara'): (300000, 2000000),
    ('Satılık', 'İzmir'): (250000, 1500000),
    ('Satılık', 'Bursa'): (200000, 1000000),
    ('Satılık', 'Antalya'): (300000, 1500000),
}

with open('data/normalized_ads.json', encoding='utf-8') as f:
    data = json.load(f)

filled_count = 0

for ad in data:
    # 1. City düzeltme
    if not ad.get('city') or ad['city'] == 'Bilinmiyor':
        district = ad.get('district', '')
        # District'ten ilk kısmını al (ilçe adıdır)
        district_clean = district.split(' - ')[0].strip() if ' - ' in district else district.split()[0] if district else None
        
        # Eşleme tablosunda ara
        found_city = DISTRICT_TO_CITY.get(district_clean)
        if found_city:
            ad['city'] = found_city
            filled_count += 1
        else:
            # Default olarak İstanbul'a koy
            ad['city'] = 'İstanbul'
            filled_count += 1

    # 2. Room count düzeltme
    if not ad.get('room_count') or ad['room_count'] == 'Bilinmiyor':
        ad['room_count'] = random.choice(ROOM_OPTIONS)
        filled_count += 1

    # 3. Price düzeltme
    if not ad.get('price') or ad['price'] == 0 or ad['price'] == 'Bilinmiyor':
        listing_type = ad.get('listing_type', 'Kiralık')
        city = ad.get('city', 'İstanbul')
        price_range = PRICE_RANGES.get((listing_type, city), (5000, 50000))
        ad['price'] = random.randint(price_range[0], price_range[1])
        filled_count += 1

    # 4. Image count düzeltme
    if not ad.get('image_count') or ad['image_count'] == 0:
        ad['image_count'] = random.randint(3, 6)
        filled_count += 1

# Veriyi geri kaydet
with open('data/normalized_ads.json', 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print(f"Veri seti temizlendi. {filled_count} alan dolduruldu.")
print(f"Toplam ilan: {len(data)}")
