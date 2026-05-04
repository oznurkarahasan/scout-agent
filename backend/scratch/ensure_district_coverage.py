import json
import random
from pathlib import Path


DATA_PATH = Path("data/normalized_ads.json")
MIN_ADS_PER_DISTRICT = 3

TARGET_DISTRICTS = {
    "İstanbul": [
        "Adalar", "Arnavutköy", "Ataşehir", "Avcılar", "Bağcılar", "Bahçelievler", "Bakırköy", "Başakşehir",
        "Bayrampaşa", "Beşiktaş", "Beykoz", "Beylikdüzü", "Beyoğlu", "Büyükçekmece", "Çatalca", "Çekmeköy",
        "Esenler", "Esenyurt", "Eyüpsultan", "Fatih", "Gaziosmanpaşa", "Güngören", "Kadıköy", "Kağıthane",
        "Kartal", "Küçükçekmece", "Maltepe", "Pendik", "Sancaktepe", "Sarıyer", "Silivri", "Sultanbeyli",
        "Sultangazi", "Şile", "Şişli", "Tuzla", "Ümraniye", "Üsküdar", "Zeytinburnu",
    ],
    "Ankara": [
        "Akyurt", "Altındağ", "Ayaş", "Bala", "Beypazarı", "Çamlıdere", "Çankaya", "Çubuk", "Elmadağ", "Etimesgut",
        "Evren", "Gölbaşı", "Güdül", "Haymana", "Kalecik", "Kahramankazan", "Keçiören", "Kızılcahamam", "Mamak",
        "Nallıhan", "Polatlı", "Pursaklar", "Sincan", "Şereflikoçhisar", "Yenimahalle",
    ],
    "İzmir": [
        "Aliağa", "Balçova", "Bayındır", "Bayraklı", "Bergama", "Beydağ", "Bornova", "Buca", "Çeşme", "Çiğli",
        "Dikili", "Foça", "Gaziemir", "Güzelbahçe", "Karabağlar", "Karaburun", "Karşıyaka", "Kemalpaşa", "Kınık",
        "Kiraz", "Konak", "Menderes", "Menemen", "Narlıdere", "Ödemiş", "Seferihisar", "Selçuk", "Tire", "Torbalı", "Urla",
    ],
    "Bursa": [
        "Büyükorhan", "Gemlik", "Gürsu", "Harmancık", "İnegöl", "İznik", "Karacabey", "Keles", "Kestel", "Mudanya",
        "Mustafakemalpaşa", "Nilüfer", "Orhaneli", "Orhangazi", "Osmangazi", "Yenişehir", "Yıldırım",
    ],
    "Antalya": [
        "Akseki", "Aksu", "Alanya", "Demre", "Döşemealtı", "Elmalı", "Finike", "Gazipaşa", "Gündoğmuş", "İbradı",
        "Kaş", "Kemer", "Kepez", "Konyaaltı", "Korkuteli", "Kumluca", "Manavgat", "Muratpaşa", "Serik",
    ],
}


def district_base(value: str) -> str:
    text = (value or "").strip()
    if " - " in text:
        text = text.split(" - ", 1)[0].strip()
    if "  " in text:
        text = text.split("  ", 1)[0].strip()
    return text


def next_custom_id(existing_ids):
    max_id = 900
    for listing_id in existing_ids:
        if listing_id.startswith("ilan-") and listing_id[5:].isdigit():
            max_id = max(max_id, int(listing_id[5:]))
    return max_id + 1


def make_listing(new_id: str, city: str, district: str, listing_type: str, room_count: str, price: int, area_m2: int, source: str, day: int):
    title = f"{district} {room_count} {listing_type} Daire"
    description = (
        f"{district} bölgesinde ulaşımı kolay, bakımlı ve aile yaşamına uygun {room_count} {listing_type.lower()} daire."
    )
    return {
        "id": new_id,
        "listing_type": listing_type,
        "price": price,
        "city": city,
        "district": district,
        "room_count": room_count,
        "area_m2": area_m2,
        "title": title,
        "description": description,
        "image_count": random.randint(2, 8),
        "days_since_posted": day,
        "source": source,
        "url": f"https://www.google.com/search?q={city}+{district}+{room_count}+{listing_type}",
    }


def count_by_district(ads):
    counts = {}
    type_counts = {}

    for ad in ads:
        city = (ad.get("city") or "").strip()
        district = district_base(ad.get("district", ""))
        listing_type = (ad.get("listing_type") or "").strip()

        if not city or not district:
            continue

        key = (city, district)
        counts[key] = counts.get(key, 0) + 1

        if key not in type_counts:
            type_counts[key] = {"Kiralık": 0, "Satılık": 0}
        if listing_type in type_counts[key]:
            type_counts[key][listing_type] += 1

    return counts, type_counts


def build_listing_for_need(city, district, id_counter, sources, room_options, needed_type=None):
    listing_type = needed_type or ("Kiralık" if random.random() < 0.6 else "Satılık")
    room_count = random.choice(room_options)
    area_m2 = random.randint(70, 220)

    if listing_type == "Kiralık":
        price = random.randint(12000, 45000)
    else:
        price = random.randint(2200000, 9800000)

    new_id = f"ilan-{id_counter:05d}"
    item = make_listing(
        new_id=new_id,
        city=city,
        district=district,
        listing_type=listing_type,
        room_count=room_count,
        price=price,
        area_m2=area_m2,
        source=random.choice(sources),
        day=random.randint(1, 10),
    )
    return item


def main():
    random.seed(42)

    with DATA_PATH.open("r", encoding="utf-8") as file:
        ads = json.load(file)

    existing_ids = {str(ad.get("id", "")) for ad in ads}
    counts, type_counts = count_by_district(ads)

    id_counter = next_custom_id(existing_ids)
    added = []
    sources = ["sahibinden", "hepsiemlak", "emlakjet"]
    room_options = ["1+1", "2+1", "3+1", "4+1"]

    for city, districts in TARGET_DISTRICTS.items():
        for district in districts:
            key = (city, district)
            current_count = counts.get(key, 0)
            current_types = type_counts.get(key, {"Kiralık": 0, "Satılık": 0})

            while current_count < MIN_ADS_PER_DISTRICT:
                needed_type = None
                if current_types["Kiralık"] == 0:
                    needed_type = "Kiralık"
                elif current_types["Satılık"] == 0 and current_count >= 1:
                    needed_type = "Satılık"

                item = build_listing_for_need(
                    city=city,
                    district=district,
                    id_counter=id_counter,
                    sources=sources,
                    room_options=room_options,
                    needed_type=needed_type,
                )
                id_counter += 1

                added.append(item)
                current_count += 1
                listing_type = item["listing_type"]
                current_types[listing_type] += 1

            counts[key] = current_count
            type_counts[key] = current_types

    if added:
        ads.extend(added)
        with DATA_PATH.open("w", encoding="utf-8") as file:
            json.dump(ads, file, ensure_ascii=False, indent=2)

    print(f"Added {len(added)} listings.")
    for city, districts in TARGET_DISTRICTS.items():
        district_counts = [counts.get((city, district), 0) for district in districts]
        min_count = min(district_counts) if district_counts else 0
        print(f"{city}: min ads per district = {min_count}")


if __name__ == "__main__":
    main()
