import json
import os
import re

class DataProcessor:
    def __init__(self, input_file="data/dataset.json"):
        self.input_file = input_file
        self.output_file = "data/normalized_ads.json"
        
        # Known cities and their prominent districts for force-correction
        self.city_district_map = {
            "İstanbul": ["Kadıköy", "Beşiktaş", "Ümraniye", "Sarıyer", "Maltepe", "Bahçelievler", "Esenyurt", "Beylikdüzü", "Pendik", "Üsküdar", "Şişli", "Fatih", "Ataşehir", "Bakırköy", "Başakşehir", "Beyoğlu", "Kartal", "Sancaktepe", "Tuzla", "Zeytinburnu"],
            "Ankara": ["Çankaya", "Keçiören", "Yenimahalle", "Mamak", "Etimesgut", "Sincan", "Altındağ", "Pursaklar", "Gölbaşı"],
            "İzmir": ["Karşıyaka", "Konak", "Bornova", "Buca", "Bayraklı", "Çiğli", "Gaziemir", "Balçova", "Narlıdere"],
            "Bursa": ["Nilüfer", "Osmangazi", "Yıldırım", "Mudanya", "Gemlik", "İnegöl"],
            "Antalya": ["Konyaaltı", "Muratpaşa", "Lara", "Kepez", "Alanya", "Manavgat", "Döşemealtı", "Kemer"]
        }
        self.cities = list(self.city_district_map.keys())

    def infer_city_from_text(self, text):
        """Looks for city names or prominent districts in text."""
        if not text:
            return None
        
        text_lower = text.lower()
        
        # 1. Direct city check
        for city in self.cities:
            if city.lower() in text_lower:
                return city
        
        # 2. District check
        for city, districts in self.city_district_map.items():
            for district in districts:
                if district.lower() in text_lower:
                    return city
                    
        return None

    def normalize(self):
        """Processes dataset.json and ads.json into normalized_ads.json for the app."""
        raw_ads = []
        
        # Load from multiple sources
        sources = ["data/dataset.json", "data/ads.json"]
        for source in sources:
            if os.path.exists(source):
                try:
                    with open(source, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        if isinstance(data, list):
                            raw_ads.extend(data)
                            print(f"Loaded {len(data)} ads from {source}")
                except Exception as e:
                    print(f"Error loading {source}: {e}")
            
        if not raw_ads:
            print("No raw data found in data/ directory!")
            return False
            
        normalized_ads = []
        seen_ads = set()
        
        for ad in raw_ads:
            # Simple De-duplication
            ad_fingerprint = (ad['title'].strip().lower(), ad.get('price', 0), ad.get('price_raw', ''))
            if ad_fingerprint in seen_ads:
                continue
            seen_ads.add(ad_fingerprint)
            
            # Clean Price
            price_numeric = 0
            if ad.get('price', 0) > 0:
                price_numeric = int(ad['price'])
            else:
                price_str = str(ad.get('price_raw', '0')).replace('.', '').replace(',', '')
                match = re.search(r'(\d+)', price_str)
                if match and int(match.group(1)) > 0:
                    price_numeric = int(match.group(1))
                else:
                    title_prices = re.findall(r'(\d{1,3}(?:\.\d{3})+)\s*[Tt][Ll]', ad['title'])
                    if title_prices:
                        price_numeric = int(title_prices[-1].replace('.', ''))

            # Clean Location
            location_raw = ad.get('location_raw', 'Bilinmiyor, Bilinmiyor')
            location_parts = location_raw.split(',')
            
            district_raw = location_parts[0].strip()
            city_raw = location_parts[1].strip() if len(location_parts) > 1 else "Bilinmiyor"
            
            # AGGRESSIVE CORRECTION - Only if raw city is unknown or clearly wrong
            city = city_raw
            district = district_raw

            # If city is not one of our known cities, try to infer
            if city not in self.cities:
                # Try infer from district first
                inferred_from_dist = self.infer_city_from_text(district_raw)
                if inferred_from_dist:
                    city = inferred_from_dist
                    district = district_raw.replace(inferred_from_dist, "").replace("-", "").strip()
                else:
                    # Try from title
                    inferred_from_title = self.infer_city_from_text(ad['title'])
                    if inferred_from_title:
                        city = inferred_from_title

            # Final check: if city is still not in our list, it's Bilinmiyor
            if city not in self.cities:
                city = "Bilinmiyor"

            # Normalize published date
            days_since = 1

            # Clean Listing Type
            listing_type = ad.get('listing_type')
            if not listing_type:
                text_to_check = (ad['title'] + " " + ad.get('description', '')).lower()
                if "satılık" in text_to_check:
                    listing_type = "Satılık"
                elif "kiralık" in text_to_check:
                    listing_type = "Kiralık"
                else:
                    # Fallback based on price if still unknown
                    if price_numeric > 200000:
                        listing_type = "Satılık"
                    else:
                        listing_type = "Kiralık"

            # Clean Area m2
            area_m2 = ad.get('area_m2')
            if not area_m2 or area_m2 == 100:
                text_to_check = (ad['title'] + " " + ad.get('description', '')).lower()
                m2_match = re.search(r'(\d+)\s*(?:m2|m²)', text_to_check)
                if m2_match:
                    area_m2 = int(m2_match.group(1))
                else:
                    area_m2 = 100 # Default

            # Clean Room Count (NEW)
            room_count = ad.get('room_count')
            if not room_count:
                text_to_check = (ad['title'] + " " + ad.get('description', '')).lower()
                room_match = re.search(r'(\d\+\d)', text_to_check)
                if room_match:
                    room_count = room_match.group(1)
                else:
                    room_count = "Bilinmiyor"

            normalized_ad = {
                "id": ad['id'],
                "listing_type": listing_type,
                "price": price_numeric,
                "city": city,
                "district": district,
                "room_count": room_count,
                "area_m2": area_m2,
                "title": ad['title'],
                "description": ad.get('description', ''),
                "image_count": ad.get('image_count', 0),
                "days_since_posted": days_since,
                "source": ad.get('source', 'Diğer'),
                "url": ad.get('url', '#')
            }
            normalized_ads.append(normalized_ad)
            
        with open(self.output_file, 'w', encoding='utf-8') as f:
            json.dump(normalized_ads, f, ensure_ascii=False, indent=2)
        
        print(f"Normalized {len(normalized_ads)} ads to {self.output_file}")
        return True

if __name__ == "__main__":
    processor = DataProcessor()
    processor.normalize()
