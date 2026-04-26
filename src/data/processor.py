import json
import os
import re

class DataProcessor:
    def __init__(self, input_file="data/dataset.json"):
        self.input_file = input_file
        self.output_file = "data/normalized_ads.json"

    def normalize(self):
        """Processes dataset.json into normalized_ads.json for the app."""
        if not os.path.exists(self.input_file):
            print(f"{self.input_file} not found!")
            return False
            
        with open(self.input_file, 'r', encoding='utf-8') as f:
            raw_ads = json.load(f)
            
        normalized_ads = []
        seen_ads = set()
        
        for ad in raw_ads:
            # Simple De-duplication based on title and description snippet
            # This handles ads with different IDs but same content
            ad_fingerprint = (ad['title'].strip().lower(), ad.get('price', 0), ad.get('price_raw', ''))
            if ad_fingerprint in seen_ads:
                continue
            seen_ads.add(ad_fingerprint)
            # Clean Price: Support both numeric 'price' and string 'price_raw'
            price_numeric = 0
            if ad.get('price', 0) > 0:
                price_numeric = int(ad['price'])
            else:
                # Fallback 1: price_raw
                price_str = str(ad.get('price_raw', '0')).replace('.', '').replace(',', '')
                match = re.search(r'(\d+)', price_str)
                if match and int(match.group(1)) > 0:
                    price_numeric = int(match.group(1))
                else:
                    # Fallback 2: Check title for price patterns like "2.050.000 TL"
                    title_prices = re.findall(r'(\d{1,3}(?:\.\d{3})+)\s*[Tt][Ll]', ad['title'])
                    if title_prices:
                        price_numeric = int(title_prices[-1].replace('.', ''))
                    else:
                        # Fallback 3: Check description
                        desc_prices = re.findall(r'(\d{1,3}(?:\.\d{3})+)\s*[Tt][Ll]', ad.get('description', ''))
                        if desc_prices:
                            price_numeric = int(desc_prices[-1].replace('.', ''))

            # Clean Location
            location_parts = ad.get('location_raw', 'Bilinmiyor, Bilinmiyor').split(',')
            district = location_parts[0].strip()
            city = location_parts[1].strip() if len(location_parts) > 1 else "Bilinmiyor"

            # Normalize published date to relative days (stub)
            days_since = 1 # Default

            normalized_ad = {
                "id": ad['id'],
                "title": ad['title'],
                "price": price_numeric,
                "city": city,
                "district": district,
                "description": ad.get('description', ''),
                "image_count": ad.get('image_count', 0),
                "days_since_posted": days_since,
                "publisher_type": ad.get('publisher_type', 'owner'),
                "is_featured": ad.get('is_featured', False),
                "source": ad.get('source', 'Diğer'),
                "views": ad.get('views', 0),
                "url": ad.get('url', '#'),
                "property_type": ad.get('property_type', 'Konut'),
                "listing_type": ad.get('listing_type', 'Kiralık'),
                "area_m2": ad.get('area_m2', 100)
            }
            normalized_ads.append(normalized_ad)
            
        with open(self.output_file, 'w', encoding='utf-8') as f:
            json.dump(normalized_ads, f, ensure_ascii=False, indent=2)
        
        print(f"Normalized {len(normalized_ads)} ads to {self.output_file}")
        return True

if __name__ == "__main__":
    processor = DataProcessor()
    processor.normalize()
