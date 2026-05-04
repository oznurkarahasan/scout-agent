import json
import re

# Simulate app.py logic
def test_filtering():
    with open('data/normalized_ads.json', 'r', encoding='utf-8') as f:
        ads = json.load(f)
    
    target_city = "Ankara"
    target_listing_type = "Kiralık"
    target_rooms = ["Hepsi"]
    min_price = 5000
    max_price = 15000
    
    scored_ads = []
    for ad in ads:
        if ad.get('city') != target_city:
            continue
        if target_listing_type != "Hepsi" and ad.get('listing_type') != target_listing_type:
            continue
        if ad.get('price', 0) > max_price:
            continue
        
        scored_ads.append(ad)
    
    print(f"Total ads: {len(ads)}")
    print(f"Filtered ads: {len(scored_ads)}")
    
    wrong_city = [ad for ad in scored_ads if ad['city'] != target_city]
    print(f"Mismatched cities: {len(wrong_city)}")
    
    for ad in scored_ads[:5]:
        print(f"Title: {ad['title']}, City: {ad['city']}, Price: {ad['price']}")

if __name__ == "__main__":
    test_filtering()
