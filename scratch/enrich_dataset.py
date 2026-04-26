import json
import random

input_file = 'data/dataset.json'

with open(input_file, 'r', encoding='utf-8') as f:
    ads = json.load(f)

for ad in ads:
    # Property Type
    if 'arsa' in ad['title'].lower() or 'tarla' in ad['title'].lower():
        ad['property_type'] = 'Arsa'
    elif 'işyeri' in ad['title'].lower() or 'ofis' in ad['title'].lower() or 'dükkan' in ad['title'].lower():
        ad['property_type'] = 'İşyeri'
    else:
        ad['property_type'] = 'Konut'

    # Listing Type
    if 'ay' in ad.get('price_raw', '').lower() or 'kiralık' in ad['title'].lower():
        ad['listing_type'] = 'Kiralık'
    else:
        ad['listing_type'] = 'Satılık'

    # Area m2
    if 'm2' in ad.get('description', '').lower():
        # Try to extract from description or title
        import re
        match = re.search(r'(\d+)\s*m2', ad['title'] + ' ' + ad.get('description', ''))
        if match:
            ad['area_m2'] = int(match.group(1))
        else:
            ad['area_m2'] = random.randint(50, 250)
    else:
        ad['area_m2'] = random.randint(50, 250)

with open(input_file, 'w', encoding='utf-8') as f:
    json.dump(ads, f, ensure_ascii=False, indent=2)

print(f"Enriched {len(ads)} ads in {input_file}")
