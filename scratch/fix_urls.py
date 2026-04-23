import json

input_path = "data/ads.json"
with open(input_path, 'r', encoding='utf-8') as f:
    ads = json.load(f)

for ad in ads:
    if 'url' not in ad or ad['url'] == '#':
        # Default to a search search if no direct URL
        ad['url'] = f"https://www.google.com/search?q={ad['title'].replace(' ', '+')}"

with open(input_path, 'w', encoding='utf-8') as f:
    json.dump(ads, f, ensure_ascii=False, indent=2)

print("Fixed URLs in ads.json")
