import asyncio
import json
import os
import re
from playwright.async_api import async_playwright
from bs4 import BeautifulSoup

class EmlakjetScraper:
    def __init__(self):
        self.base_url = "https://www.emlakjet.com"
        self.output_file = "data/ads.json"

    async def fetch_listings(self, city="istanbul", pages=1):
        """Fetches listings from Emlakjet for a specific city."""
        results = []
        
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            context = await browser.new_context(
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            )
            page = await context.new_page()
            
            # Block heavy resources like images and fonts to speed up
            await page.route("**/*.{png,jpg,jpeg,svg,woff,woff2}", lambda route: route.abort())

            for i in range(1, pages + 1):
                url = f"{self.base_url}/satilik-konut/{city}/{i}"
                try:
                    # 'domcontentloaded' is much faster than 'networkidle'
                    await page.goto(url, wait_until="domcontentloaded", timeout=30000)
                    # Give a tiny bit of time for Next.js to hydrate the core HTML
                    await page.wait_for_timeout(2000)
                    content = await page.content()
                    
                    soup = BeautifulSoup(content, 'html.parser')
                    cards = soup.select('a[class*="styles_wrapper"]')
                    
                    if cards:
                        for card in cards:
                            try:
                                title_elem = card.select_one('h3')
                                if not title_elem: continue
                                title = title_elem.get_text(strip=True)
                                
                                price_text = "0 TL"
                                price_elem = card.find('span', string=re.compile(r'TL'))
                                if price_elem:
                                    price_text = price_elem.get_text(strip=True)
                                
                                loc_text = "Bilinmiyor, Bilinmiyor"
                                loc_elem = card.select_one('div[class*="styles_location"]')
                                if loc_elem:
                                    loc_text = loc_elem.get_text(strip=True)
                                
                                meta_elem = card.select_one('div[class*="styles_characteristics"]')
                                meta_text = meta_elem.get_text(strip=True) if meta_elem else ""

                                ad_url = card.get('href', '')
                                if ad_url and not ad_url.startswith('http'):
                                    ad_url = f"{self.base_url}{ad_url}"

                                ad = {
                                    "id": f"ej-{hash(title + price_text + loc_text)}",
                                    "title": title,
                                    "price_raw": price_text,
                                    "location_raw": loc_text,
                                    "description": f"{meta_text} | {title}",
                                    "image_count": 5,
                                    "posted_date": "2026-04-22",
                                    "source": "Emlakjet",
                                    "publisher_type": "agent",
                                    "url": ad_url
                                }
                                results.append(ad)
                            except Exception:
                                continue
                except Exception as e:
                    print(f"Error on page {i}: {e}")
            
            await browser.close()
        return results

    def save_listings(self, listings):
        """Saves listings to data/ads.json (merging with existing ones)"""
        existing_ads = []
        if os.path.exists(self.output_file):
            try:
                with open(self.output_file, 'r', encoding='utf-8') as f:
                    existing_ads = json.load(f)
            except json.JSONDecodeError:
                pass
        
        # Merge (prevent duplicates by ID)
        ad_ids = {ad['id'] for ad in existing_ads}
        new_count = 0
        for ad in listings:
            if ad['id'] not in ad_ids:
                existing_ads.append(ad)
                new_count += 1
        
        with open(self.output_file, 'w', encoding='utf-8') as f:
            json.dump(existing_ads, f, ensure_ascii=False, indent=2)
        
        print(f"Saved {new_count} new listings to {self.output_file}")

async def main():
    scraper = EmlakjetScraper()
    # Fetch top 2 pages for Istanbul
    listings = await scraper.fetch_listings(city="istanbul", pages=2)
    scraper.save_listings(listings)

if __name__ == "__main__":
    asyncio.run(main())
