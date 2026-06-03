import os
import json
import time
import requests
from bs4 import BeautifulSoup

def scrape_hotel_page(url, category_name):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }

    print(f"Scraping category [{category_name}] from: {url}...")
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status() 
    except requests.exceptions.RequestException as e:
        print(f"ERROR: Failed to fetch {url}. Reason: {e}")
        return None

    soup = BeautifulSoup(response.text, 'html.parser')
    
    for script in soup(["script", "style"]):
        script.extract()

    target_tags = ['h1', 'h2', 'h3', 'p', 'li']
    found_tags = soup.find_all(target_tags)
    
    extracted_text_segments = []
    for tag in found_tags:
        text = tag.get_text(strip=True)
        
        if len(text) > 25: 
            extracted_text_segments.append(text)

    full_content = " ".join(extracted_text_segments)
    
    if not full_content.strip():
        print(f"WARNING: No significant text content extracted from {url}")
        return None

    return {
        "category_name": category_name,
        "source_url": url,
        "content": full_content
    }


def main():
    os.makedirs("data", exist_ok=True)

    target_pages = [
        {
            "url": "https://www.jupiterhotel.com/faq", 
            "category": "FAQ & General Policies"
        },
        {
            "url": "https://www.jupiterhotel.com/guest-policies", 
            "category": "Guest Rules & Fees"
        },
        {
            "url": "https://www.jupiterhotel.com/terms-conditions", 
            "category": "Terms and Conditions"
        },
        {
            "url": "https://www.jupiterhotel.com/the-next", 
            "category": "Room Configurations & Amenities"
        }
    ]

    scraped_knowledge_base = []

    for page in target_pages:
        if "example.com" in page["url"]:
            print(f"Skipping placeholder URL: {page['url']}. Please replace with a real hotel URL.")
            continue
            
        data = scrape_hotel_page(page["url"], page["category"])
        if data:
            scraped_knowledge_base.append(data)
        
        time.sleep(2)

    if scraped_knowledge_base:
        output_path = os.path.join("data", "messy_hotel_kb.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(scraped_knowledge_base, f, indent=4, ensure_ascii=False)
        print(f"\nSuccess! Knowledge base written to: {output_path}")
    else:
        print("\nNo data was written because no valid custom URLs were executed.")


if __name__ == "__main__":
    main()