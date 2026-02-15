#!/usr/bin/env python3
"""
Near Partner Company Pages Scraper

Scrapes static company pages (services, culture, resources, etc.)
and stores them in a format compatible with the RAG chatbot.
"""

import requests
from bs4 import BeautifulSoup
import json
import time
import os
from datetime import datetime, timezone
from typing import List, Dict, Optional

# Configuration
HEADERS = {
    "User-Agent": "NearPartnerScraper/1.0 (+https://nearpartner.com; internal use)",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
}
REQUEST_TIMEOUT = 15
DELAY_BETWEEN_REQUESTS = 2

OUTPUT_FILE = "nearpartner_company_pages.json"

# Company pages to scrape with their categories
COMPANY_PAGES = [
    # Homepage
    {
        "url": "https://www.nearpartner.com/",
        "category": "About",
        "title_override": "Near Partner - Homepage"
    },

    # Why Near Partner
    {
        "url": "https://www.nearpartner.com/why-near-partner/",
        "category": "About",
        "title_override": "Why Choose Near Partner"
    },

    # About / Team
    {
        "url": "https://www.nearpartner.com/about/",
        "category": "About",
        "title_override": "About Near Partner"
    },
    {
        "url": "https://www.nearpartner.com/team/",
        "category": "About",
        "title_override": "Near Partner Team"
    },

    # Services
    {
        "url": "https://www.nearpartner.com/low-code-development/",
        "category": "Services",
        "title_override": "Low-Code Development Services"
    },
    {
        "url": "https://www.nearpartner.com/salesforce-developer/",
        "category": "Services",
        "title_override": "Salesforce Development Services"
    },
    {
        "url": "https://www.nearpartner.com/software-development/",
        "category": "Services",
        "title_override": "Software Development Services"
    },
    {
        "url": "https://www.nearpartner.com/risk-sharing-model/",
        "category": "Services",
        "title_override": "Risk Sharing Model"
    },
    {
        "url": "https://www.nearpartner.com/ai-solutions/",
        "category": "Services",
        "title_override": "AI Solutions"
    },
    {
        "url": "https://www.nearpartner.com/digital-transformation/",
        "category": "Services",
        "title_override": "Digital Transformation"
    },

    # Technologies / Partners
    {
        "url": "https://www.nearpartner.com/technologies/",
        "category": "Technologies",
        "title_override": "Technologies & Partners"
    },
    {
        "url": "https://www.nearpartner.com/partners/",
        "category": "Technologies",
        "title_override": "Technology Partners"
    },
    {
        "url": "https://www.nearpartner.com/outsystems/",
        "category": "Technologies",
        "title_override": "OutSystems Development"
    },
    {
        "url": "https://www.nearpartner.com/mendix/",
        "category": "Technologies",
        "title_override": "Mendix Development"
    },

    # Resources
    {
        "url": "https://www.nearpartner.com/free-consultation/",
        "category": "Resources",
        "title_override": "Free Consultation"
    },
    {
        "url": "https://www.nearpartner.com/salesforce-diagnostic-tool/",
        "category": "Resources",
        "title_override": "Salesforce Audit & Diagnostic Tool"
    },
    {
        "url": "https://www.nearpartner.com/success-stories/",
        "category": "Resources",
        "title_override": "Success Stories"
    },
    {
        "url": "https://www.nearpartner.com/case-studies/",
        "category": "Resources",
        "title_override": "Case Studies"
    },

    # Culture & Values
    {
        "url": "https://www.nearpartner.com/culture/",
        "category": "Culture",
        "title_override": "Near Partner Culture & Values"
    },
    {
        "url": "https://www.nearpartner.com/careers/",
        "category": "Culture",
        "title_override": "Careers at Near Partner"
    },
    {
        "url": "https://www.nearpartner.com/jobs/",
        "category": "Culture",
        "title_override": "Job Openings"
    },

    # Contact
    {
        "url": "https://www.nearpartner.com/contact/",
        "category": "Contact",
        "title_override": "Contact Near Partner"
    },
    {
        "url": "https://www.nearpartner.com/contact-us/",
        "category": "Contact",
        "title_override": "Contact Us"
    },
]


def get_soup(url: str) -> Optional[BeautifulSoup]:
    """Fetch a URL and return BeautifulSoup object."""
    try:
        resp = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        return BeautifulSoup(resp.text, "html.parser")
    except requests.RequestException as e:
        print(f"  ✗ Failed to fetch {url} → {e}")
        return None


def clean_text(text: str) -> str:
    """Clean up extracted text."""
    # Remove extra whitespace
    lines = [line.strip() for line in text.split('\n') if line.strip()]

    # Remove common navigation/footer elements
    skip_phrases = [
        "LATEST POST", "Read More", "Learn More", "Get in Touch",
        "Cookie Policy", "Privacy Policy", "All Rights Reserved",
        "Follow us on", "LinkedIn", "Twitter", "Facebook",
        "Subscribe to our newsletter"
    ]

    filtered_lines = []
    for line in lines:
        # Skip very short lines (likely menu items)
        if len(line) < 10:
            continue
        # Skip lines that are just navigation
        if any(phrase.lower() in line.lower() for phrase in skip_phrases):
            continue
        filtered_lines.append(line)

    return '\n\n'.join(filtered_lines)


def extract_page_content(soup: BeautifulSoup, page_info: Dict) -> Dict:
    """Extract content from a company page."""

    # Try to find main content area
    content_selectors = [
        ("main", {}),
        ("article", {}),
        ("div", {"class": "elementor"}),
        ("div", {"class": "content"}),
        ("div", {"id": "content"}),
    ]

    content_container = None
    for tag, attrs in content_selectors:
        content_container = soup.find(tag, attrs) if attrs else soup.find(tag)
        if content_container:
            break

    if not content_container:
        content_container = soup.find("body")

    # Extract title
    title = page_info.get("title_override", "")
    if not title:
        h1 = soup.find("h1")
        title = h1.get_text(strip=True) if h1 else page_info["url"].split("/")[-2].replace("-", " ").title()

    # Extract all text content
    if content_container:
        # Remove script, style, nav, footer elements
        for tag in content_container.find_all(["script", "style", "nav", "footer", "header", "aside"]):
            tag.decompose()

        raw_text = content_container.get_text(separator="\n")
    else:
        raw_text = soup.get_text(separator="\n")

    # Clean the text
    content = clean_text(raw_text)

    # Extract any specific sections (headings and their content)
    sections = []
    for heading in soup.find_all(["h2", "h3"]):
        heading_text = heading.get_text(strip=True)
        if heading_text and len(heading_text) > 3:
            sections.append(heading_text)

    return {
        "url": page_info["url"],
        "title": title,
        "category": page_info["category"],
        "categories": [page_info["category"]],
        "content": content,
        "content_length_chars": len(content),
        "sections": sections,
        "page_type": "company_page",
        "scraped_at": datetime.now(timezone.utc).isoformat(),
        "author": "Near Partner"
    }


def scrape_success_stories(soup: BeautifulSoup, base_url: str) -> List[Dict]:
    """
    Special handler for success stories / case studies pages
    that may list multiple items.
    """
    stories = []

    # Look for individual story cards/links
    story_links = soup.find_all("a", href=True)
    story_urls = set()

    for link in story_links:
        href = link["href"]
        # Look for case study or success story URLs
        if "/case-study/" in href or "/success-story/" in href:
            if href.startswith("/"):
                href = "https://www.nearpartner.com" + href
            story_urls.add(href)

    # Scrape individual stories
    for url in list(story_urls)[:20]:  # Limit to 20
        print(f"    → Scraping story: {url}")
        story_soup = get_soup(url)
        if story_soup:
            story_data = extract_page_content(story_soup, {
                "url": url,
                "category": "Case Study",
                "title_override": ""
            })
            story_data["page_type"] = "case_study"
            stories.append(story_data)
            time.sleep(DELAY_BETWEEN_REQUESTS)

    return stories


def load_existing_pages() -> tuple[list, set]:
    """Load existing pages from JSON file."""
    if not os.path.exists(OUTPUT_FILE):
        return [], set()

    try:
        with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
            pages = json.load(f)
        urls = {p.get("url") for p in pages if p.get("url")}
        return pages, urls
    except (json.JSONDecodeError, Exception) as e:
        print(f"  Warning: Could not load existing file: {e}")
        return [], set()


def main():
    print("═" * 60)
    print("  Near Partner Company Pages Scraper")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("═" * 60)

    # Load existing pages
    print("\nChecking for existing pages...")
    existing_pages, existing_urls = load_existing_pages()
    print(f"  Found {len(existing_pages)} existing pages in {OUTPUT_FILE}")

    # Filter to only new pages
    pages_to_scrape = [p for p in COMPANY_PAGES if p["url"] not in existing_urls]
    print(f"  New pages to scrape: {len(pages_to_scrape)}")

    if not pages_to_scrape:
        print("\n✓ All company pages are already scraped. Nothing to do!")
        print("═" * 60)
        return

    results = []

    for i, page_info in enumerate(pages_to_scrape, 1):
        url = page_info["url"]
        print(f"\n[{i}/{len(pages_to_scrape)}] {page_info['category']}: {page_info.get('title_override', url)}")

        soup = get_soup(url)
        if not soup:
            print(f"  ⚠ Skipping {url} (page not found or error)")
            continue

        # Skip pages with very little content (likely 404 pages or empty redirects)
        page_data = extract_page_content(soup, page_info)
        if page_data['content_length_chars'] < 100:
            print(f"  ⚠ Skipping {url} (content too short: {page_data['content_length_chars']} chars)")
            continue

        results.append(page_data)
        print(f"  ✓ Extracted {page_data['content_length_chars']} chars")

        # Special handling for success stories / case studies
        if "success-stories" in url or "case-studies" in url:
            print("  → Looking for individual stories...")
            stories = scrape_success_stories(soup, url)
            if stories:
                print(f"  ✓ Found {len(stories)} individual stories")
                results.extend(stories)

        time.sleep(DELAY_BETWEEN_REQUESTS)

    # Merge and save results
    if results:
        all_pages = existing_pages + results
        print(f"\n{'═' * 60}")
        print(f"Saving {len(results)} new pages to {OUTPUT_FILE}...")

        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            json.dump(all_pages, f, ensure_ascii=False, indent=2)

        print(f"✓ Saved successfully!")
        print(f"  Total pages in file: {len(all_pages)}")

        # Summary
        print(f"\n{'─' * 40}")
        print("Summary by category:")
        categories = {}
        for r in all_pages:
            cat = r.get("category", "Unknown")
            categories[cat] = categories.get(cat, 0) + 1
        for cat, count in sorted(categories.items()):
            print(f"  {cat}: {count} pages")

        total_chars = sum(r["content_length_chars"] for r in all_pages)
        print(f"\nTotal content: {total_chars:,} characters")
    else:
        print("\nNo new pages were scraped.")

    print(f"\nFinished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("═" * 60)


if __name__ == "__main__":
    main()
