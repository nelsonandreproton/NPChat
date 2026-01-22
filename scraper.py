#!/usr/bin/env python3
"""
Near Partner Blog Scraper

Scrapes blog posts from nearpartner.com with incremental support.
Only scrapes new posts that aren't already in the JSON file.
"""

import requests
from bs4 import BeautifulSoup
import re
import json
import time
from urllib.parse import urljoin
from datetime import datetime, timezone
import os
from pathlib import Path
from typing import List, Dict, Optional, Set

# CONFIGURATION
BASE_URL = "https://www.nearpartner.com"
BLOG_BASE = f"{BASE_URL}/blog/"
OUTPUT_FILE = "nearpartner_blog_posts.json"
HEADERS = {
    "User-Agent": "NearPartnerBlogScraper/1.0 (+https://nearpartner.com; internal use)",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
}
REQUEST_TIMEOUT = 12
DELAY_BETWEEN_REQUESTS = 2.2


def get_soup(url: str) -> Optional[BeautifulSoup]:
    """Fetch URL and return BeautifulSoup object."""
    try:
        resp = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        return BeautifulSoup(resp.text, "html.parser")
    except requests.RequestException as e:
        print(f"  ✗ Failed to fetch {url} → {e}")
        return None


def extract_post_urls_from_page(soup: BeautifulSoup) -> List[str]:
    """Extract blog post URLs from a page."""
    urls = set()
    pattern = re.compile(r'/\d{4}/\d{2}/[^/]+/$')
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if pattern.search(href):
            full_url = urljoin(BASE_URL, href)
            urls.add(full_url)
    return sorted(urls)


def get_published_date_from_url(url: str) -> str:
    """Extract date from URL pattern /YYYY/MM/."""
    m = re.search(r'/(\d{4})/(\d{2})/', url)
    if m:
        return f"{m.group(1)}-{m.group(2).zfill(2)}"
    return "unknown"


def scrape_single_post(url: str) -> Optional[Dict]:
    """Scrape a single blog post."""
    soup = get_soup(url)
    if not soup:
        return None

    data: Dict = {
        "url": url,
        "scraped_at": datetime.now(timezone.utc).isoformat(),
        "page_type": "blog_post"
    }

    # Title - try multiple selectors
    title_tag = soup.find("h1", class_="post-title") or soup.find("h1")
    data["title"] = title_tag.get_text(strip=True) if title_tag else ""

    # Get full text content for metadata extraction
    content_container = soup.find("article", class_="post-content") or soup.find("article")
    if not content_container:
        content_container = soup

    full_text = content_container.get_text(separator="\n")

    # Extract metadata from the pattern: YYYY-MM-DD | Author Name | Category1 | Category2
    metadata_pattern = r'(\d{4}-\d{2}-\d{2})\s*\|\s*([^\|]+?)\s*\|\s*(.+?)\s*Written\s+By'
    metadata_match = re.search(metadata_pattern, full_text, re.DOTALL)

    if metadata_match:
        data["published_date"] = metadata_match.group(1)
        data["author"] = metadata_match.group(2).strip()
        cats_raw = metadata_match.group(3)
        categories = [c.strip() for c in re.split(r'\s*\|\s*', cats_raw) if c.strip()]
        data["categories"] = categories
    else:
        data["published_date"] = get_published_date_from_url(url)
        author_tag = soup.find("p", class_="author")
        data["author"] = author_tag.get_text(strip=True).replace("Written By", "").strip() if author_tag else "Not found"
        data["categories"] = []

    # Extract clean content
    clean_content = full_text

    # Remove everything before "Written By" (first occurrence)
    written_by_pos = clean_content.find("Written By")
    if written_by_pos != -1:
        after_author = clean_content[written_by_pos + 10:]
        lines = after_author.split('\n')
        content_start = 0
        for i, line in enumerate(lines):
            if len(line.strip()) > 50:
                content_start = i
                break
        clean_content = '\n'.join(lines[content_start:])

    # Remove everything after the end metadata
    end_metadata_pattern = r'\n\s*(\d{4}-\d{2}-\d{2})\s*\|\s*[^\|]+\s*\|\s*[^\n]+'
    end_match = re.search(end_metadata_pattern, clean_content)
    if end_match:
        clean_content = clean_content[:end_match.start()]

    # Remove "LATEST POST" sidebar content
    latest_post_pos = clean_content.find("LATEST POST")
    if latest_post_pos != -1:
        clean_content = clean_content[:latest_post_pos]

    # Clean up extra whitespace
    lines = [line.strip() for line in clean_content.split('\n') if line.strip()]
    data["content"] = '\n\n'.join(lines).strip()
    data["content_length_chars"] = len(data["content"])

    return data


def load_existing_posts() -> tuple[List[Dict], Set[str]]:
    """
    Load existing posts from JSON file.

    Returns:
        Tuple of (list of existing posts, set of existing URLs)
    """
    if not os.path.exists(OUTPUT_FILE):
        return [], set()

    try:
        with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
            posts = json.load(f)
        urls = {p.get("url") for p in posts if p.get("url")}
        return posts, urls
    except (json.JSONDecodeError, Exception) as e:
        print(f"  Warning: Could not load existing file: {e}")
        return [], set()


def save_posts(posts: List[Dict]):
    """Save posts to JSON file."""
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(posts, f, ensure_ascii=False, indent=2)


def discover_all_post_urls() -> List[str]:
    """Discover all blog post URLs from the blog pages."""
    print("\nDiscovering post URLs...")
    all_urls: List[str] = []
    seen_urls: Set[str] = set()
    page = 1

    while True:
        url = BLOG_BASE if page == 1 else f"{BLOG_BASE}page/{page}/"
        print(f"  Page {page} → {url}")
        soup = get_soup(url)
        if not soup:
            break

        new_urls = extract_post_urls_from_page(soup)
        new_count = 0
        for u in new_urls:
            if u not in seen_urls:
                all_urls.append(u)
                seen_urls.add(u)
                new_count += 1

        print(f"    Found {len(new_urls)} links → {new_count} new")
        if new_count == 0:
            break

        page += 1
        time.sleep(DELAY_BETWEEN_REQUESTS)

    return all_urls


def main():
    print("═" * 60)
    print("  Near Partner Blog Scraper")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("═" * 60)

    # Load existing posts
    print("\nChecking for existing posts...")
    existing_posts, existing_urls = load_existing_posts()
    print(f"  Found {len(existing_posts)} existing posts in {OUTPUT_FILE}")

    # Discover all post URLs
    all_post_urls = discover_all_post_urls()
    print(f"\nTotal unique post URLs found: {len(all_post_urls)}")

    # Filter to only new posts
    new_urls = [url for url in all_post_urls if url not in existing_urls]
    print(f"New posts to scrape: {len(new_urls)}")

    if not new_urls:
        print("\n✓ All posts are already scraped. Nothing to do!")
        print("═" * 60)
        return

    # Scrape new posts
    print(f"\nScraping {len(new_urls)} new posts...")
    new_posts: List[Dict] = []

    for i, url in enumerate(new_urls, 1):
        print(f"  [{i:3d}/{len(new_urls)}] ", end="")
        post = scrape_single_post(url)
        if post:
            new_posts.append(post)
            print(f"✓ {post.get('title', '—')[:60]}...")
        else:
            print(f"✗ Failed")
        time.sleep(DELAY_BETWEEN_REQUESTS)

    # Merge and save
    if new_posts:
        all_posts = existing_posts + new_posts
        # Sort by published date (newest first)
        all_posts.sort(key=lambda x: x.get("published_date", ""), reverse=True)
        save_posts(all_posts)
        print(f"\n✓ Saved {len(new_posts)} new posts")
        print(f"  Total posts in file: {len(all_posts)}")
    else:
        print("\nNo new posts were successfully scraped.")

    print(f"\nFinished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("═" * 60)


if __name__ == "__main__":
    main()
