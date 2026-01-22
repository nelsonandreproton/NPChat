#!/usr/bin/env python3
"""
CLI script for ingesting content into the vector store.

This script handles both blog posts and company pages.

Usage:
    python scripts/ingest_blogs.py                    # Ingest all content (blog posts + company pages)
    python scripts/ingest_blogs.py --incremental      # Only add new content
    python scripts/ingest_blogs.py --blogs-only       # Only ingest blog posts
    python scripts/ingest_blogs.py --pages-only       # Only ingest company pages
"""
import argparse
import sys
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ingestion import IngestPipeline
from src.config import config


def main():
    parser = argparse.ArgumentParser(description="Ingest content into vector store")
    parser.add_argument(
        "--incremental",
        action="store_true",
        help="Only add new content (skip existing URLs)"
    )
    parser.add_argument(
        "--blogs-only",
        action="store_true",
        help="Only ingest blog posts"
    )
    parser.add_argument(
        "--pages-only",
        action="store_true",
        help="Only ingest company pages"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Near Partner Knowledge Base Ingestion")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # Check what files exist
    blog_exists = config.blog_posts_path.exists()
    pages_exists = config.company_pages_path.exists()

    if not blog_exists and not pages_exists:
        print("\nError: No content files found!")
        print("Run the scrapers first:")
        print("  python scraper.py                    # Scrape blog posts")
        print("  python scrape_company_pages.py       # Scrape company pages")
        sys.exit(1)

    print(f"\nContent sources:")
    print(f"  Blog posts:    {'Found' if blog_exists else 'Not found'} ({config.blog_posts_path})")
    print(f"  Company pages: {'Found' if pages_exists else 'Not found'} ({config.company_pages_path})")

    # Initialize pipeline
    pipeline = IngestPipeline()

    # Determine what to ingest
    if args.blogs_only:
        if not blog_exists:
            print("\nError: Blog posts file not found!")
            sys.exit(1)
        print("\nIngesting blog posts only...")
        if args.incremental:
            stats = pipeline.ingest_from_file(config.blog_posts_path, incremental=True)
        else:
            stats = pipeline.reingest_all(config.blog_posts_path)

    elif args.pages_only:
        if not pages_exists:
            print("\nError: Company pages file not found!")
            sys.exit(1)
        print("\nIngesting company pages only...")
        if args.incremental:
            stats = pipeline.ingest_from_file(config.company_pages_path, incremental=True)
        else:
            stats = pipeline.reingest_all(config.company_pages_path)

    else:
        # Ingest all content
        print("\nIngesting all content (blog posts + company pages)...")
        if args.incremental:
            # For incremental, process each file
            all_content = pipeline.load_all_content()
            new_content = pipeline.get_new_posts(all_content)
            stats = pipeline.ingest_posts(new_content) if new_content else {"posts": 0, "chunks": 0}
        else:
            # Full re-ingestion of all content
            stats = pipeline.reingest_all()

    print("\n" + "=" * 60)
    print("Ingestion complete!")
    print(f"  Content items processed: {stats['posts']}")
    print(f"  Chunks added: {stats['chunks']}")
    print(f"  Total chunks in store: {pipeline.vector_store.count()}")
    print(f"\nFinished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)


if __name__ == "__main__":
    main()
