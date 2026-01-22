#!/usr/bin/env python3
"""
Script to update the knowledge base with new blog posts.

This script:
1. Runs the scraper to get latest blog posts
2. Compares with existing posts in vector store
3. Ingests only new posts

Usage:
    python scripts/update_knowledge.py              # Update with new posts
    python scripts/update_knowledge.py --scrape     # Scrape first, then update
    python scripts/update_knowledge.py --full       # Full re-scrape and re-ingest
"""
import argparse
import subprocess
import sys
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ingestion import IngestPipeline
from src.retrieval.vector_store import VectorStore
from src.config import config


def run_scraper(full: bool = False) -> bool:
    """
    Run the blog scraper.

    Args:
        full: If True, scrape all posts (modify scraper limit)

    Returns:
        True if successful
    """
    print("Running blog scraper...")
    try:
        result = subprocess.run(
            [sys.executable, "scraper.py"],
            cwd=config.base_dir,
            capture_output=True,
            text=True
        )
        print(result.stdout)
        if result.returncode != 0:
            print(f"Scraper error: {result.stderr}")
            return False
        return True
    except Exception as e:
        print(f"Failed to run scraper: {e}")
        return False


def get_knowledge_stats() -> dict:
    """Get current knowledge base statistics."""
    try:
        vs = VectorStore()
        return {
            "chunk_count": vs.count(),
            "url_count": len(vs.get_all_urls())
        }
    except Exception as e:
        return {"error": str(e)}


def main():
    parser = argparse.ArgumentParser(description="Update knowledge base with new blog posts")
    parser.add_argument(
        "--scrape",
        action="store_true",
        help="Run scraper before updating"
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Full re-scrape and re-ingest"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Near Partner Knowledge Base Update")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # Show current stats
    print("\nCurrent knowledge base stats:")
    stats = get_knowledge_stats()
    if "error" in stats:
        print(f"  Warning: {stats['error']}")
    else:
        print(f"  Chunks: {stats['chunk_count']}")
        print(f"  Blog posts: {stats['url_count']}")

    # Run scraper if requested
    if args.scrape or args.full:
        print("\n" + "-" * 40)
        if not run_scraper(full=args.full):
            print("Scraper failed. Aborting update.")
            sys.exit(1)

    # Check if blog posts file exists
    if not config.blog_posts_path.exists():
        print(f"\nError: Blog posts file not found: {config.blog_posts_path}")
        print("Run with --scrape flag to scrape blog posts first.")
        sys.exit(1)

    # Run ingestion
    print("\n" + "-" * 40)
    pipeline = IngestPipeline()

    if args.full:
        print("Running full re-ingestion...")
        result = pipeline.reingest_all()
    else:
        print("Running incremental ingestion...")
        result = pipeline.ingest_from_file(incremental=True)

    # Show results
    print("\n" + "=" * 60)
    print("Update complete!")
    print(f"  Posts processed: {result['posts']}")
    print(f"  Chunks added: {result['chunks']}")

    # Show new stats
    new_stats = get_knowledge_stats()
    if "error" not in new_stats:
        print(f"\nUpdated knowledge base stats:")
        print(f"  Total chunks: {new_stats['chunk_count']}")
        print(f"  Total blog posts: {new_stats['url_count']}")

    print(f"\nFinished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)


if __name__ == "__main__":
    main()
