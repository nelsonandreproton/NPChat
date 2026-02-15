"""
Automatic scheduling for scraping and knowledge base updates.

Runs periodic tasks:
- Weekly scraping of new blog posts and company pages
- Daily cache cleanup
- Weekly knowledge gap report generation
"""
import subprocess
import sys
from pathlib import Path
from datetime import datetime, timezone

try:
    from apscheduler.schedulers.background import BackgroundScheduler
    from apscheduler.triggers.cron import CronTrigger
    APSCHEDULER_AVAILABLE = True
except ImportError:
    APSCHEDULER_AVAILABLE = False

BASE_DIR = Path(__file__).parent.parent


def run_scraper():
    """Run the blog post scraper."""
    print(f"[Scheduler] Running blog scraper at {datetime.now(timezone.utc).isoformat()}")
    try:
        result = subprocess.run(
            [sys.executable, "scraper.py"],
            cwd=str(BASE_DIR),
            capture_output=True,
            text=True,
            timeout=600
        )
        if result.returncode == 0:
            print("[Scheduler] Blog scraping completed successfully")
        else:
            print(f"[Scheduler] Blog scraping failed: {result.stderr[:500]}")
    except subprocess.TimeoutExpired:
        print("[Scheduler] Blog scraping timed out")
    except Exception as e:
        print(f"[Scheduler] Blog scraping error: {e}")


def run_company_scraper():
    """Run the company pages scraper."""
    print(f"[Scheduler] Running company pages scraper at {datetime.now(timezone.utc).isoformat()}")
    try:
        result = subprocess.run(
            [sys.executable, "scrape_company_pages.py"],
            cwd=str(BASE_DIR),
            capture_output=True,
            text=True,
            timeout=300
        )
        if result.returncode == 0:
            print("[Scheduler] Company pages scraping completed")
        else:
            print(f"[Scheduler] Company scraping failed: {result.stderr[:500]}")
    except Exception as e:
        print(f"[Scheduler] Company scraping error: {e}")


def run_ingestion():
    """Run the ingestion pipeline to update the knowledge base."""
    print(f"[Scheduler] Running ingestion at {datetime.now(timezone.utc).isoformat()}")
    try:
        result = subprocess.run(
            [sys.executable, "scripts/ingest_blogs.py"],
            cwd=str(BASE_DIR),
            capture_output=True,
            text=True,
            timeout=1200
        )
        if result.returncode == 0:
            print("[Scheduler] Ingestion completed successfully")
        else:
            print(f"[Scheduler] Ingestion failed: {result.stderr[:500]}")
    except Exception as e:
        print(f"[Scheduler] Ingestion error: {e}")


def run_full_update():
    """Run scraping + ingestion pipeline."""
    print(f"[Scheduler] Starting full update at {datetime.now(timezone.utc).isoformat()}")
    run_scraper()
    run_company_scraper()
    run_ingestion()
    print(f"[Scheduler] Full update completed at {datetime.now(timezone.utc).isoformat()}")


def run_cache_cleanup():
    """Clean up expired cache entries."""
    try:
        from src.analytics.response_cache import ResponseCache
        cache = ResponseCache()
        deleted = cache.clear_expired()
        print(f"[Scheduler] Cache cleanup: removed {deleted} expired entries")
    except Exception as e:
        print(f"[Scheduler] Cache cleanup error: {e}")


def generate_weekly_report():
    """Generate a weekly knowledge gap report."""
    try:
        from src.analytics.query_logger import QueryLogger
        from src.feedback.feedback_learner import FeedbackLearner

        logger = QueryLogger()
        learner = FeedbackLearner()

        stats = logger.get_stats()
        learning_stats = learner.get_stats()
        low_score = logger.get_low_score_queries(threshold=0.4, limit=20)
        negative = logger.get_negative_feedback_queries(limit=20)
        flagged = learner.get_flagged_queries(status='pending')

        report_lines = [
            f"=== Weekly Report - {datetime.now(timezone.utc).strftime('%Y-%m-%d')} ===",
            f"Total queries this week: {stats['total_queries']}",
            f"Average retrieval score: {stats['avg_retrieval_score']:.3f}",
            f"Positive feedback: {stats['positive_feedback']}",
            f"Negative feedback: {stats['negative_feedback']}",
            f"Chunks adjusted by learning: {learning_stats['chunks']['total_adjusted']}",
            "",
            f"--- Knowledge Gaps (low retrieval score) ---",
        ]

        for q in low_score[:10]:
            report_lines.append(f"  Score {q.avg_retrieval_score:.3f}: {q.query[:80]}")

        report_lines.append("")
        report_lines.append("--- Flagged Queries ---")
        for q in flagged[:10]:
            report_lines.append(f"  ({q['negative_count']}x negative): {q['query'][:80]}")

        report = "\n".join(report_lines)
        print(report)

        # Save report to file
        reports_dir = BASE_DIR / "data" / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        report_file = reports_dir / f"weekly_{datetime.now(timezone.utc).strftime('%Y%m%d')}.txt"
        report_file.write_text(report, encoding="utf-8")
        print(f"[Scheduler] Report saved to {report_file}")

    except Exception as e:
        print(f"[Scheduler] Report generation error: {e}")


def create_scheduler() -> "BackgroundScheduler":
    """
    Create and configure the background scheduler.

    Schedule:
    - Monday 02:00: Full update (scrape + ingest)
    - Daily 03:00: Cache cleanup
    - Sunday 23:00: Weekly report
    """
    if not APSCHEDULER_AVAILABLE:
        raise ImportError(
            "APScheduler not installed. Run: pip install apscheduler"
        )

    scheduler = BackgroundScheduler()

    # Weekly full update - Monday at 2 AM
    scheduler.add_job(
        run_full_update,
        CronTrigger(day_of_week="mon", hour=2, minute=0),
        id="weekly_update",
        name="Weekly scrape + ingest",
        replace_existing=True
    )

    # Daily cache cleanup - 3 AM
    scheduler.add_job(
        run_cache_cleanup,
        CronTrigger(hour=3, minute=0),
        id="daily_cache_cleanup",
        name="Daily cache cleanup",
        replace_existing=True
    )

    # Weekly report - Sunday at 11 PM
    scheduler.add_job(
        generate_weekly_report,
        CronTrigger(day_of_week="sun", hour=23, minute=0),
        id="weekly_report",
        name="Weekly knowledge gap report",
        replace_existing=True
    )

    return scheduler


def start_scheduler():
    """Start the background scheduler. Call this from the Streamlit app."""
    if not APSCHEDULER_AVAILABLE:
        print("[Scheduler] APScheduler not available - scheduling disabled")
        return None

    scheduler = create_scheduler()
    scheduler.start()
    print("[Scheduler] Background scheduler started")
    print("[Scheduler] Jobs scheduled:")
    for job in scheduler.get_jobs():
        print(f"  - {job.name}: {job.next_run_time}")
    return scheduler
