#!/usr/bin/env python3
"""
Export and backup tool for NPChat data.

Exports:
- Knowledge base (ChromaDB) as JSON
- Query logs
- Feedback data
- Learning data (chunk adjustments, flagged queries)
- Response cache stats
"""
import json
import shutil
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.config import config


def export_knowledge_base(output_dir: Path) -> int:
    """Export ChromaDB contents to JSON."""
    try:
        from src.retrieval.vector_store import VectorStore
        vs = VectorStore()
        collection = vs._collection

        all_data = collection.get(include=["documents", "metadatas"])
        docs = all_data.get("documents") or []
        metas = all_data.get("metadatas") or []
        ids = all_data.get("ids") or []

        export = [
            {"id": doc_id, "text": doc, "metadata": meta}
            for doc_id, doc, meta in zip(ids, docs, metas)
        ]

        output_file = output_dir / "knowledge_base.json"
        output_file.write_text(json.dumps(export, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  ✓ Exported {len(export)} chunks to {output_file}")
        return len(export)
    except Exception as e:
        print(f"  ✗ Knowledge base export failed: {e}")
        return 0


def export_sqlite_table(db_path: str, table: str, output_file: Path) -> int:
    """Export a SQLite table to JSON."""
    try:
        with sqlite3.connect(db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(f"SELECT * FROM {table}").fetchall()

        data = [dict(row) for row in rows]
        output_file.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  ✓ Exported {len(data)} rows from {table} to {output_file}")
        return len(data)
    except Exception as e:
        print(f"  ✗ Export of {table} failed: {e}")
        return 0


def backup_chroma_db(output_dir: Path):
    """Create a copy of the ChromaDB directory."""
    src = config.chroma_db_path
    dst = output_dir / "chroma_db_backup"
    if src.exists():
        shutil.copytree(src, dst, dirs_exist_ok=True)
        print(f"  ✓ ChromaDB backup copied to {dst}")
    else:
        print(f"  ⚠ ChromaDB not found at {src}")


def main():
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    export_dir = config.data_dir / "exports" / timestamp
    export_dir.mkdir(parents=True, exist_ok=True)

    print("═" * 60)
    print(f"  NPChat Data Export")
    print(f"  Timestamp: {timestamp}")
    print(f"  Output: {export_dir}")
    print("═" * 60)

    print("\n📚 Exporting knowledge base...")
    export_knowledge_base(export_dir)

    print("\n🗃️ Exporting ChromaDB backup...")
    backup_chroma_db(export_dir)

    print("\n📋 Exporting query logs...")
    export_sqlite_table(
        str(config.data_dir / "query_logs.db"),
        "query_logs",
        export_dir / "query_logs.json"
    )

    print("\n💬 Exporting feedback...")
    export_sqlite_table(
        str(config.feedback_db_path),
        "feedback",
        export_dir / "feedback.json"
    )

    print("\n🧠 Exporting learning data...")
    learning_db = str(config.data_dir / "feedback_learning.db")
    export_sqlite_table(learning_db, "chunk_adjustments", export_dir / "chunk_adjustments.json")
    export_sqlite_table(learning_db, "flagged_queries", export_dir / "flagged_queries.json")
    export_sqlite_table(learning_db, "query_mappings", export_dir / "query_mappings.json")

    print("\n⚡ Exporting cache entries...")
    export_sqlite_table(
        str(config.data_dir / "response_cache.db"),
        "response_cache",
        export_dir / "response_cache.json"
    )

    # Create a manifest
    manifest = {
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "export_dir": str(export_dir),
        "files": [str(f.name) for f in export_dir.iterdir()]
    }
    (export_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    print(f"\n✓ Export complete: {export_dir}")
    print("═" * 60)


if __name__ == "__main__":
    main()
