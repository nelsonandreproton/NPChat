"""
RAGAS evaluation script for the NPChat RAG pipeline.

Usage:
    python scripts/evaluate_rag.py                    # run built-in question set
    python scripts/evaluate_rag.py --questions q.json # custom questions JSON
    python scripts/evaluate_rag.py --out results.json # custom output path
    python scripts/evaluate_rag.py --dry-run          # print questions only, no LLM calls

The built-in question set covers the main Near Partner topics so results are
comparable across runs and pipeline changes.

Output JSON is written to data/eval_results/<timestamp>.json and also printed
as a summary table.
"""
import argparse
import json
import sys
from datetime import datetime, UTC
from pathlib import Path

# Project root on path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import config
from src.generation.enhanced_rag_chain import EnhancedRAGChain
from src.evaluation.ragas_evaluator import RAGASEvaluator, EvalSample


# --------------------------------------------------------------------------- #
# Built-in evaluation question set                                             #
# --------------------------------------------------------------------------- #

BUILTIN_QUESTIONS = [
    "O que é a Near Partner?",
    "Quais são os serviços principais da Near Partner?",
    "O que é low-code development e como a Near Partner o usa?",
    "Como a Near Partner trabalha com Salesforce?",
    "O que é o modelo de partilha de risco da Near Partner?",
    "Que tecnologias de IA oferece a Near Partner?",
    "O que é transformação digital?",
    "What is Near Partner's approach to software development?",
    "How does Near Partner help with OutSystems development?",
    "What makes Near Partner different from other tech consultancies?",
]


# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #

def _load_questions(path: Path) -> list[str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list) and all(isinstance(q, str) for q in data):
        return data
    if isinstance(data, list) and all(isinstance(q, dict) and "question" in q for q in data):
        return [q["question"] for q in data]
    raise ValueError("Questions file must be a JSON array of strings or objects with a 'question' key.")


def _print_table(report) -> None:
    summary = {**report.summary(), "mean_overall": report.mean_overall()}
    print("\n" + "=" * 72)
    print(f"RAGAS Evaluation Report — {report.timestamp}")
    print(f"LLM: {report.model_llm}   Embeddings: {report.model_embedding}")
    print("=" * 72)
    print(f"{'Question':<45} {'Faith':>6} {'AnsRel':>6} {'CtxPr':>6} {'CtxRl':>6} {'Mean':>6}")
    print("-" * 72)
    for r in report.results:
        def fmt(v):
            return f"{v:.3f}" if v is not None else "  ERR"
        q = r.question[:44]
        print(f"{q:<45} {fmt(r.faithfulness):>6} {fmt(r.answer_relevancy):>6} "
              f"{fmt(r.context_precision):>6} {fmt(r.context_relevance):>6} "
              f"{fmt(r.mean_score()):>6}")
    print("-" * 72)
    def sfmt(v):
        return f"{v:.3f}" if v is not None else "  N/A"
    print(
        f"{'AVERAGE':<45} "
        f"{sfmt(summary['faithfulness']):>6} "
        f"{sfmt(summary['answer_relevancy']):>6} "
        f"{sfmt(summary['context_precision']):>6} "
        f"{sfmt(summary['context_relevance']):>6} "
        f"{sfmt(summary['mean_overall']):>6}"
    )
    print("=" * 72 + "\n")


# --------------------------------------------------------------------------- #
# Main                                                                         #
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser(description="Run RAGAS evaluation on the NPChat RAG pipeline")
    parser.add_argument("--questions", type=Path, help="Path to JSON file with questions")
    parser.add_argument("--out", type=Path, help="Output JSON path (default: data/eval_results/<timestamp>.json)")
    parser.add_argument("--dry-run", action="store_true", help="Print questions without running evaluation")
    args = parser.parse_args()

    # Load questions
    if args.questions:
        questions = _load_questions(args.questions)
        print(f"Loaded {len(questions)} questions from {args.questions}")
    else:
        questions = BUILTIN_QUESTIONS
        print(f"Using {len(questions)} built-in evaluation questions")

    if args.dry_run:
        print("\nQuestions:")
        for i, q in enumerate(questions, 1):
            print(f"  {i:2}. {q}")
        return

    # Build output path
    ts = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    out_path = args.out or (config.data_dir / "eval_results" / f"eval_{ts}.json")

    # Run RAG pipeline to collect (answer, contexts) for each question
    print(f"\nRunning RAG pipeline for {len(questions)} questions...")
    rag = EnhancedRAGChain(use_logging=False)
    samples: list[EvalSample] = []

    for i, question in enumerate(questions, 1):
        print(f"  [{i}/{len(questions)}] {question[:70]}")
        try:
            resp = rag.query(
                question=question,
                top_k=config.top_k,
                temperature=0.1,  # low temp for consistent eval answers
            )
            contexts = [c.get("text", "") for c in resp.retrieved_chunks if c.get("text")]
            samples.append(EvalSample(
                question=question,
                answer=resp.answer,
                contexts=contexts,
            ))
        except Exception as e:
            print(f"    RAG error: {e} — skipping")

    if not samples:
        print("No samples collected — is LM Studio running?")
        sys.exit(1)

    # Score with RAGAS
    print(f"\nScoring {len(samples)} samples with RAGAS metrics...")
    evaluator = RAGASEvaluator()
    report = evaluator.evaluate(samples)

    # Print table and save
    _print_table(report)
    report.save(out_path)
    print(f"Saved to: {out_path}")


if __name__ == "__main__":
    main()
