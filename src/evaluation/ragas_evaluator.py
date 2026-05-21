"""
RAGAS evaluation harness for the NPChat RAG pipeline.

Metrics (all reference-free — no ground-truth answers required):
  - Faithfulness:                  does the answer stick to the retrieved context?
  - AnswerRelevancy:               is the answer relevant to the question?
  - ContextPrecisionWithoutReference: are the retrieved chunks ranked well for the question?
  - ContextRelevance:              are the retrieved chunks relevant to the question?

All four use the local LM Studio LLM and embedding model so no OpenAI API key is needed.
"""
import asyncio
import json
from dataclasses import dataclass, field, asdict
from datetime import datetime, UTC
from pathlib import Path
from typing import List, Optional

from openai import OpenAI

from ..config import config


# --------------------------------------------------------------------------- #
# Data classes                                                                 #
# --------------------------------------------------------------------------- #

@dataclass
class EvalSample:
    question: str
    answer: str
    contexts: List[str]


@dataclass
class SampleResult:
    question: str
    faithfulness: Optional[float]
    answer_relevancy: Optional[float]
    context_precision: Optional[float]
    context_relevance: Optional[float]
    error: Optional[str] = None

    def mean_score(self) -> Optional[float]:
        scores = [
            s for s in [
                self.faithfulness,
                self.answer_relevancy,
                self.context_precision,
                self.context_relevance,
            ]
            if s is not None
        ]
        return round(sum(scores) / len(scores), 4) if scores else None


@dataclass
class EvalReport:
    timestamp: str
    model_llm: str
    model_embedding: str
    n_samples: int
    results: List[SampleResult] = field(default_factory=list)

    def summary(self) -> dict:
        def _avg(attr: str) -> Optional[float]:
            vals = [getattr(r, attr) for r in self.results if getattr(r, attr) is not None]
            return round(sum(vals) / len(vals), 4) if vals else None

        return {
            "faithfulness":      _avg("faithfulness"),
            "answer_relevancy":  _avg("answer_relevancy"),
            "context_precision": _avg("context_precision"),
            "context_relevance": _avg("context_relevance"),
        }

    def mean_overall(self) -> Optional[float]:
        scores = [r.mean_score() for r in self.results if r.mean_score() is not None]
        return round(sum(scores) / len(scores), 4) if scores else None

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "model_llm": self.model_llm,
            "model_embedding": self.model_embedding,
            "n_samples": self.n_samples,
            "summary": {**self.summary(), "mean_overall": self.mean_overall()},
            "results": [asdict(r) for r in self.results],
        }

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        print(f"[Evaluator] Report saved to {path}")


# --------------------------------------------------------------------------- #
# Evaluator                                                                    #
# --------------------------------------------------------------------------- #

class RAGASEvaluator:
    """
    Evaluates the RAG pipeline using four reference-free RAGAS metrics,
    all scored by the local LM Studio instance.
    """

    def __init__(self):
        self._client = OpenAI(
            base_url=config.lmstudio_base_url,
            api_key="lm-studio",
        )
        self._llm = None
        self._emb = None
        self._metrics_ready = False

    def _init_metrics(self):
        if self._metrics_ready:
            return
        from ragas.llms import llm_factory
        from ragas.embeddings import OpenAIEmbeddings as RagasOAIEmb
        from ragas.metrics.collections import (
            Faithfulness,
            AnswerRelevancy,
            ContextPrecisionWithoutReference,
            ContextRelevance,
        )

        self._llm = llm_factory(config.llm_model, client=self._client)
        self._emb = RagasOAIEmb(model=config.embedding_model, client=self._client)

        self._faithfulness = Faithfulness(llm=self._llm)
        self._answer_relevancy = AnswerRelevancy(llm=self._llm, embeddings=self._emb)
        self._context_precision = ContextPrecisionWithoutReference(llm=self._llm)
        self._context_relevance = ContextRelevance(llm=self._llm)

        self._metrics_ready = True

    async def _score_sample_async(self, sample: EvalSample) -> SampleResult:
        try:
            faithfulness, answer_rel, ctx_prec, ctx_rel = await asyncio.gather(
                self._faithfulness.ascore(
                    user_input=sample.question,
                    response=sample.answer,
                    retrieved_contexts=sample.contexts,
                ),
                self._answer_relevancy.ascore(
                    user_input=sample.question,
                    response=sample.answer,
                    retrieved_contexts=sample.contexts,
                ),
                self._context_precision.ascore(
                    user_input=sample.question,
                    response=sample.answer,
                    retrieved_contexts=sample.contexts,
                ),
                self._context_relevance.ascore(
                    user_input=sample.question,
                    retrieved_contexts=sample.contexts,
                ),
            )
            return SampleResult(
                question=sample.question,
                faithfulness=round(float(faithfulness.score), 4),
                answer_relevancy=round(float(answer_rel.score), 4),
                context_precision=round(float(ctx_prec.score), 4),
                context_relevance=round(float(ctx_rel.score), 4),
            )
        except Exception as e:
            return SampleResult(
                question=sample.question,
                faithfulness=None,
                answer_relevancy=None,
                context_precision=None,
                context_relevance=None,
                error=str(e),
            )

    def evaluate(self, samples: List[EvalSample]) -> EvalReport:
        """
        Score all samples and return a report.
        Runs metric calls concurrently per sample (4 awaited in parallel),
        then samples are processed sequentially to avoid overloading LM Studio.
        """
        self._init_metrics()

        report = EvalReport(
            timestamp=datetime.now(UTC).isoformat(),
            model_llm=config.llm_model,
            model_embedding=config.embedding_model,
            n_samples=len(samples),
        )

        for i, sample in enumerate(samples, 1):
            print(f"[Evaluator] Scoring sample {i}/{len(samples)}: {sample.question[:60]}...")
            result = asyncio.run(self._score_sample_async(sample))
            report.results.append(result)
            if result.error:
                print(f"[Evaluator]   ERROR: {result.error}")
            else:
                print(
                    f"[Evaluator]   faith={result.faithfulness}  "
                    f"ans_rel={result.answer_relevancy}  "
                    f"ctx_prec={result.context_precision}  "
                    f"ctx_rel={result.context_relevance}  "
                    f"mean={result.mean_score()}"
                )

        return report
