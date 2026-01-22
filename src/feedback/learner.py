"""
Feedback-based learning and improvement system.

This module analyzes user feedback to improve the chatbot over time.
"""
from typing import List, Dict, Any, Optional
from collections import Counter
from datetime import datetime, timedelta
from .models import Feedback, FeedbackType, FeedbackStats, QueryPattern
from .store import FeedbackStore


class FeedbackLearner:
    """
    Analyzes feedback to identify patterns and suggest improvements.
    """

    def __init__(self, store: Optional[FeedbackStore] = None):
        """
        Initialize the learner.

        Args:
            store: FeedbackStore instance
        """
        self.store = store or FeedbackStore()

    def get_performance_summary(self) -> Dict[str, Any]:
        """
        Get overall performance summary.

        Returns:
            Dict with performance metrics
        """
        stats = self.store.get_stats()

        return {
            "total_interactions": stats.total_feedback,
            "satisfaction_rate": round(stats.satisfaction_rate * 100, 1),
            "thumbs_up": stats.thumbs_up,
            "thumbs_down": stats.thumbs_down,
            "corrections_received": stats.corrections,
            "knowledge_gaps_reported": stats.missing_info
        }

    def identify_problem_patterns(self, min_count: int = 2) -> List[QueryPattern]:
        """
        Identify query patterns that lead to negative feedback.

        Args:
            min_count: Minimum occurrences to be considered a pattern

        Returns:
            List of problematic patterns
        """
        # Get negative feedback
        negative = self.store.get_by_type(FeedbackType.THUMBS_DOWN, limit=500)
        negative.extend(self.store.get_by_type(FeedbackType.CORRECTION, limit=500))

        if not negative:
            return []

        # Extract common words/phrases from queries
        word_counts = Counter()
        query_examples = {}

        for fb in negative:
            # Simple word extraction (could be improved with NLP)
            words = fb.query.lower().split()
            for word in words:
                if len(word) > 3:  # Skip short words
                    word_counts[word] += 1
                    if word not in query_examples:
                        query_examples[word] = []
                    if len(query_examples[word]) < 3:
                        query_examples[word].append(fb.query)

        # Create patterns from common words
        patterns = []
        for word, count in word_counts.most_common(10):
            if count >= min_count:
                patterns.append(QueryPattern(
                    pattern=word,
                    feedback_type=FeedbackType.THUMBS_DOWN,
                    count=count,
                    example_queries=query_examples.get(word, []),
                    suggested_action=f"Review content related to '{word}' - users report issues"
                ))

        return patterns

    def get_knowledge_gaps(self) -> List[Dict[str, Any]]:
        """
        Identify topics where the chatbot lacks information.

        Returns:
            List of knowledge gap reports
        """
        missing_info = self.store.get_by_type(FeedbackType.MISSING_INFO, limit=100)

        gaps = []
        for fb in missing_info:
            gaps.append({
                "query": fb.query,
                "comment": fb.comment,
                "date": fb.created_at.isoformat() if fb.created_at else None
            })

        return gaps

    def get_correction_examples(self, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Get examples of user corrections for training/prompt improvement.

        Returns:
            List of correction examples
        """
        corrections = self.store.get_by_type(FeedbackType.CORRECTION, limit=limit)

        examples = []
        for fb in corrections:
            if fb.correction:
                examples.append({
                    "query": fb.query,
                    "original_response": fb.response,
                    "user_correction": fb.correction,
                    "comment": fb.comment
                })

        return examples

    def get_good_qa_pairs(self, limit: int = 100) -> List[Dict[str, str]]:
        """
        Get Q&A pairs that received positive feedback.
        These can be used for few-shot prompting or fine-tuning.

        Returns:
            List of good Q&A pairs
        """
        positive = self.store.get_by_type(FeedbackType.THUMBS_UP, limit=limit)

        pairs = []
        for fb in positive:
            pairs.append({
                "question": fb.query,
                "answer": fb.response
            })

        return pairs

    def get_chunk_performance(self) -> Dict[str, Dict[str, int]]:
        """
        Analyze which chunks lead to positive vs negative feedback.

        Returns:
            Dict mapping chunk_id to {positive: n, negative: n}
        """
        positive = self.store.get_by_type(FeedbackType.THUMBS_UP, limit=1000)
        negative = self.store.get_by_type(FeedbackType.THUMBS_DOWN, limit=1000)

        performance = {}

        for fb in positive:
            for chunk_id in fb.chunk_ids:
                if chunk_id not in performance:
                    performance[chunk_id] = {"positive": 0, "negative": 0}
                performance[chunk_id]["positive"] += 1

        for fb in negative:
            for chunk_id in fb.chunk_ids:
                if chunk_id not in performance:
                    performance[chunk_id] = {"positive": 0, "negative": 0}
                performance[chunk_id]["negative"] += 1

        return performance

    def generate_improvement_report(self) -> str:
        """
        Generate a human-readable improvement report.

        Returns:
            Markdown-formatted report
        """
        report = ["# Chatbot Improvement Report"]
        report.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")

        # Performance summary
        summary = self.get_performance_summary()
        report.append("## Performance Summary")
        report.append(f"- Total interactions: {summary['total_interactions']}")
        report.append(f"- Satisfaction rate: {summary['satisfaction_rate']}%")
        report.append(f"- Thumbs up: {summary['thumbs_up']}")
        report.append(f"- Thumbs down: {summary['thumbs_down']}")
        report.append(f"- Corrections: {summary['corrections_received']}")
        report.append(f"- Knowledge gaps: {summary['knowledge_gaps_reported']}")
        report.append("")

        # Problem patterns
        patterns = self.identify_problem_patterns()
        if patterns:
            report.append("## Problem Patterns")
            for p in patterns[:5]:
                report.append(f"\n### Pattern: '{p.pattern}' ({p.count} occurrences)")
                report.append(f"**Action:** {p.suggested_action}")
                report.append("Example queries:")
                for q in p.example_queries[:3]:
                    report.append(f"  - {q}")
            report.append("")

        # Knowledge gaps
        gaps = self.get_knowledge_gaps()
        if gaps:
            report.append("## Knowledge Gaps")
            for g in gaps[:10]:
                report.append(f"- **Query:** {g['query']}")
                if g['comment']:
                    report.append(f"  Comment: {g['comment']}")
            report.append("")

        # Recommendations
        report.append("## Recommendations")
        if summary['satisfaction_rate'] < 70:
            report.append("- Review and improve prompt templates")
            report.append("- Consider adding more blog content to knowledge base")
        if summary['corrections_received'] > 5:
            report.append("- Review correction examples for common mistakes")
        if summary['knowledge_gaps_reported'] > 3:
            report.append("- Prioritize content creation for identified gaps")

        return "\n".join(report)
