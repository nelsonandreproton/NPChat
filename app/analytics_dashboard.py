"""
Analytics dashboard for monitoring query performance and identifying improvements.
"""
import streamlit as st
import pandas as pd
import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.analytics.query_logger import QueryLogger

st.set_page_config(
    page_title="RAG Analytics",
    page_icon="📊",
    layout="wide"
)

st.title("📊 RAG Analytics Dashboard")

@st.cache_resource
def get_logger():
    return QueryLogger()

logger = get_logger()

# Overview stats
st.header("Overview")
stats = logger.get_stats()

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Total Queries", stats["total_queries"])
col2.metric("Avg Retrieval Score", f"{stats['avg_retrieval_score']:.3f}")
col3.metric("Avg Response Time", f"{stats['avg_response_time_ms']:.0f}ms")
col4.metric("👍 Positive", stats["positive_feedback"])
col5.metric("👎 Negative", stats["negative_feedback"])

st.divider()

# Tabs for different views
tab1, tab2, tab3, tab4 = st.tabs([
    "📋 Recent Queries",
    "⚠️ Low Score Queries",
    "👎 Negative Feedback",
    "🔥 Common Queries"
])

with tab1:
    st.subheader("Recent Queries")

    recent = logger.get_recent(limit=50)

    if recent:
        data = [{
            "Time": q.timestamp[:19],
            "Query": q.query[:80] + "..." if len(q.query) > 80 else q.query,
            "Avg Score": f"{q.avg_retrieval_score:.3f}",
            "Chunks": q.num_chunks_retrieved,
            "Time (ms)": q.response_time_ms,
            "Feedback": q.feedback or "-",
            "Expanded": "Yes" if q.expanded_query else "No"
        } for q in recent]

        df = pd.DataFrame(data)
        st.dataframe(df, use_container_width=True)
    else:
        st.info("No queries logged yet. Start using the chatbot to see data here.")

with tab2:
    st.subheader("Low Retrieval Score Queries")
    st.caption("These queries might indicate knowledge gaps in your content.")

    threshold = st.slider("Score threshold", 0.0, 1.0, 0.5)
    low_score = logger.get_low_score_queries(threshold=threshold, limit=30)

    if low_score:
        for q in low_score:
            with st.expander(f"Score: {q.avg_retrieval_score:.3f} | {q.query[:60]}..."):
                st.markdown(f"**Query:** {q.query}")
                st.markdown(f"**Time:** {q.timestamp}")
                st.markdown(f"**Retrieval Scores:** {q.retrieval_scores}")
                if q.expanded_query:
                    st.markdown(f"**Expanded Query:** {q.expanded_query}")

                st.divider()
                st.markdown("**Possible actions:**")
                st.markdown("- Add content covering this topic")
                st.markdown("- Improve chunking for related content")
                st.markdown("- Add synonyms/keywords to existing content")
    else:
        st.success("No low-score queries found!")

with tab3:
    st.subheader("Queries with Negative Feedback")
    st.caption("Users indicated these responses could be better.")

    negative = logger.get_negative_feedback_queries(limit=30)

    if negative:
        for q in negative:
            with st.expander(f"{q.timestamp[:10]} | {q.query[:60]}..."):
                st.markdown(f"**Query:** {q.query}")
                st.markdown(f"**Avg Score:** {q.avg_retrieval_score:.3f}")
                st.markdown(f"**Response Time:** {q.response_time_ms}ms")

                st.divider()
                st.markdown("**Analysis:**")
                if q.avg_retrieval_score < 0.5:
                    st.warning("Low retrieval score - relevant content may be missing")
                else:
                    st.info("Retrieval was OK - issue may be with LLM generation or prompt")
    else:
        st.success("No negative feedback received!")

with tab4:
    st.subheader("Most Common Queries")
    st.caption("Popular topics users are asking about.")

    common = logger.get_common_queries(limit=20)

    if common:
        data = [{
            "Query": q["query"][:60] + "..." if len(q["query"]) > 60 else q["query"],
            "Count": q["count"],
            "Avg Score": f"{q['avg_score']:.3f}" if q['avg_score'] else "N/A"
        } for q in common]

        df = pd.DataFrame(data)
        st.dataframe(df, use_container_width=True)

        st.divider()
        st.markdown("**Insights:**")
        st.markdown("- High count + low score = priority content gap")
        st.markdown("- High count + high score = your content is working well")
    else:
        st.info("No queries logged yet.")

# Footer
st.divider()
st.caption("💡 Tip: Use this data to identify content gaps and improve your knowledge base.")
