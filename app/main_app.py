"""
Unified Near Partner Chatbot App with Chat, Analytics, ChromaDB Browser, and Settings.
"""
import streamlit as st
import pandas as pd
import sys
import time
import threading
import subprocess
import json
from pathlib import Path
from typing import List

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.generation.enhanced_rag_chain import EnhancedRAGChain
from src.retrieval.vector_store import VectorStore
from src.retrieval.retriever import Retriever
from src.feedback.store import FeedbackStore
from src.feedback.models import Feedback, FeedbackType
from src.feedback.feedback_learner import FeedbackLearner
from src.analytics.query_logger import QueryLogger
from src.analytics.response_cache import ResponseCache

# Page config
st.set_page_config(
    page_title="Near Partner Chatbot",
    page_icon="🤖",
    layout="wide"
)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "feedback_given" not in st.session_state:
    st.session_state.feedback_given = set()

if "scrape_status" not in st.session_state:
    st.session_state.scrape_status = None

if "ingest_status" not in st.session_state:
    st.session_state.ingest_status = None

if "scheduler_started" not in st.session_state:
    st.session_state.scheduler_started = False

_SETTINGS_FILE = Path(__file__).parent.parent / "data" / "app_settings.json"
# Defaults: LEAN config (hybrid + reranking + generation = 1 LLM call per query).
# Pre-retrieval LLM steps (expansion / HyDE / multi-query) are off by default —
# on this single-slot ~12 tok/s server they each add ~3-12s for negligible recall
# gain. Left as opt-in toggles for hard-recall queries. See
# docs/latency_ablation_findings.md.
_DEFAULT_SETTINGS = {
    "top_k": 5,
    "temperature": 0.7,
    "use_expansion": False,
    "use_hybrid": True,
    "use_hyde": False,
    "use_reranking": True,
    "use_multi_query": False,
    "use_contextual_retrieval": False,
    "use_cache": True,
    "cache_ttl_hours": 24,
    "evaluate_confidence": False,
    "show_confidence": True,
}


def _load_settings() -> dict:
    try:
        if _SETTINGS_FILE.exists():
            saved = json.loads(_SETTINGS_FILE.read_text(encoding="utf-8"))
            # Merge with defaults so new keys added in future are always present
            return {**_DEFAULT_SETTINGS, **saved}
    except Exception:
        pass
    return dict(_DEFAULT_SETTINGS)


def _save_settings(settings: dict) -> None:
    try:
        _SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
        _SETTINGS_FILE.write_text(json.dumps(settings, indent=2), encoding="utf-8")
    except Exception as e:
        st.error(f"Erro ao guardar definições: {e}")


# Default settings — loaded from disk on first run of each browser session
if "settings" not in st.session_state:
    st.session_state.settings = _load_settings()

# Start background scheduler once per session
if not st.session_state.scheduler_started:
    try:
        from src.scheduler import start_scheduler
        start_scheduler()
        st.session_state.scheduler_started = True
    except Exception:
        pass  # Scheduler is optional


# Cached resources
@st.cache_resource
def get_rag_chain(use_expansion: bool, use_hybrid: bool, use_reranking: bool = True, use_multi_query: bool = True):
    """Initialize enhanced RAG chain."""
    return EnhancedRAGChain(
        use_query_expansion=use_expansion,
        use_hybrid_search=use_hybrid,
        use_logging=True,
        use_reranking=use_reranking,
        use_multi_query=use_multi_query,
    )


@st.cache_resource
def get_vector_store():
    """Initialize vector store."""
    return VectorStore()


@st.cache_resource
def get_retriever():
    """Initialize retriever."""
    return Retriever()


@st.cache_resource
def get_feedback_store():
    """Initialize feedback store."""
    return FeedbackStore()


@st.cache_resource
def get_query_logger():
    """Initialize query logger."""
    return QueryLogger()


@st.cache_resource
def get_response_cache():
    """Initialize response cache."""
    return ResponseCache()


@st.cache_resource
def get_feedback_learner():
    """Initialize feedback learner."""
    return FeedbackLearner()


def save_feedback(query: str, response: str, feedback_type: FeedbackType, log_id: int = None, chunk_ids: List[str] = None):
    """Save user feedback and apply automatic learning actions."""
    try:
        store = get_feedback_store()
        feedback = Feedback(
            id="",
            query=query,
            response=response,
            feedback_type=feedback_type,
            correction=None
        )
        store.add(feedback)

        # Update analytics log
        if log_id:
            try:
                logger = get_query_logger()
                feedback_str = "positive" if feedback_type == FeedbackType.THUMBS_UP else "negative"
                logger.update_feedback(log_id, feedback_str)
            except Exception:
                pass

        # Apply automatic feedback learning actions
        try:
            learner = get_feedback_learner()
            cache = get_response_cache()
            is_positive = feedback_type == FeedbackType.THUMBS_UP

            actions = learner.process_feedback(
                query=query,
                is_positive=is_positive,
                chunk_ids=chunk_ids or [],
                cache=cache
            )

            # Log what actions were taken
            if actions.get("cache_invalidated"):
                print(f"[Feedback] Cache invalidated for query: {query[:50]}...")
            if actions.get("query_flagged"):
                print(f"[Feedback] Query flagged: {actions.get('flag_reason')}")
            if actions.get("query_learned"):
                print(f"[Feedback] Learned successful query mapping")

        except Exception as e:
            print(f"[Feedback] Learning error: {e}")

        return True
    except Exception:
        return False


def render_chat_tab():
    """Render the chat interface."""
    st.header("💬 Chat")
    st.markdown("Ask me anything about Near Partner's services, technology insights, and expertise!")
    st.caption(
        "🤖 Está a conversar com um assistente de IA (LLM local, sem envio de dados para "
        "serviços externos). As respostas podem conter imprecisões — verifique informação "
        "crítica. Ver [PRIVACY.md](https://github.com/nelsonandreproton/NPChat/blob/main/PRIVACY.md) "
        "para saber que dados são guardados."
    )

    settings = st.session_state.settings
    cache = get_response_cache()

    # Display chat history
    for i, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            if message["role"] == "assistant":
                # Show if from cache
                if message.get("cached"):
                    st.caption("⚡ From cache")

                # Show sources
                if message.get("sources"):
                    with st.expander("📚 Sources"):
                        for source in message["sources"]:
                            st.markdown(f"- [{source['title']}]({source['url']}) by {source['author']}")

                # Feedback buttons - only show if not already given
                msg_key = f"msg_{i}"
                if msg_key not in st.session_state.feedback_given:
                    col1, col2, col3 = st.columns([1, 1, 6])
                    original_query = st.session_state.messages[i-1]["content"] if i > 0 else ""
                    log_id = message.get("log_id")
                    chunk_ids = message.get("chunk_ids", [])

                    with col1:
                        if st.button("👍", key=f"up_{i}"):
                            if save_feedback(original_query, message["content"], FeedbackType.THUMBS_UP, log_id=log_id, chunk_ids=chunk_ids):
                                st.session_state.feedback_given.add(msg_key)
                                st.rerun()

                    with col2:
                        if st.button("👎", key=f"down_{i}"):
                            if save_feedback(original_query, message["content"], FeedbackType.THUMBS_DOWN, log_id=log_id, chunk_ids=chunk_ids):
                                st.session_state.feedback_given.add(msg_key)
                                st.rerun()
                else:
                    st.caption("✓ Feedback received")

    # Chat input
    if prompt := st.chat_input("Ask a question about Near Partner..."):
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            thinking_placeholder = st.empty()
            response_placeholder = st.empty()

            try:
                # Check cache first
                cached_response = None
                if settings["use_cache"]:
                    cached_response = cache.get(prompt, settings)

                if cached_response:
                    # Use cached response
                    response_placeholder.markdown(cached_response["answer"])
                    st.caption("⚡ From cache (instant)")

                    if cached_response["sources"]:
                        with st.expander("📚 Sources"):
                            for source in cached_response["sources"]:
                                st.markdown(f"- [{source['title']}]({source['url']}) by {source['author']}")

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": cached_response["answer"],
                        "sources": cached_response["sources"],
                        "cached": True
                    })

                else:
                    # Generate new response
                    rag_chain = get_rag_chain(
                        use_expansion=settings["use_expansion"],
                        use_hybrid=settings["use_hybrid"],
                        use_reranking=settings.get("use_reranking", True),
                        use_multi_query=settings.get("use_multi_query", True),
                    )

                    result_container = {"result": None, "error": None, "done": False}

                    # Build conversation history for multi-turn context
                    history = [
                        {"role": m["role"], "content": m["content"]}
                        for m in st.session_state.messages[-10:]  # Last 5 turns
                        if m["role"] in ("user", "assistant")
                    ]

                    def run_query():
                        try:
                            result_container["result"] = rag_chain.query(
                                question=prompt,
                                top_k=settings["top_k"],
                                temperature=settings["temperature"],
                                use_hyde=settings["use_hyde"],
                                conversation_history=history,
                                evaluate_confidence=settings.get("evaluate_confidence", False)
                            )
                        except Exception as e:
                            result_container["error"] = e
                        finally:
                            result_container["done"] = True

                    query_thread = threading.Thread(target=run_query)
                    query_thread.start()

                    start_time = time.time()
                    while not result_container["done"]:
                        elapsed = int(time.time() - start_time)
                        thinking_placeholder.markdown(f"⏳ *Thinking for {elapsed} seconds...*")
                        time.sleep(0.5)

                    thinking_placeholder.empty()

                    if result_container["error"]:
                        raise result_container["error"]

                    result = result_container["result"]
                    elapsed_total = round(time.time() - start_time, 1)

                    response_placeholder.markdown(result.answer)

                    # Low confidence warning
                    if getattr(result, 'low_confidence', False):
                        st.warning("⚠️ Nota: A confiança nesta resposta é baixa. Pode não haver informação suficiente na base de conhecimento.")

                    if hasattr(result, 'timings') and result.timings:
                        t = result.timings
                        rerank_str = f" | Rerank: {t['reranking']}s" if "reranking" in t else ""
                        confidence_str = ""
                        if settings.get("show_confidence") and result.confidence_score is not None:
                            confidence_str = f" | Confiança: {result.confidence_score:.0%}"
                        st.caption(
                            f"⏱️ Retrieval: {t.get('retrieval', 0)}s"
                            f"{rerank_str} | "
                            f"LLM: {t.get('llm_generation', 0)}s | "
                            f"Total: {t.get('total', elapsed_total)}s"
                            f"{confidence_str}"
                        )

                    if result.sources:
                        with st.expander("📚 Sources"):
                            for source in result.sources:
                                st.markdown(f"- [{source['title']}]({source['url']}) by {source['author']}")

                    # Cache the response
                    if settings["use_cache"]:
                        cache.set(prompt, settings, result.answer, result.sources)

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": result.answer,
                        "sources": result.sources,
                        "chunk_ids": getattr(result, 'chunk_ids', []),
                        "log_id": getattr(result, 'log_id', None),
                        "cached": False
                    })

                st.rerun()

            except Exception as e:
                thinking_placeholder.empty()
                st.error(f"Error: {str(e)}")
                if "connection" in str(e).lower() or "refused" in str(e).lower():
                    st.warning("LLM server not reachable. Run start_llm.ps1 to start llama-server.")

    # Sidebar for chat
    with st.sidebar:
        st.header("Chat Controls")

        try:
            vs = get_vector_store()
            st.metric("Knowledge Base", f"{vs.count()} chunks")
        except Exception:
            st.warning("KB not connected")

        cache_stats = cache.get_stats()
        st.metric("Cache Entries", cache_stats["total_entries"])

        if st.button("🗑️ Clear Chat"):
            st.session_state.messages = []
            st.session_state.feedback_given = set()
            st.rerun()


def render_analytics_tab():
    """Render the analytics dashboard."""
    st.header("📊 Analytics Dashboard")

    logger = get_query_logger()
    cache = get_response_cache()

    # Overview
    stats = logger.get_stats()
    cache_stats = cache.get_stats()

    col1, col2, col3, col4, col5, col6 = st.columns(6)
    col1.metric("Queries", stats["total_queries"])
    col2.metric("Avg Score", f"{stats['avg_retrieval_score']:.3f}")
    col3.metric("Avg Time", f"{stats['avg_response_time_ms']:.0f}ms")
    col4.metric("👍", stats["positive_feedback"])
    col5.metric("👎", stats["negative_feedback"])
    col6.metric("Cache Hits", cache_stats["total_hits"])

    st.divider()

    sub_tab1, sub_tab2, sub_tab3, sub_tab4, sub_tab5, sub_tab6 = st.tabs([
        "📋 Recent", "⚠️ Low Score", "👎 Negative", "🔥 Common", "⚡ Cache", "🧠 Learning"
    ])

    with sub_tab1:
        recent = logger.get_recent(limit=50)
        if recent:
            data = [{
                "Time": q.timestamp[:19],
                "Query": q.query[:50] + "..." if len(q.query) > 50 else q.query,
                "Score": f"{q.avg_retrieval_score:.3f}",
                "ms": q.response_time_ms,
                "FB": q.feedback or "-"
            } for q in recent]
            st.dataframe(pd.DataFrame(data), width="stretch", hide_index=True)
        else:
            st.info("No queries logged yet.")

    with sub_tab2:
        threshold = st.slider("Score threshold", 0.0, 1.0, 0.5, key="thresh")
        low_score = logger.get_low_score_queries(threshold=threshold, limit=30)
        if low_score:
            for q in low_score:
                with st.expander(f"Score: {q.avg_retrieval_score:.3f} | {q.query[:40]}..."):
                    st.markdown(f"**Query:** {q.query}")
                    st.caption(f"Time: {q.timestamp}")
        else:
            st.success("No low-score queries!")

    with sub_tab3:
        negative = logger.get_negative_feedback_queries(limit=30)
        if negative:
            for q in negative:
                with st.expander(f"{q.timestamp[:10]} | {q.query[:40]}..."):
                    st.markdown(f"**Query:** {q.query}")
                    st.markdown(f"**Score:** {q.avg_retrieval_score:.3f}")
        else:
            st.success("No negative feedback!")

    with sub_tab4:
        common = logger.get_common_queries(limit=20)
        if common:
            data = [{
                "Query": q["query"][:40],
                "Count": q["count"],
                "Avg Score": f"{q['avg_score']:.3f}" if q['avg_score'] else "-"
            } for q in common]
            st.dataframe(pd.DataFrame(data), width="stretch", hide_index=True)
        else:
            st.info("No queries logged yet.")

    with sub_tab5:
        st.subheader("Response Cache")
        st.json(cache_stats)

        recent_cache = cache.get_recent(limit=10)
        if recent_cache:
            st.markdown("**Recent cached queries:**")
            for item in recent_cache:
                st.markdown(f"- {item['query'][:50]}... (hits: {item['hits']})")

    with sub_tab6:
        st.subheader("Feedback Learning")

        learner = get_feedback_learner()
        learning_stats = learner.get_stats()

        # Overview metrics
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Chunks Adjusted", learning_stats["chunks"]["total_adjusted"])
        col2.metric("Boosted", learning_stats["chunks"]["boosted"])
        col3.metric("Penalized", learning_stats["chunks"]["penalized"])
        col4.metric("Queries Learned", learning_stats["mappings"]["total_queries"])

        st.divider()

        # Flagged Queries Section
        st.markdown("### 🚩 Flagged Queries")
        st.caption("Queries that received multiple negative feedbacks and need attention")

        flagged = learner.get_flagged_queries(status='pending')
        if flagged:
            for item in flagged:
                with st.expander(f"⚠️ {item['query'][:50]}... ({item['negative_count']} negative)"):
                    st.markdown(f"**Query:** {item['query']}")
                    st.markdown(f"**Reason:** {item['flag_reason']}")
                    st.markdown(f"**Created:** {item['created_at'][:19]}")

                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("✅ Mark Resolved", key=f"resolve_{item['query'][:20]}"):
                            learner.resolve_flag(item['query'], 'resolved')
                            st.success("Marked as resolved")
                            st.rerun()
                    with col2:
                        if st.button("🗑️ Dismiss", key=f"dismiss_{item['query'][:20]}"):
                            learner.resolve_flag(item['query'], 'dismissed')
                            st.rerun()
        else:
            st.success("No flagged queries!")

        st.divider()

        # Learning Stats
        st.markdown("### 📊 Learning Statistics")
        st.json(learning_stats)


def render_browser_tab():
    """Render the ChromaDB browser."""
    st.header("🗄️ ChromaDB Browser")

    vs = get_vector_store()
    retriever = get_retriever()
    collection = vs._collection

    total = collection.count()
    st.metric("Total Chunks", total)

    st.divider()

    sub_tab1, sub_tab2 = st.tabs(["📋 Browse", "🔍 Search"])

    with sub_tab1:
        col1, col2 = st.columns(2)
        with col1:
            per_page = st.selectbox("Per page", [10, 25, 50], index=0)
        with col2:
            page = st.number_input("Page", min_value=1, max_value=max(1, total // per_page + 1), value=1)

        offset = (page - 1) * per_page

        results = collection.get(
            limit=per_page,
            offset=offset,
            include=["documents", "metadatas"]
        )

        if results['ids']:
            for doc_id, doc, meta in zip(results['ids'], results['documents'], results['metadatas']):
                with st.expander(f"**{meta.get('title', 'Unknown')}** - {meta.get('author', '')}"):
                    st.caption(f"URL: {meta.get('url', 'N/A')}")
                    st.text(doc[:800] + "..." if len(doc) > 800 else doc)

    with sub_tab2:
        query = st.text_input("Search query", key="chroma_search")
        top_k = st.slider("Results", 1, 20, 5, key="chroma_topk")

        if query:
            results = retriever.retrieve_with_scores(query, top_k=top_k)
            for i, r in enumerate(results, 1):
                meta = r.get('metadata', {})
                distance = r.get('distance', 0)
                with st.expander(f"{i}. {meta.get('title', 'Unknown')} (dist: {distance:.3f})"):
                    st.caption(f"Author: {meta.get('author', 'N/A')}")
                    st.text(r.get('text', ''))


def _run_streaming(cmd: list, label: str = "Running") -> bool:
    """
    Run a subprocess and stream its stdout line-by-line into a Streamlit
    text area. No timeout — returns True on success, False on failure.
    Shows a live spinner and updates a single text element in place.
    """
    project_root = str(Path(__file__).parent.parent)
    log_lines: list[str] = []
    log_box = st.empty()

    with st.spinner(f"{label}…"):
        try:
            proc = subprocess.Popen(
                cmd,
                cwd=project_root,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            for line in proc.stdout:
                log_lines.append(line.rstrip())
                # Keep last 40 lines visible so the box doesn't grow forever
                log_box.code("\n".join(log_lines[-40:]), language=None)
            proc.wait()
        except Exception as e:
            st.error(f"{label} error: {e}")
            return False

    if proc.returncode == 0:
        st.success(f"{label} complete!")
        return True
    else:
        st.error(f"{label} failed (exit {proc.returncode}). See output above.")
        return False


def render_settings_tab():
    """Render the settings page."""
    st.header("⚙️ Settings")

    # ML Features
    st.subheader("🤖 ML/AI Features")

    col1, col2 = st.columns(2)

    with col1:
        use_expansion = st.toggle(
            "Query Expansion",
            value=st.session_state.settings["use_expansion"],
            help="Expand queries with related terms"
        )

        use_hybrid = st.toggle(
            "Hybrid Search",
            value=st.session_state.settings["use_hybrid"],
            help="Combine semantic + keyword search"
        )

        use_multi_query = st.toggle(
            "Multi-Query Retrieval",
            value=st.session_state.settings.get("use_multi_query", True),
            help="Generate 3 query variants and union results before reranking (+recall, ~1s overhead)"
        )

        use_contextual_retrieval = st.toggle(
            "Contextual Retrieval (ingest)",
            value=st.session_state.settings.get("use_contextual_retrieval", False),
            help="Prepend LLM-generated context to each chunk at ingest time (Anthropic 2024). Requires full re-ingest to take effect."
        )

    with col2:
        use_hyde = st.toggle(
            "HyDE",
            value=st.session_state.settings["use_hyde"],
            help="Hypothetical Document Embedding (slower but can improve results)"
        )

        use_reranking = st.toggle(
            "Cross-Encoder Reranking",
            value=st.session_state.settings.get("use_reranking", True),
            help="Rerank retrieved chunks with FlashRank cross-encoder (+30–40% accuracy, ~20ms overhead)"
        )

        use_cache = st.toggle(
            "Response Caching",
            value=st.session_state.settings["use_cache"],
            help="Cache responses to reduce LLM calls"
        )

        evaluate_confidence = st.toggle(
            "Auto-Quality Evaluation",
            value=st.session_state.settings.get("evaluate_confidence", False),
            help="Automatically evaluate response confidence (adds ~1-2s per query)"
        )

        show_confidence = st.toggle(
            "Show Confidence Score",
            value=st.session_state.settings.get("show_confidence", True),
            help="Display confidence score below responses"
        )

    st.divider()

    # Retrieval Settings
    st.subheader("🔍 Retrieval Settings")

    col1, col2, col3 = st.columns(3)

    with col1:
        top_k = st.slider("Sources (top_k)", 1, 15, st.session_state.settings["top_k"])

    with col2:
        temperature = st.slider("Temperature", 0.0, 1.0, st.session_state.settings["temperature"])

    with col3:
        cache_ttl = st.slider("Cache TTL (hours)", 1, 72, st.session_state.settings["cache_ttl_hours"])

    st.divider()

    # Save settings button
    if st.button("💾 Save Settings", type="primary"):
        new_settings = {
            "top_k": top_k,
            "temperature": temperature,
            "use_expansion": use_expansion,
            "use_hybrid": use_hybrid,
            "use_hyde": use_hyde,
            "use_reranking": use_reranking,
            "use_multi_query": use_multi_query,
            "use_contextual_retrieval": use_contextual_retrieval,
            "use_cache": use_cache,
            "cache_ttl_hours": cache_ttl,
            "evaluate_confidence": evaluate_confidence,
            "show_confidence": show_confidence
        }
        st.session_state.settings = new_settings
        _save_settings(new_settings)
        st.success("Definições guardadas!")

    st.divider()

    # Knowledge Base Management
    st.subheader("📚 Knowledge Base Management")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Scrape Blogs**")
        st.caption("Buscar novos artigos do blog nearpartner.com")
        if st.button("🌐 Scrape New Posts"):
            with st.spinner("Scraping..."):
                try:
                    result = subprocess.run(
                        [sys.executable, "scraper.py"],
                        cwd=str(Path(__file__).parent.parent),
                        capture_output=True,
                        text=True,
                        timeout=300
                    )
                    if result.returncode == 0:
                        st.success("Scraping complete!")
                        st.text(result.stdout[-1000:] if len(result.stdout) > 1000 else result.stdout)
                    else:
                        st.error(f"Error: {result.stderr[:300]}")
                except subprocess.TimeoutExpired:
                    st.error("Scraping timed out")
                except Exception as e:
                    st.error(f"Error: {e}")

    with col2:
        st.markdown("**Ingest New Content**")
        st.caption("Only embed new/changed content (fast, incremental)")
        if st.button("📥 Ingest Content"):
            ingest_cmd = [sys.executable, "scripts/ingest_blogs.py", "--incremental"]
            if st.session_state.settings.get("use_contextual_retrieval", False):
                ingest_cmd.append("--contextual")
            ok = _run_streaming(ingest_cmd, label="Ingesting")
            if ok:
                get_vector_store.clear()
                get_retriever.clear()

    with col3:
        st.markdown("**Full Update**")
        st.caption("Scrape + incremental ingest in one step")
        if st.button("🔄 Full Update"):
            st.markdown("**Step 1/2 — Scraping**")
            ok = _run_streaming([sys.executable, "scraper.py"], label="Scraping")
            if ok:
                st.markdown("**Step 2/2 — Ingesting**")
                ingest_cmd2 = [sys.executable, "scripts/ingest_blogs.py", "--incremental"]
                if st.session_state.settings.get("use_contextual_retrieval", False):
                    ingest_cmd2.append("--contextual")
                ok2 = _run_streaming(ingest_cmd2, label="Ingesting")
                if ok2:
                    get_vector_store.clear()
                    get_retriever.clear()

    st.divider()
    st.subheader("⚠️ Re-ingest All")
    st.caption("Wipe the vector store and rebuild from scratch. Required after changing embedding model or toggling Contextual Retrieval / Parent-Child Chunking.")
    if st.button("🗑️ Re-ingest All", type="secondary"):
        reingest_cmd = [sys.executable, "scripts/ingest_blogs.py"]
        if st.session_state.settings.get("use_contextual_retrieval", False):
            reingest_cmd.append("--contextual")
        ok = _run_streaming(reingest_cmd, label="Re-ingesting")
        if ok:
            get_vector_store.clear()
            get_retriever.clear()

    st.divider()

    # Cache Management
    st.subheader("🗄️ Cache Management")

    cache = get_response_cache()
    cache_stats = cache.get_stats()

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Cached Responses", cache_stats["total_entries"])

    with col2:
        st.metric("Total Cache Hits", cache_stats["total_hits"])

    with col3:
        if st.button("🗑️ Clear Cache"):
            cache.clear()
            st.success("Cache cleared!")
            st.rerun()

    st.divider()

    st.divider()

    # Inference Status
    st.subheader("⚙️ Inference Status")

    from src.config import config
    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"**LLM server:** `{config.llm_base_url}`")
        st.markdown(f"**LLM model:** `{config.llm_model}`")
        st.markdown(f"**Embedding model:** `{config.embedding_model}` *(in-process)*")

    with col2:
        try:
            from openai import OpenAI
            client = OpenAI(base_url=config.llm_base_url, api_key="not-needed")
            models = client.models.list()
            model_ids = [m.id for m in models.data]
            st.success(f"✅ LLM server connected — {len(model_ids)} model(s) loaded")
            if model_ids:
                st.caption("Loaded: " + ", ".join(model_ids))
        except Exception as e:
            st.error("❌ LLM server not reachable")
            st.caption("Run start_llm.ps1 (or: llama-server.exe -m Qwen2.5-7B... --port 8080)")

    st.divider()

    # Current Settings Display
    with st.expander("📋 Current Settings"):
        st.json(st.session_state.settings)


def main():
    st.title("🤖 Near Partner Chatbot")

    # Main tabs
    tab_chat, tab_analytics, tab_browser, tab_settings = st.tabs([
        "💬 Chat",
        "📊 Analytics",
        "🗄️ ChromaDB",
        "⚙️ Settings"
    ])

    with tab_chat:
        render_chat_tab()

    with tab_analytics:
        render_analytics_tab()

    with tab_browser:
        render_browser_tab()

    with tab_settings:
        render_settings_tab()


if __name__ == "__main__":
    main()
