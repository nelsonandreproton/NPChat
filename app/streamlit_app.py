"""
Streamlit demo UI for the Near Partner chatbot.
"""
import streamlit as st
import sys
import time
import threading
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.generation.enhanced_rag_chain import EnhancedRAGChain
from src.retrieval.vector_store import VectorStore
from src.feedback.store import FeedbackStore
from src.feedback.models import Feedback, FeedbackType


# Page config
st.set_page_config(
    page_title="Near Partner Chatbot",
    page_icon="🤖",
    layout="wide"
)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None


@st.cache_resource
def get_rag_chain(use_expansion: bool = True, use_hybrid: bool = True):
    """Initialize enhanced RAG chain (cached)."""
    return EnhancedRAGChain(
        use_query_expansion=use_expansion,
        use_hybrid_search=use_hybrid,
        use_logging=True
    )


@st.cache_resource
def get_vector_store():
    """Initialize vector store (cached)."""
    return VectorStore()


@st.cache_resource
def get_feedback_store():
    """Initialize feedback store (cached)."""
    return FeedbackStore()


def save_feedback(query: str, response: str, feedback_type: FeedbackType, correction: str = None, log_id: int = None):
    """Save user feedback."""
    try:
        store = get_feedback_store()
        feedback = Feedback(
            id="",
            query=query,
            response=response,
            feedback_type=feedback_type,
            correction=correction
        )
        store.add(feedback)

        # Also update analytics log
        if log_id:
            try:
                from src.analytics.query_logger import QueryLogger
                logger = QueryLogger()
                feedback_str = "positive" if feedback_type == FeedbackType.THUMBS_UP else "negative"
                logger.update_feedback(log_id, feedback_str)
            except Exception:
                pass

        return True
    except Exception:
        return False


def main():
    st.title("🤖 Near Partner Chatbot")
    st.markdown("Ask me anything about Near Partner's services, technology insights, and expertise!")

    # Sidebar with info
    with st.sidebar:
        st.header("About")
        st.markdown("""
        This chatbot answers questions based on Near Partner's blog posts.

        **Capabilities:**
        - Technology consulting insights
        - Digital transformation topics
        - AI and software development
        - Business strategy advice
        """)

        st.divider()

        # Show knowledge base stats
        try:
            vs = get_vector_store()
            chunk_count = vs.count()
            st.metric("Knowledge Base", f"{chunk_count} chunks")
        except Exception as e:
            st.warning(f"Could not connect to knowledge base: {e}")

        st.divider()

        # Settings
        st.header("Settings")
        top_k = st.slider("Number of sources", 1, 10, 5)
        temperature = st.slider("Creativity", 0.0, 1.0, 0.7)

        st.divider()

        # ML/AI Features
        st.header("ML Features")
        use_expansion = st.toggle("Query Expansion", value=True, help="Expand queries with related terms")
        use_hybrid = st.toggle("Hybrid Search", value=True, help="Combine semantic + keyword search")
        use_hyde = st.toggle("HyDE", value=False, help="Hypothetical Document Embedding")

        st.divider()

        if st.button("Clear Chat"):
            st.session_state.messages = []
            st.rerun()

        if st.button("📊 View Analytics"):
            st.switch_page("pages/analytics.py") if Path("pages/analytics.py").exists() else st.info("Run: streamlit run app/analytics_dashboard.py")

    # Display chat history
    for i, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            # Show sources and feedback for assistant messages
            if message["role"] == "assistant":
                if "sources" in message:
                    with st.expander("📚 Sources"):
                        for source in message["sources"]:
                            st.markdown(f"- [{source['title']}]({source['url']}) by {source['author']}")

                # Feedback buttons
                col1, col2, col3 = st.columns([1, 1, 6])
                feedback_key = f"feedback_{i}"

                # Get the original query (previous message)
                original_query = st.session_state.messages[i-1]["content"] if i > 0 else ""

                with col1:
                    if st.button("👍", key=f"up_{i}", help="Good response"):
                        log_id = message.get("log_id")
                        if save_feedback(original_query, message["content"], FeedbackType.THUMBS_UP, log_id=log_id):
                            st.toast("Thanks for the feedback!")

                with col2:
                    if st.button("👎", key=f"down_{i}", help="Could be better"):
                        log_id = message.get("log_id")
                        if save_feedback(original_query, message["content"], FeedbackType.THUMBS_DOWN, log_id=log_id):
                            st.toast("Thanks for the feedback! We'll improve.")

    # Chat input
    if prompt := st.chat_input("Ask a question about Near Partner..."):
        # Add user message to history
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate response
        with st.chat_message("assistant"):
            # Create placeholder for thinking message and response
            thinking_placeholder = st.empty()
            response_placeholder = st.empty()

            try:
                rag_chain = get_rag_chain(use_expansion=use_expansion, use_hybrid=use_hybrid)

                # Variables to store result from thread
                result_container = {"result": None, "error": None, "done": False}

                def run_query():
                    try:
                        result_container["result"] = rag_chain.query(
                            question=prompt,
                            top_k=top_k,
                            temperature=temperature,
                            use_hyde=use_hyde
                        )
                    except Exception as e:
                        result_container["error"] = e
                    finally:
                        result_container["done"] = True

                # Start query in background thread
                query_thread = threading.Thread(target=run_query)
                query_thread.start()

                # Show timer while waiting
                start_time = time.time()
                while not result_container["done"]:
                    elapsed = int(time.time() - start_time)
                    thinking_placeholder.markdown(f"⏳ *Thinking for {elapsed} seconds...*")
                    time.sleep(0.5)

                # Clear thinking message
                thinking_placeholder.empty()

                # Check for errors
                if result_container["error"]:
                    raise result_container["error"]

                result = result_container["result"]
                elapsed_total = round(time.time() - start_time, 1)

                # Display response
                response_placeholder.markdown(result.answer)

                # Show timing breakdown
                if hasattr(result, 'timings') and result.timings:
                    t = result.timings
                    st.caption(
                        f"⏱️ Retrieval: {t.get('retrieval', 0)}s | "
                        f"LLM: {t.get('llm_generation', 0)}s | "
                        f"Total: {t.get('total', elapsed_total)}s"
                    )
                else:
                    st.caption(f"Response generated in {elapsed_total}s")

                # Display sources
                if result.sources:
                    with st.expander("📚 Sources"):
                        for source in result.sources:
                            st.markdown(f"- [{source['title']}]({source['url']}) by {source['author']}")

                # Show expanded query if used
                if hasattr(result, 'expanded_query') and result.expanded_query and result.expanded_query != prompt:
                    with st.expander("🔍 Query Expansion"):
                        st.text(result.expanded_query[:500])

                # Add to history (include log_id for feedback)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": result.answer,
                    "sources": result.sources,
                    "log_id": getattr(result, 'log_id', None)
                })

            except Exception as e:
                thinking_placeholder.empty()
                error_msg = f"Error: {str(e)}"
                st.error(error_msg)

                # Check common issues
                if "connection" in str(e).lower() or "ollama" in str(e).lower():
                    st.warning("Make sure Ollama is running: `ollama serve`")
                elif "collection" in str(e).lower() or "empty" in str(e).lower():
                    st.warning("Knowledge base may be empty. Run: `python scripts/ingest_blogs.py`")

    # Footer
    st.divider()
    st.caption("Powered by Near Partner's blog content | Built with Ollama, ChromaDB, and Streamlit")


if __name__ == "__main__":
    main()
