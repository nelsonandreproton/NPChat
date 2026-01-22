"""
Simple ChromaDB browser using Streamlit.
"""
import streamlit as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.retrieval.vector_store import VectorStore
from src.retrieval.retriever import Retriever

st.set_page_config(page_title="ChromaDB Browser", page_icon="🗄️", layout="wide")

st.title("🗄️ ChromaDB Browser")

@st.cache_resource
def get_vector_store():
    return VectorStore()

@st.cache_resource
def get_retriever():
    return Retriever()

vs = get_vector_store()
retriever = get_retriever()
collection = vs._collection

# Stats
total = collection.count()
st.metric("Total Chunks", total)

st.divider()

# Tabs
tab1, tab2 = st.tabs(["📋 Browse", "🔍 Search"])

with tab1:
    st.subheader("Browse Chunks")

    # Pagination
    per_page = st.selectbox("Items per page", [10, 25, 50, 100], index=0)
    page = st.number_input("Page", min_value=1, max_value=max(1, total // per_page + 1), value=1)

    offset = (page - 1) * per_page

    # Fetch data
    results = collection.get(
        limit=per_page,
        offset=offset,
        include=["documents", "metadatas"]
    )

    for i, (doc_id, doc, meta) in enumerate(zip(results['ids'], results['documents'], results['metadatas'])):
        with st.expander(f"**{meta.get('title', 'Unknown')}** - {meta.get('author', 'Unknown')}"):
            st.markdown(f"**URL:** {meta.get('url', 'N/A')}")
            st.markdown(f"**Chunk:** {meta.get('chunk_index', 'N/A')}")
            st.markdown(f"**ID:** `{doc_id}`")
            st.divider()
            st.text(doc)

with tab2:
    st.subheader("Semantic Search")

    query = st.text_input("Search query")
    top_k = st.slider("Results", 1, 20, 5)

    if query:
        results = retriever.retrieve_with_scores(query, top_k=top_k)

        for i, r in enumerate(results, 1):
            meta = r.get('metadata', {})
            distance = r.get('distance', 0)
            with st.expander(f"**{i}. {meta.get('title', 'Unknown')}** (distance: {distance:.3f})"):
                st.markdown(f"**Author:** {meta.get('author', 'N/A')}")
                st.markdown(f"**URL:** {meta.get('url', 'N/A')}")
                st.divider()
                st.text(r.get('text', ''))
