#!/bin/bash
# Setup script for Ollama models

echo "=================================================="
echo "Setting up Ollama models for Near Partner Chatbot"
echo "=================================================="

# Check if Ollama is installed
if ! command -v ollama &> /dev/null; then
    echo "Ollama is not installed!"
    echo "Please install from: https://ollama.ai"
    exit 1
fi

echo ""
echo "Pulling embedding model (nomic-embed-text)..."
ollama pull nomic-embed-text

echo ""
echo "Pulling LLM model (mistral:7b)..."
echo "This may take a while depending on your connection..."
ollama pull mistral:7b

echo ""
echo "=================================================="
echo "Setup complete!"
echo ""
echo "Models installed:"
ollama list
echo ""
echo "To start using the chatbot:"
echo "  1. Run: python scripts/ingest_blogs.py"
echo "  2. Run: streamlit run app/streamlit_app.py"
echo "=================================================="
