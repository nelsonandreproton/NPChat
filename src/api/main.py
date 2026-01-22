"""
FastAPI application for the Near Partner chatbot.
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .routes import router
from ..config import config

app = FastAPI(
    title="Near Partner Chatbot API",
    description="RAG-based chatbot API for Near Partner knowledge base",
    version="1.0.0"
)

# CORS middleware for frontend access
# WARNING: In production, replace "*" with specific allowed origins
# Example: allow_origins=["https://yourdomain.com", "http://localhost:8501"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501", "http://127.0.0.1:8501"],  # Streamlit default ports
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Include routes
app.include_router(router, prefix="/api/v1", tags=["chatbot"])


@app.get("/")
async def root():
    """Root endpoint with API info."""
    return {
        "name": "Near Partner Chatbot API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/api/v1/health"
    }


def start_server():
    """Start the API server."""
    import uvicorn
    uvicorn.run(
        "src.api.main:app",
        host=config.api_host,
        port=config.api_port,
        reload=True
    )


if __name__ == "__main__":
    start_server()
