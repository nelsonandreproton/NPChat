"""
FastAPI application for the Near Partner chatbot.
"""
import time
from collections import defaultdict
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from .routes import router
from ..config import config

app = FastAPI(
    title="Near Partner Chatbot API",
    description="RAG-based chatbot API for Near Partner knowledge base",
    version="1.0.0"
)

# In-memory rate limiter (per IP): max 30 requests per minute for chat endpoints
_rate_limit_store: dict = defaultdict(list)
RATE_LIMIT_REQUESTS = 30
RATE_LIMIT_WINDOW = 60  # seconds


@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    """Simple in-memory rate limiter for chat endpoints."""
    if request.url.path.startswith("/api/v1/chat"):
        client_ip = request.client.host if request.client else "unknown"
        now = time.time()
        window_start = now - RATE_LIMIT_WINDOW

        # Clean old entries
        _rate_limit_store[client_ip] = [
            t for t in _rate_limit_store[client_ip] if t > window_start
        ]

        if len(_rate_limit_store[client_ip]) >= RATE_LIMIT_REQUESTS:
            return JSONResponse(
                status_code=429,
                content={"detail": "Too many requests. Please wait before sending more."}
            )

        _rate_limit_store[client_ip].append(now)

    return await call_next(request)


# CORS middleware for frontend access
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
