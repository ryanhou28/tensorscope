"""FastAPI application entry point for Tensorscope."""

from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware

from .routes import router
from .state import state
from .websocket import handle_websocket

# Import scenarios for registration
from ..scenarios import least_squares_2d


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler for startup/shutdown."""
    # Startup: Register scenarios
    state.register_scenario(least_squares_2d)

    yield

    # Shutdown: Clean up if needed
    pass


app = FastAPI(
    title="Tensorscope",
    description="Linear algebra visualization API",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS middleware for frontend development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Vite dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include REST API routes
app.include_router(router)


@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "Tensorscope API", "version": "0.1.0"}


@app.get("/api/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok"}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time updates."""
    await handle_websocket(websocket)
