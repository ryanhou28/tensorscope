"""FastAPI application entry point for Tensorscope."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="Tensorscope",
    description="Linear algebra visualization API",
    version="0.1.0",
)

# CORS middleware for frontend development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Vite dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "Tensorscope API", "version": "0.1.0"}


@app.get("/api/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok"}
