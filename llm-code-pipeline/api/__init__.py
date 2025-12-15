"""
API module for LLM Code Pipeline.
OpenAI-compatible REST API server.
"""

from .main import app, create_app

__all__ = ["app", "create_app"]
