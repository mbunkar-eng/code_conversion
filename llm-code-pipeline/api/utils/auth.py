"""
Authentication utilities for API access control.
"""

import os
import hashlib
import secrets
from typing import Optional
from functools import wraps

from fastapi import HTTPException, Header, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials


# Security scheme for OpenAPI docs
security = HTTPBearer(auto_error=False)


class APIKeyManager:
    """Manages API key validation and generation."""

    def __init__(self):
        self._valid_keys: set[str] = set()
        self._load_keys_from_env()

    def _load_keys_from_env(self) -> None:
        """Load API keys from environment variables."""
        # Load single key
        single_key = os.environ.get("LLM_API_KEY")
        if single_key:
            self._valid_keys.add(single_key)

        # Load multiple keys (comma-separated)
        multi_keys = os.environ.get("LLM_API_KEYS", "")
        for key in multi_keys.split(","):
            key = key.strip()
            if key:
                self._valid_keys.add(key)

        # Load from file if specified
        keys_file = os.environ.get("LLM_API_KEYS_FILE")
        if keys_file and os.path.exists(keys_file):
            with open(keys_file, "r") as f:
                for line in f:
                    key = line.strip()
                    if key and not key.startswith("#"):
                        self._valid_keys.add(key)

    def is_valid_key(self, key: str) -> bool:
        """Check if an API key is valid."""
        if not self._valid_keys:
            # No keys configured = no authentication required
            return True
        return key in self._valid_keys

    def add_key(self, key: str) -> None:
        """Add a valid API key."""
        self._valid_keys.add(key)

    def remove_key(self, key: str) -> bool:
        """Remove an API key."""
        if key in self._valid_keys:
            self._valid_keys.discard(key)
            return True
        return False

    def generate_key(self, prefix: str = "llm") -> str:
        """Generate a new API key."""
        random_part = secrets.token_hex(24)
        key = f"{prefix}-{random_part}"
        self._valid_keys.add(key)
        return key

    @property
    def auth_required(self) -> bool:
        """Check if authentication is required."""
        return len(self._valid_keys) > 0


# Global key manager instance
key_manager = APIKeyManager()


def get_api_key_from_header(
    authorization: Optional[str] = Header(None, alias="Authorization"),
    x_api_key: Optional[str] = Header(None, alias="X-API-Key")
) -> Optional[str]:
    """
    Extract API key from request headers.

    Supports:
    - Authorization: Bearer <key>
    - X-API-Key: <key>
    """
    if authorization:
        parts = authorization.split()
        if len(parts) == 2 and parts[0].lower() == "bearer":
            return parts[1]

    if x_api_key:
        return x_api_key

    return None


async def verify_api_key(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
    x_api_key: Optional[str] = Header(None, alias="X-API-Key")
) -> Optional[str]:
    """
    Verify API key from request.

    FastAPI dependency that validates the API key.
    Raises HTTPException if authentication fails when required.
    """
    # If no authentication configured, allow all requests
    if not key_manager.auth_required:
        return None

    # Try to get key from different sources
    key = None

    if credentials and credentials.credentials:
        key = credentials.credentials
    elif x_api_key:
        key = x_api_key

    if not key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "error": {
                    "message": "Missing API key. Include 'Authorization: Bearer <key>' header.",
                    "type": "invalid_request_error",
                    "code": "missing_api_key"
                }
            }
        )

    if not key_manager.is_valid_key(key):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "error": {
                    "message": "Invalid API key provided.",
                    "type": "invalid_request_error",
                    "code": "invalid_api_key"
                }
            }
        )

    return key


def require_api_key(func):
    """Decorator to require API key for a route."""
    @wraps(func)
    async def wrapper(*args, api_key: str = Depends(verify_api_key), **kwargs):
        return await func(*args, **kwargs)
    return wrapper


def hash_api_key(key: str) -> str:
    """Hash an API key for secure storage/logging."""
    return hashlib.sha256(key.encode()).hexdigest()[:16]


def mask_api_key(key: str) -> str:
    """Mask an API key for display (show first/last 4 chars)."""
    if len(key) <= 8:
        return "*" * len(key)
    return f"{key[:4]}...{key[-4:]}"
