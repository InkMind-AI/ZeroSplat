"""Require explicit CLI values (no implicit paths or API credentials)."""

from __future__ import annotations


def require_nonempty(value: str | None, name: str) -> str:
    if value is None:
        raise SystemExit(f"Missing required {name}.")
    if isinstance(value, str) and not value.strip():
        raise SystemExit(f"Missing required {name}.")
    return value


def require_api_key(value: str | None) -> str:
    if value is None or (isinstance(value, str) and not value.strip()):
        raise SystemExit(
            "Missing required --api_key (use EMPTY for local OpenAI-compatible servers)."
        )
    return value


def require_vlm_endpoint(api_key: str | None, base_url: str | None, model_name: str | None) -> None:
    require_api_key(api_key)
    require_nonempty(base_url, "--base_url")
    require_nonempty(model_name, "--model_name")
