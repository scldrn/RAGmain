from __future__ import annotations

import os

import pytest

from agentic_rag.app import ensure_google_api_key


def test_ensure_google_api_key_accepts_google_api_key(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "google-key")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    ensure_google_api_key()


def test_ensure_google_api_key_promotes_gemini_api_key(monkeypatch):
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", "gemini-key")

    ensure_google_api_key()

    assert "gemini-key" == os.environ["GOOGLE_API_KEY"]


def test_ensure_google_api_key_raises_when_missing(monkeypatch):
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    with pytest.raises(RuntimeError):
        ensure_google_api_key()
