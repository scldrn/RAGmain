from __future__ import annotations


class AgenticRagError(Exception):
    """Base exception for all repository-specific runtime failures."""


class ConfigurationError(AgenticRagError):
    """Raised when runtime configuration is invalid or incomplete."""


class DocumentLoadError(AgenticRagError):
    """Raised when source documents cannot be loaded or preprocessed."""


class VectorStoreError(AgenticRagError):
    """Raised when the local vector index cannot be created, loaded, or written."""


class LLMError(AgenticRagError):
    """Raised when model or graph execution fails."""
