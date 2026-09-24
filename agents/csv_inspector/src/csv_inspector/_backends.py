"""LLM backend selection for the csv_inspector agent.

Kept free of third-party imports so the backend choice can be expressed
everywhere (library API, CLIs, settings) without pulling in any SDK.
"""

from __future__ import annotations

from enum import Enum


class LLMBackend(str, Enum):
    """The LLM backend used to run an inspection.

    Attributes:
        LOCAL: A local Ollama model. Free, needs no credentials; the default.
        API: A Google Gemini model, through the Gemini Developer API (API
            key) or Vertex AI (Application Default Credentials). Opt-in;
            requires the ``[cloud]`` extra.
    """

    LOCAL = "local"
    API = "api"
