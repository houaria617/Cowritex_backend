"""
orchestrator/nodes/_llm.py
───────────────────────────
Builds the correct LangChain chat model based on the project's llm_provider
preference (groq | gemini).  Used by intent_classifier and chat nodes.
"""

from __future__ import annotations

import os


def get_llm(provider: str = "groq", temperature: float = 0.3):
    """
    Return a LangChain chat model for the given provider.
    The caller must have the matching env var set:
        GROQ_API_KEY   for groq
        GOOGLE_API_KEY for gemini
    """
    provider = (provider or "groq").lower()

    if provider == "groq":
        from langchain_groq import ChatGroq
        return ChatGroq(
            model=os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile"),
            temperature=temperature,
            api_key=os.environ["GROQ_API_KEY"],
        )

    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model=os.environ.get("GEMINI_MODEL", "gemini-2.0-flash"),
            temperature=temperature,
            # .get() not [] to avoid KeyError
            google_api_key=os.environ.get("GOOGLE_API_KEY"),
        )
    raise ValueError(
        f"Unknown llm_provider: {provider!r}. Choose 'groq' or 'gemini'.")
