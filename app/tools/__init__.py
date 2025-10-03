"""Tool factories for Chatty Genie."""

from .web_search import create_disabled_web_search_tool, create_ollama_web_search_tool

__all__ = ["create_ollama_web_search_tool", "create_disabled_web_search_tool"]
