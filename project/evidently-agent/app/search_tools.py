"""
search_tools.py
---------------
Search tool that wraps the minsearch Index for use with the Pydantic AI agent.
"""

from typing import Any

from minsearch import Index


class SearchTool:
    """Wraps a minsearch Index and exposes a search method for Pydantic AI."""

    def __init__(self, index: Index, num_results: int = 5) -> None:
        self.index = index
        self.num_results = num_results

    def search(self, query: str) -> list[dict[str, Any]]:
        """
        Search the Evidently AI documentation index.

        Performs a full-text search over documentation chunks.
        Use this whenever you need to look up information about Evidently AI
        features, APIs, concepts, or usage examples.

        Args:
            query: Natural language search query.

        Returns:
            List of up to 5 matching document chunks with metadata.
        """
        results = self.index.search(query, num_results=self.num_results)
        # Return only the fields useful for the LLM to avoid bloating the context
        clean = []
        for r in results:
            clean.append(
                {
                    "filename": r.get("filename", ""),
                    "title": r.get("title", ""),
                    "content": r.get("content", "")[:1500],  # cap per chunk
                }
            )
        return clean
