from __future__ import annotations


class SearchTool:
    """
    Search tool adapter.

    This project has not wired a real web search backend yet, so the tool
    returns an explicit integration reminder instead of synthetic fallback data.
    """

    def search(self, query: str) -> str:
        normalized = query.strip()
        if not normalized:
            return (
                "Search tool was selected, but the search query is empty. "
                "Provide a real search query or connect a real search backend."
            )
        return (
            "Search tool backend is not integrated yet. "
            "Please connect a real web search provider before relying on live search. "
            f"Requested query: {normalized}"
        )
