from typing import Protocol, Dict, Any


class Provider(Protocol):
    """Protocol every provider adapter should implement."""

    def chat(self, messages: list[Dict[str, Any]], config: Dict[str, Any]) -> str:
        """Send chat messages and return string content (model reply)."""
        ...

    async def achat(self, messages: list[Dict[str, Any]], config: Dict[str, Any]) -> str:
        """Asynchronously send chat messages and return string content."""
        ...
