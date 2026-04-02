"""Placeholder for Agent class - to be implemented in M1."""

from __future__ import annotations

from isotopes_core.types import Context
from isotopes_core.tools import Tool
from isotopes_core.providers.base import Provider


class Agent:
    """Stateful agent wrapping the agent loop.
    
    TODO: Implement in M1 milestone.
    """

    def __init__(
        self,
        provider: Provider | None = None,
        system_prompt: str = "",
        tools: list[Tool] | None = None,
        max_turns: int | None = None,
    ) -> None:
        self.provider = provider
        self.system_prompt = system_prompt
        self.tools = tools or []
        self.max_turns = max_turns
        self._context = Context(system_prompt=system_prompt)

    async def prompt(self, message: str):
        """Send a message and stream events.
        
        TODO: Implement in M1.
        """
        raise NotImplementedError("Agent.prompt() not yet implemented")
        yield  # Make this a generator
