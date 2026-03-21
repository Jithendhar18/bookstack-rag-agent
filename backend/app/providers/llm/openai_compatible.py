"""OpenAI-compatible LLM provider — works for OpenAI, OpenRouter, and Groq."""

import logging
from typing import List, Dict, Any, AsyncIterator

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

from app.providers.base import BaseLLM

logger = logging.getLogger(__name__)


def _to_langchain_messages(messages: List[Dict[str, str]]):
    """Convert dicts to LangChain message objects."""
    lc_msgs = []
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        if role == "system":
            lc_msgs.append(SystemMessage(content=content))
        elif role == "assistant":
            lc_msgs.append(AIMessage(content=content))
        else:
            lc_msgs.append(HumanMessage(content=content))
    return lc_msgs


class OpenAICompatibleLLM(BaseLLM):
    """LLM provider for any OpenAI-compatible API (OpenAI, OpenRouter, Groq)."""

    def __init__(
        self,
        model: str,
        api_key: str,
        base_url: str,
        temperature: float = 0.1,
        max_tokens: int = 2048,
        provider_name: str = "openai",
    ):
        self._model = model
        self._provider = provider_name
        self._client = ChatOpenAI(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=api_key,
            base_url=base_url,
        )
        logger.info(f"LLM initialized: provider={provider_name}, model={model}, base_url={base_url}")

    async def generate(self, messages: List[Dict[str, str]], **kwargs) -> str:
        lc_messages = _to_langchain_messages(messages)
        response = await self._client.ainvoke(lc_messages, **kwargs)
        return response.content

    async def stream(self, messages: List[Dict[str, str]], **kwargs) -> AsyncIterator[str]:
        lc_messages = _to_langchain_messages(messages)
        async for chunk in self._client.astream(lc_messages, **kwargs):
            if chunk.content:
                yield chunk.content

    @property
    def model_name(self) -> str:
        return f"{self._provider}/{self._model}"

    @property
    def langchain_client(self) -> ChatOpenAI:
        """Expose the underlying LangChain client for nodes that need it directly."""
        return self._client
