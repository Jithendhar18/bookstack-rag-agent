"""Ollama LLM provider — local inference via Ollama API."""

import logging
from typing import List, Dict, Any, AsyncIterator

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

from app.providers.base import BaseLLM

logger = logging.getLogger(__name__)


def _to_langchain_messages(messages: List[Dict[str, str]]):
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


class OllamaLLM(BaseLLM):
    """Local LLM via Ollama (uses OpenAI-compatible API internally)."""

    def __init__(
        self,
        model: str = "llama3",
        base_url: str = "http://localhost:11434/v1",
        temperature: float = 0.1,
        max_tokens: int = 2048,
    ):
        self._model = model
        self._client = ChatOpenAI(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            api_key="not-needed",
            base_url=base_url,
        )
        logger.info(f"Ollama LLM initialized: model={model}, base_url={base_url}")

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
        return f"ollama/{self._model}"

    @property
    def langchain_client(self) -> ChatOpenAI:
        return self._client
