"""BookStack API client for pulling content."""

import logging
from typing import Optional

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class BookStackClient:
    """Async client for BookStack REST API."""

    def __init__(self):
        self.base_url = settings.BOOKSTACK_BASE_URL.rstrip("/")
        self.headers = {
            "Authorization": f"Token {settings.BOOKSTACK_TOKEN_ID}:{settings.BOOKSTACK_TOKEN_SECRET}",
            "Content-Type": "application/json",
        }
        self._client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                headers=self.headers,
                timeout=30.0,
            )
        return self._client

    async def close(self):
        if self._client and not self._client.is_closed:
            await self._client.aclose()

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    async def _get(self, endpoint: str, params: Optional[dict] = None) -> dict:
        client = await self._get_client()
        response = await client.get(f"/api/{endpoint}", params=params)
        response.raise_for_status()
        return response.json()

    async def list_pages(self, offset: int = 0, count: int = 100) -> dict:
        return await self._get("pages", params={"offset": offset, "count": count})

    async def get_page(self, page_id: int) -> dict:
        return await self._get(f"pages/{page_id}")

    async def list_books(self, offset: int = 0, count: int = 100) -> dict:
        return await self._get("books", params={"offset": offset, "count": count})

    async def get_book(self, book_id: int) -> dict:
        return await self._get(f"books/{book_id}")

    async def list_chapters(self, offset: int = 0, count: int = 100) -> dict:
        return await self._get("chapters", params={"offset": offset, "count": count})

    async def get_chapter(self, chapter_id: int) -> dict:
        return await self._get(f"chapters/{chapter_id}")

    async def list_shelves(self, offset: int = 0, count: int = 100) -> dict:
        return await self._get("shelves", params={"offset": offset, "count": count})

    async def get_all_pages(self) -> list[dict]:
        """Paginate through all pages."""
        all_pages = []
        offset = 0
        while True:
            result = await self.list_pages(offset=offset, count=100)
            data = result.get("data", [])
            if not data:
                break
            all_pages.extend(data)
            if len(data) < 100:
                break
            offset += 100
        return all_pages
