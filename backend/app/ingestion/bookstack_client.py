"""BookStack API client for pulling content."""

import logging
import time
from typing import Optional

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential, before_sleep_log

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

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(min=1, max=10),
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )
    async def _get(self, endpoint: str, params: Optional[dict] = None) -> dict:
        client = await self._get_client()
        url = f"/api/{endpoint}"
        logger.debug("BookStack API request", extra={
            "stage": "fetch",
            "url": f"{self.base_url}{url}",
            "params": params,
        })
        t0 = time.monotonic()
        try:
            response = await client.get(url, params=params)
            elapsed = round(time.monotonic() - t0, 3)
            logger.debug("BookStack API response", extra={
                "stage": "fetch",
                "url": f"{self.base_url}{url}",
                "status_code": response.status_code,
                "elapsed_s": elapsed,
            })
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as exc:
            logger.error("BookStack API HTTP error", exc_info=True, extra={
                "stage": "fetch",
                "url": f"{self.base_url}{url}",
                "status_code": exc.response.status_code,
                "response_body": exc.response.text[:500],
            })
            raise
        except httpx.RequestError as exc:
            logger.error("BookStack API request error", exc_info=True, extra={
                "stage": "fetch",
                "url": f"{self.base_url}{url}",
                "error": str(exc),
            })
            raise

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

    def resolve_page_url(self, page: dict) -> str:
        """Resolve the BookStack page URL using a priority-based fallback.

        Priority:
          1. Direct ``url`` field returned by the API
          2. Construct from book_slug + page_slug: /books/{book_slug}/pages/{slug}
          3. Generic link fallback: /link/{page_id}
        """
        base = self.base_url

        direct = page.get("url")
        if direct:
            return direct.rstrip("/")

        book_slug = page.get("book_slug", "")
        page_slug = page.get("slug", "")
        if book_slug and page_slug:
            return f"{base}/books/{book_slug}/pages/{page_slug}"

        page_id = page.get("id")
        return f"{base}/link/{page_id}"

    async def get_all_pages(self) -> list[dict]:
        """Paginate through all pages."""
        all_pages = []
        offset = 0
        while True:
            logger.debug("Fetching page list batch", extra={
                "stage": "fetch", "offset": offset, "count": 100,
            })
            result = await self.list_pages(offset=offset, count=100)
            data = result.get("data", [])
            if not data:
                break
            all_pages.extend(data)
            logger.debug("Page list batch received", extra={
                "stage": "fetch",
                "batch_size": len(data),
                "total_so_far": len(all_pages),
            })
            if len(data) < 100:
                break
            offset += 100

        logger.info("All BookStack pages fetched", extra={
            "stage": "fetch",
            "total_pages": len(all_pages),
        })
        return all_pages

    async def get_all_books(self) -> list[dict]:
        """Paginate through all books and return a flat list."""
        all_books: list[dict] = []
        offset = 0
        while True:
            result = await self.list_books(offset=offset, count=100)
            data = result.get("data", [])
            if not data:
                break
            all_books.extend(data)
            if len(data) < 100:
                break
            offset += 100
        logger.info("All BookStack books fetched", extra={
            "stage": "fetch", "total_books": len(all_books),
        })
        return all_books

    async def get_all_chapters(self) -> list[dict]:
        """Paginate through all chapters and return a flat list."""
        all_chapters: list[dict] = []
        offset = 0
        while True:
            result = await self.list_chapters(offset=offset, count=100)
            data = result.get("data", [])
            if not data:
                break
            all_chapters.extend(data)
            if len(data) < 100:
                break
            offset += 100
        logger.info("All BookStack chapters fetched", extra={
            "stage": "fetch", "total_chapters": len(all_chapters),
        })
        return all_chapters

    async def build_hierarchy_caches(self) -> tuple[dict, dict, dict, dict]:
        """Walk all books + their contents to build four lookup caches in one pass.

        The Book detail response (GET /api/books/{id}) contains a ``contents``
        array that includes:
          - Direct pages:  {type:"page", id, url, slug, book_id, ...}
          - Chapters:      {type:"chapter", id, name, url, pages:[{id, url, ...}]}

        Pages inside the ``contents`` array carry the **canonical URL** that the
        page-detail endpoint (GET /api/pages/{id}) does NOT return.  We cache
        these URLs so that resolve_page_url always hits its highest-priority
        branch (direct ``url`` field) rather than falling back to /link/{id}.

        Returns:
            book_name_cache     {book_id  -> book_name}
            book_slug_cache     {book_id  -> book_slug}
            page_url_cache      {page_id  -> canonical_url}
            chapter_name_cache  {chapter_id -> chapter_name}
        """
        all_books = await self.get_all_books()
        book_name_cache: dict[int, str] = {}
        book_slug_cache: dict[int, str] = {}
        page_url_cache: dict[int, str] = {}
        chapter_name_cache: dict[int, str] = {}

        for book_summary in all_books:
            bid = book_summary["id"]
            book_name_cache[bid] = book_summary.get("name", "")
            book_slug_cache[bid] = book_summary.get("slug", "")

            try:
                book_detail = await self.get_book(bid)
                for item in book_detail.get("contents", []):
                    item_type = item.get("type")
                    if item_type == "page":
                        pid = item["id"]
                        if item.get("url"):
                            page_url_cache[pid] = item["url"]
                    elif item_type == "chapter":
                        ch_id = item["id"]
                        chapter_name_cache[ch_id] = item.get("name", "")
                        for page in item.get("pages", []):
                            pid = page["id"]
                            if page.get("url"):
                                page_url_cache[pid] = page["url"]
            except Exception as exc:
                logger.warning(
                    "Could not fetch book detail for hierarchy caches",
                    extra={"stage": "fetch", "book_id": bid, "error": str(exc)},
                )

        logger.info("Hierarchy caches built", extra={
            "stage": "fetch",
            "books": len(book_name_cache),
            "chapters": len(chapter_name_cache),
            "pages_with_url": len(page_url_cache),
        })
        return book_name_cache, book_slug_cache, page_url_cache, chapter_name_cache

