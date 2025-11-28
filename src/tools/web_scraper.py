"""
Robust web scraper with persistent browser context.
Handles JavaScript-rendered pages with proper timeout and error recovery.
"""

import asyncio
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from langchain_core.tools import tool
from playwright.async_api import async_playwright, Browser, Page, Playwright

from ..config import get_settings
from ..logging_config import get_logger

logger = get_logger(__name__)


class BrowserManager:
    """
    Manages a persistent browser instance for efficient page rendering.
    Reuses browser across multiple requests and handles crashes gracefully.
    """

    _instance: "BrowserManager | None" = None
    _playwright: Playwright | None = None
    _browser: Browser | None = None
    _lock: asyncio.Lock | None = None

    def __new__(cls) -> "BrowserManager":
        """Singleton pattern for browser manager."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    async def _ensure_lock(self) -> asyncio.Lock:
        """Ensure lock exists (create in async context)."""
        if self._lock is None:
            self._lock = asyncio.Lock()
        return self._lock

    async def get_browser(self) -> Browser:
        """Get or create the browser instance."""
        lock = await self._ensure_lock()
        async with lock:
            if self._browser is None or not self._browser.is_connected():
                await self._start_browser()
            return self._browser  # type: ignore

    async def _start_browser(self) -> None:
        """Start a new browser instance."""
        if self._playwright is None:
            self._playwright = await async_playwright().start()

        logger.info("starting_browser")
        self._browser = await self._playwright.chromium.launch(
            headless=True,
            args=[
                "--no-sandbox",
                "--disable-setuid-sandbox",
                "--disable-dev-shm-usage",
                "--disable-gpu",
            ],
        )
        logger.info("browser_started")

    async def close(self) -> None:
        """Close the browser and playwright."""
        lock = await self._ensure_lock()
        async with lock:
            if self._browser:
                await self._browser.close()
                self._browser = None
            if self._playwright:
                await self._playwright.stop()
                self._playwright = None
            logger.info("browser_closed")

    @asynccontextmanager
    async def get_page(self) -> AsyncGenerator[Page, None]:
        """Get a new page in a context manager."""
        browser = await self.get_browser()
        context = await browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
            viewport={"width": 1920, "height": 1080},
        )
        page = await context.new_page()
        try:
            yield page
        finally:
            await page.close()
            await context.close()


# Global browser manager instance
_browser_manager: BrowserManager | None = None


def get_browser_manager() -> BrowserManager:
    """Get the global browser manager instance."""
    global _browser_manager
    if _browser_manager is None:
        _browser_manager = BrowserManager()
    return _browser_manager


async def _fetch_rendered_html(url: str, timeout_ms: int | None = None) -> str:
    """
    Internal async function to fetch rendered HTML.
    
    Args:
        url: URL to fetch
        timeout_ms: Page load timeout in milliseconds
        
    Returns:
        Rendered HTML content
    """
    settings = get_settings()
    timeout = timeout_ms or (settings.page_load_timeout_seconds * 1000)
    
    manager = get_browser_manager()
    
    try:
        async with manager.get_page() as page:
            logger.info("fetching_page", url=url)
            
            # Navigate with timeout
            await page.goto(
                url,
                wait_until="networkidle",
                timeout=timeout,
            )
            
            # Wait a bit more for any late JS execution
            await page.wait_for_timeout(1000)
            
            # Get the rendered content
            content = await page.content()
            
            logger.info(
                "page_fetched",
                url=url,
                content_length=len(content),
            )
            
            return content
            
    except Exception as e:
        error_msg = str(e)
        logger.error(
            "page_fetch_failed",
            url=url,
            error=error_msg,
        )
        
        # Check for specific error types
        if "timeout" in error_msg.lower():
            return f"Error: Page load timeout after {timeout}ms for URL: {url}"
        elif "net::" in error_msg.lower():
            return f"Error: Network error loading URL: {url} - {error_msg}"
        else:
            return f"Error fetching/rendering page: {error_msg}"


@tool
async def get_rendered_html(url: str) -> str:
    """
    Fetch and return the fully rendered HTML of a webpage.

    This function uses Playwright to load a webpage in a headless Chromium
    browser, allowing all JavaScript on the page to execute. Use this for
    dynamic websites that require JavaScript rendering.

    IMPORTANT RESTRICTIONS:
    - ONLY use this for actual HTML webpages (articles, documentation, dashboards).
    - DO NOT use this for direct file links (URLs ending in .csv, .pdf, .zip, .png).
      Playwright cannot render these and will fail. Use the 'download_file' tool instead.

    Parameters
    ----------
    url : str
        The URL of the webpage to retrieve and render.

    Returns
    -------
    str
        The fully rendered HTML content, or an error message if fetching failed.
    """
    return await _fetch_rendered_html(url)


async def cleanup_browser() -> None:
    """Clean up the browser manager. Call this on shutdown."""
    manager = get_browser_manager()
    await manager.close()
