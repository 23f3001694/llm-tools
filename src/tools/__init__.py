"""Tools package initialization."""

from .web_scraper import get_rendered_html, BrowserManager
from .download_file import download_file
from .run_code import run_code
from .send_request import post_request, get_request
from .add_dependencies import add_dependencies
from .fetch_media import fetch_media

__all__ = [
    "get_rendered_html",
    "BrowserManager",
    "download_file",
    "run_code",
    "post_request",
    "get_request",
    "add_dependencies",
    "fetch_media",
]

# List of all tools for agent binding
TOOLS = [
    run_code,
    get_rendered_html,
    download_file,
    post_request,
    get_request,
    add_dependencies,
    fetch_media,
]
