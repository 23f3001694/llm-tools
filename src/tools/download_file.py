"""
Robust file downloader with size limits, timeouts, and streaming.
"""

import os
from pathlib import Path
from urllib.parse import urlparse, unquote

import httpx
from langchain_core.tools import tool

from ..config import get_settings
from ..logging_config import get_logger

logger = get_logger(__name__)

# Global workspace path - set by the agent when starting a task
_current_workspace_path: Path | None = None


def set_workspace_path(path: Path) -> None:
    """Set the current workspace path for file downloads."""
    global _current_workspace_path
    _current_workspace_path = path


def get_workspace_path() -> Path:
    """Get the current workspace path, or fallback to LLMFiles."""
    if _current_workspace_path:
        return _current_workspace_path / "downloads"
    # Fallback for backwards compatibility
    fallback = Path("LLMFiles")
    fallback.mkdir(exist_ok=True)
    return fallback


def _extract_filename_from_url(url: str) -> str:
    """Extract a reasonable filename from a URL."""
    parsed = urlparse(url)
    path = unquote(parsed.path)
    
    # Get the last component of the path
    filename = os.path.basename(path)
    
    # If no filename, use the last path component or a default
    if not filename or filename == "/":
        # Try to use the second-to-last path component
        parts = [p for p in path.split("/") if p]
        if parts:
            filename = parts[-1]
        else:
            filename = "downloaded_file"
    
    # Ensure we have an extension if possible
    if "." not in filename:
        # Check content-type later, for now just keep the name
        pass
    
    return filename


@tool
def download_file(url: str, filename: str | None = None) -> str:
    """
    Download a file from a URL and save it to the workspace.

    This function downloads files with proper timeout handling, size limits,
    and streaming to avoid memory issues with large files.

    Use this for downloading:
    - CSV, JSON, Excel files
    - PDF documents
    - Images (PNG, JPG, etc.)
    - Any direct file links

    DO NOT use this for HTML pages that need JavaScript rendering.
    Use 'get_rendered_html' for those instead.

    Parameters
    ----------
    url : str
        Direct URL to the file to download.
    filename : str, optional
        The filename to save the downloaded content as.
        If not provided, will be extracted from the URL.

    Returns
    -------
    str
        Full path to the saved file on success, or an error message on failure.
    """
    settings = get_settings()
    max_size_bytes = settings.max_download_size_mb * 1024 * 1024
    timeout = settings.http_request_timeout_seconds

    # Determine filename
    if not filename:
        filename = _extract_filename_from_url(url)

    # Ensure filename is safe
    filename = "".join(c for c in filename if c.isalnum() or c in "._-")
    if not filename:
        filename = "downloaded_file"

    # Get download directory
    download_dir = get_workspace_path()
    download_dir.mkdir(parents=True, exist_ok=True)
    filepath = download_dir / filename

    logger.info(
        "downloading_file",
        url=url,
        filename=filename,
        max_size_mb=settings.max_download_size_mb,
    )

    try:
        with httpx.Client(timeout=timeout, follow_redirects=True) as client:
            # First, do a HEAD request to check size
            try:
                head_response = client.head(url)
                content_length = head_response.headers.get("content-length")
                if content_length and int(content_length) > max_size_bytes:
                    return (
                        f"Error: File too large ({int(content_length) / 1024 / 1024:.1f}MB). "
                        f"Maximum allowed size is {settings.max_download_size_mb}MB."
                    )
            except httpx.HTTPError:
                # HEAD not supported, continue with GET
                pass

            # Stream the download
            with client.stream("GET", url) as response:
                response.raise_for_status()

                # Check content-length header
                content_length = response.headers.get("content-length")
                if content_length and int(content_length) > max_size_bytes:
                    return (
                        f"Error: File too large ({int(content_length) / 1024 / 1024:.1f}MB). "
                        f"Maximum allowed size is {settings.max_download_size_mb}MB."
                    )

                # Update filename with proper extension from content-type if needed
                content_type = response.headers.get("content-type", "")
                if "." not in filename and content_type:
                    ext = _get_extension_from_content_type(content_type)
                    if ext:
                        filename = f"{filename}.{ext}"
                        filepath = download_dir / filename

                # Download with size tracking
                downloaded_bytes = 0
                with open(filepath, "wb") as f:
                    for chunk in response.iter_bytes(chunk_size=8192):
                        downloaded_bytes += len(chunk)
                        if downloaded_bytes > max_size_bytes:
                            f.close()
                            filepath.unlink(missing_ok=True)
                            return (
                                f"Error: Download exceeded maximum size of "
                                f"{settings.max_download_size_mb}MB."
                            )
                        f.write(chunk)

        logger.info(
            "file_downloaded",
            url=url,
            filepath=str(filepath),
            size_bytes=downloaded_bytes,
        )

        return str(filepath)

    except httpx.TimeoutException:
        return f"Error: Download timed out after {timeout} seconds for URL: {url}"
    except httpx.HTTPStatusError as e:
        return f"Error: HTTP {e.response.status_code} when downloading {url}"
    except httpx.RequestError as e:
        return f"Error: Network error downloading {url}: {str(e)}"
    except Exception as e:
        logger.error(
            "download_failed",
            url=url,
            error=str(e),
        )
        return f"Error downloading file: {str(e)}"


def _get_extension_from_content_type(content_type: str) -> str | None:
    """Map content-type to file extension."""
    content_type = content_type.lower().split(";")[0].strip()
    
    mapping = {
        "application/pdf": "pdf",
        "application/json": "json",
        "text/csv": "csv",
        "text/plain": "txt",
        "text/html": "html",
        "image/png": "png",
        "image/jpeg": "jpg",
        "image/gif": "gif",
        "image/webp": "webp",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": "xlsx",
        "application/vnd.ms-excel": "xls",
        "application/zip": "zip",
        "application/gzip": "gz",
    }
    
    return mapping.get(content_type)
