"""
Media fetching tool for multimodal Gemini analysis.
Downloads images, audio, and video files and returns base64-encoded data for Gemini to analyze directly.
Handles unknown file extensions by detecting type from content-type header.
"""

import base64
import httpx
import structlog
from langchain_core.tools import tool

log = structlog.get_logger(__name__)


# MIME types for supported media formats
MIME_TYPES = {
    # Audio
    "opus": "audio/ogg",
    "ogg": "audio/ogg",
    "mp3": "audio/mp3",
    "wav": "audio/wav",
    "aiff": "audio/aiff",
    "aac": "audio/aac",
    "flac": "audio/flac",
    "m4a": "audio/mp4",
    # Images
    "png": "image/png",
    "jpg": "image/jpeg",
    "jpeg": "image/jpeg",
    "gif": "image/gif",
    "webp": "image/webp",
    "bmp": "image/bmp",
    "svg": "image/svg+xml",
    # Video
    "mp4": "video/mp4",
    "webm": "video/webm",
    "mov": "video/quicktime",
    "avi": "video/x-msvideo",
    # Documents (PDF)
    "pdf": "application/pdf",
}

# Reverse mapping: mime type to category
MIME_TO_CATEGORY = {
    # Audio
    "audio/ogg": "audio",
    "audio/mpeg": "audio",
    "audio/mp3": "audio",
    "audio/wav": "audio",
    "audio/x-wav": "audio",
    "audio/aiff": "audio",
    "audio/aac": "audio",
    "audio/flac": "audio",
    "audio/mp4": "audio",
    "audio/webm": "audio",
    # Images
    "image/png": "image",
    "image/jpeg": "image",
    "image/gif": "image",
    "image/webp": "image",
    "image/bmp": "image",
    "image/svg+xml": "image",
    # Video
    "video/mp4": "video",
    "video/webm": "video",
    "video/quicktime": "video",
    "video/x-msvideo": "video",
    "video/ogg": "video",
    # Documents
    "application/pdf": "document",
}

# Categories for logging (by extension)
MEDIA_CATEGORIES = {
    "audio": ["opus", "ogg", "mp3", "wav", "aiff", "aac", "flac", "m4a"],
    "image": ["png", "jpg", "jpeg", "gif", "webp", "bmp", "svg"],
    "video": ["mp4", "webm", "mov", "avi"],
    "document": ["pdf"],
}


def get_media_category(ext: str) -> str:
    """Get the category of media based on extension."""
    for category, extensions in MEDIA_CATEGORIES.items():
        if ext in extensions:
            return category
    return "unknown"


def get_category_from_mime(mime_type: str) -> str:
    """Get category from mime type."""
    # Check exact match first
    if mime_type in MIME_TO_CATEGORY:
        return MIME_TO_CATEGORY[mime_type]
    
    # Check prefix
    if mime_type.startswith("audio/"):
        return "audio"
    elif mime_type.startswith("image/"):
        return "image"
    elif mime_type.startswith("video/"):
        return "video"
    elif mime_type == "application/pdf":
        return "document"
    
    return "unknown"


@tool
def fetch_media(media_url: str) -> str:
    """
    Fetch a media file (image, audio, video, or PDF) for Gemini to analyze directly.
    
    Use this tool when you encounter:
    - <audio> tags → fetch the audio file
    - <img> tags → fetch the image file  
    - <video> tags → fetch the video file
    - PDF links → fetch the PDF document
    - Any media URL, even with unknown extension
    
    Gemini can directly understand and analyze all these media types.
    The tool will detect the media type from the server's content-type header
    if the file extension is unknown.
    
    Args:
        media_url: URL of the media file to fetch
        
    Returns:
        A structured response with the media data that Gemini can analyze.
    """
    log.info("fetching_media", url=media_url)
    
    try:
        # Try to determine MIME type from URL extension first
        ext = media_url.split(".")[-1].split("?")[0].lower() if "." in media_url else ""
        mime_type = MIME_TYPES.get(ext)
        category = get_media_category(ext) if ext else "unknown"
        
        # Make initial HEAD request to get content-type
        with httpx.Client(timeout=60.0, follow_redirects=True) as client:
            try:
                head_response = client.head(media_url)
                content_type = head_response.headers.get("content-type", "")
                server_mime = content_type.split(";")[0].strip()
                
                # Use server's mime type if we don't have one or it's more specific
                if not mime_type or category == "unknown":
                    if server_mime:
                        mime_type = server_mime
                        category = get_category_from_mime(server_mime)
                        log.info("detected_mime_from_header", mime_type=mime_type, category=category)
            except httpx.HTTPError as e:
                log.warning("head_request_failed", error=str(e), url=media_url)
                # Continue - will try GET anyway
        
        # If still unknown, we'll try to download and detect from content
        if not mime_type or category == "unknown":
            log.info("unknown_extension_will_detect", url=media_url, ext=ext)
        
        # Download the media file
        with httpx.Client(timeout=60.0, follow_redirects=True) as client:
            response = client.get(media_url)
            response.raise_for_status()
            media_data = response.content
            
            # Get content-type from GET response if we still don't have it
            if not mime_type or category == "unknown":
                content_type = response.headers.get("content-type", "")
                server_mime = content_type.split(";")[0].strip()
                if server_mime:
                    mime_type = server_mime
                    category = get_category_from_mime(server_mime)
                    log.info("detected_mime_from_get", mime_type=mime_type, category=category)
        
        # Last resort: try to detect from magic bytes
        if not mime_type or category == "unknown":
            detected = detect_media_type_from_bytes(media_data)
            if detected:
                mime_type, category = detected
                log.info("detected_mime_from_bytes", mime_type=mime_type, category=category)
            else:
                # Default to binary if we can't detect
                log.warning("could_not_detect_media_type", url=media_url)
                return f"Error: Could not determine media type for {media_url}. The file may not be a supported media format."
        
        # Check size (Gemini has limits - 20MB for inline)
        size_mb = len(media_data) / (1024 * 1024)
        if size_mb > 20:
            return f"Error: Media file too large ({size_mb:.1f}MB). Maximum is 20MB for inline analysis."
        
        # Encode to base64
        encoded_media = base64.b64encode(media_data).decode("utf-8")
        
        log.info(
            "media_fetched",
            url=media_url,
            size_bytes=len(media_data),
            mime_type=mime_type,
            category=category,
        )
        
        # Return a structured response
        return f"""MEDIA_DATA_FOR_GEMINI:
category: {category}
mime_type: {mime_type}
base64_data: {encoded_media}

The {category} has been fetched successfully. Gemini will now analyze it.
Please describe what you see/hear in this {category}."""

    except httpx.HTTPStatusError as e:
        log.error("media_fetch_failed", url=media_url, status=e.response.status_code)
        return f"Error fetching media: HTTP {e.response.status_code}"
    except Exception as e:
        log.error("media_fetch_error", url=media_url, error=str(e))
        return f"Error fetching media: {str(e)}"


def detect_media_type_from_bytes(data: bytes) -> tuple[str, str] | None:
    """
    Detect media type from magic bytes.
    
    Returns (mime_type, category) or None if unknown.
    """
    if len(data) < 12:
        return None
    
    # PNG
    if data[:8] == b'\x89PNG\r\n\x1a\n':
        return "image/png", "image"
    
    # JPEG
    if data[:3] == b'\xff\xd8\xff':
        return "image/jpeg", "image"
    
    # GIF
    if data[:6] in (b'GIF87a', b'GIF89a'):
        return "image/gif", "image"
    
    # WebP
    if data[:4] == b'RIFF' and data[8:12] == b'WEBP':
        return "image/webp", "image"
    
    # PDF
    if data[:5] == b'%PDF-':
        return "application/pdf", "document"
    
    # MP3 (ID3 tag or sync bytes)
    if data[:3] == b'ID3' or data[:2] == b'\xff\xfb' or data[:2] == b'\xff\xfa':
        return "audio/mp3", "audio"
    
    # OGG (Vorbis, Opus, etc.)
    if data[:4] == b'OggS':
        return "audio/ogg", "audio"
    
    # WAV
    if data[:4] == b'RIFF' and data[8:12] == b'WAVE':
        return "audio/wav", "audio"
    
    # MP4/M4A (ftyp box)
    if data[4:8] == b'ftyp':
        # Check for audio-specific brands
        brand = data[8:12]
        if brand in (b'M4A ', b'M4B ', b'mp42'):
            return "audio/mp4", "audio"
        return "video/mp4", "video"
    
    # WebM (starts with EBML header for Matroska)
    if data[:4] == b'\x1aE\xdf\xa3':
        return "video/webm", "video"
    
    # FLAC
    if data[:4] == b'fLaC':
        return "audio/flac", "audio"
    
    return None


# Backward compatibility alias
fetch_audio = fetch_media
