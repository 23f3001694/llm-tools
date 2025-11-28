"""
HTTP request tools with proper timeout handling and error reporting.
"""

import json
from typing import Any

import httpx
from langchain_core.tools import tool

from ..config import get_settings
from ..logging_config import get_logger

logger = get_logger(__name__)


@tool
def post_request(
    url: str,
    payload: dict[str, Any],
    headers: dict[str, str] | None = None,
) -> dict[str, Any]:
    """
    Send an HTTP POST request with a JSON payload.

    Use this tool to submit answers to quiz endpoints or interact with APIs.
    The response is automatically parsed as JSON if possible.

    IMPORTANT:
    - This is a blocking function that waits for the response
    - Check the 'correct' field in response to see if the answer was right
    - If 'correct' is false and 'delay' < 180, you can retry with a different answer
    - If a new 'url' is provided, proceed to solve that quiz next

    Parameters
    ----------
    url : str
        The endpoint URL to send the POST request to.
    payload : dict
        The JSON-serializable request body.
    headers : dict, optional
        Additional HTTP headers. Default includes Content-Type: application/json.

    Returns
    -------
    dict
        Response from the server containing:
        - 'correct': bool - whether the answer was correct
        - 'url': str | None - next quiz URL if available
        - 'reason': str | None - explanation if answer was wrong
        - 'delay': int - time elapsed since task started
        Or an error dict if the request failed.
    """
    settings = get_settings()
    timeout = settings.http_request_timeout_seconds
    
    # Merge headers with defaults
    default_headers = {"Content-Type": "application/json"}
    if headers:
        default_headers.update(headers)
    
    logger.info(
        "sending_post_request",
        url=url,
        payload_keys=list(payload.keys()),
        answer=payload.get("answer"),  # Log the actual answer for debugging
    )
    
    try:
        with httpx.Client(timeout=timeout, follow_redirects=True) as client:
            response = client.post(url, json=payload, headers=default_headers)
            response.raise_for_status()
            
            # Try to parse as JSON
            try:
                data = response.json()
            except json.JSONDecodeError:
                data = {"raw_response": response.text}
            
            logger.info(
                "post_response_received",
                url=url,
                status_code=response.status_code,
                correct=data.get("correct"),
                has_next_url=bool(data.get("url")),
            )
            
            # Process the response for the agent
            delay = data.get("delay", 0)
            correct = data.get("correct")
            
            # If answer is wrong and still have time, indicate retry is possible
            if correct is False and isinstance(delay, (int, float)) and delay < 180:
                data["_can_retry"] = True
                data["_time_remaining"] = 180 - delay
                logger.info(
                    "answer_incorrect_can_retry",
                    time_remaining=180 - delay,
                )
            
            # If time is up, only return the next URL
            if isinstance(delay, (int, float)) and delay >= 180:
                logger.warning(
                    "time_limit_exceeded",
                    delay=delay,
                )
                return {
                    "url": data.get("url"),
                    "_time_exceeded": True,
                    "reason": "Time limit exceeded. Move to next question if URL is provided.",
                }
            
            return data
            
    except httpx.TimeoutException:
        logger.error("post_request_timeout", url=url, timeout=timeout)
        return {
            "error": f"Request timed out after {timeout} seconds",
            "url": url,
            "_can_retry": True,
        }
    except httpx.HTTPStatusError as e:
        # Try to get error details from response
        try:
            error_data = e.response.json()
        except (json.JSONDecodeError, AttributeError):
            error_data = {"raw_response": e.response.text if e.response else str(e)}
        
        logger.error(
            "post_request_http_error",
            url=url,
            status_code=e.response.status_code if e.response else None,
            error=str(error_data),
        )
        
        return {
            "error": f"HTTP {e.response.status_code if e.response else 'unknown'}",
            "details": error_data,
            "_can_retry": e.response.status_code >= 500 if e.response else True,
        }
    except httpx.RequestError as e:
        logger.error("post_request_error", url=url, error=str(e))
        return {
            "error": f"Request failed: {str(e)}",
            "_can_retry": True,
        }
    except Exception as e:
        logger.error("post_request_unexpected_error", url=url, error=str(e))
        return {
            "error": f"Unexpected error: {str(e)}",
            "_can_retry": False,
        }


@tool
def get_request(
    url: str,
    headers: dict[str, str] | None = None,
    params: dict[str, str] | None = None,
) -> dict[str, Any]:
    """
    Send an HTTP GET request to retrieve data from an API or webpage.

    Use this tool for:
    - Fetching JSON data from APIs
    - Retrieving text content that doesn't need JavaScript rendering

    For JavaScript-rendered pages, use 'get_rendered_html' instead.
    For downloading files, use 'download_file' instead.

    Parameters
    ----------
    url : str
        The URL to send the GET request to.
    headers : dict, optional
        HTTP headers to include in the request.
    params : dict, optional
        Query parameters to append to the URL.

    Returns
    -------
    dict
        Response data, either:
        - Parsed JSON if response is JSON
        - {'text': str} if response is text
        - {'error': str} if request failed
    """
    settings = get_settings()
    timeout = settings.http_request_timeout_seconds
    
    logger.info("sending_get_request", url=url)
    
    try:
        with httpx.Client(timeout=timeout, follow_redirects=True) as client:
            response = client.get(url, headers=headers, params=params)
            response.raise_for_status()
            
            # Try to parse as JSON
            content_type = response.headers.get("content-type", "")
            
            if "application/json" in content_type:
                try:
                    data = response.json()
                    logger.info(
                        "get_response_received",
                        url=url,
                        content_type="json",
                    )
                    return data if isinstance(data, dict) else {"data": data}
                except json.JSONDecodeError:
                    pass
            
            # Return as text
            logger.info(
                "get_response_received",
                url=url,
                content_type="text",
                length=len(response.text),
            )
            return {"text": response.text, "content_type": content_type}
            
    except httpx.TimeoutException:
        logger.error("get_request_timeout", url=url, timeout=timeout)
        return {"error": f"Request timed out after {timeout} seconds"}
    except httpx.HTTPStatusError as e:
        logger.error(
            "get_request_http_error",
            url=url,
            status_code=e.response.status_code if e.response else None,
        )
        return {
            "error": f"HTTP {e.response.status_code if e.response else 'unknown'}",
            "response": e.response.text[:500] if e.response else None,
        }
    except httpx.RequestError as e:
        logger.error("get_request_error", url=url, error=str(e))
        return {"error": f"Request failed: {str(e)}"}
    except Exception as e:
        logger.error("get_request_unexpected_error", url=url, error=str(e))
        return {"error": f"Unexpected error: {str(e)}"}
