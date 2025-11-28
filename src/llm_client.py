"""
Simplified LLM client using only Gemini 2.5 Flash.
Gemini 2.5 Flash supports multimodal input (text, images, audio, video).

Configured for paid API tier with no rate limiting.
"""

from typing import Any

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.language_models import BaseChatModel

from .config import get_settings
from .logging_config import get_logger

logger = get_logger(__name__)


class LLMClientError(Exception):
    """Base exception for LLM client errors."""
    pass


class RateLimitError(LLMClientError):
    """Raised when rate limit is exceeded and retries exhausted."""
    pass


class GeminiClient:
    """
    Gemini 2.5 Flash client with multimodal support.
    
    Supports:
    - Text input/output
    - Image understanding
    - Audio understanding (opus, mp3, wav, etc.)
    - Video understanding
    
    Configured for paid API tier - no rate limiting.
    """

    MAX_RETRIES = 3
    INITIAL_BACKOFF = 2  # seconds
    MAX_BACKOFF = 30  # seconds

    def __init__(
        self,
        tools: list[Any] | None = None,
        model_name: str = "gemini-2.5-flash",
    ):
        """
        Initialize the Gemini client.
        
        Args:
            tools: List of LangChain tools to bind to the model.
            model_name: Gemini model to use.
        """
        self.settings = get_settings()
        self.tools = tools or []
        self.model_name = model_name
        self._model: BaseChatModel | None = None
        
        self._initialize_model()

    def _initialize_model(self) -> None:
        """Initialize the Gemini model."""
        if not self.settings.google_api_key:
            raise LLMClientError("GOOGLE_API_KEY is required for Gemini")

        self._model = ChatGoogleGenerativeAI(
            model=self.model_name,
            api_key=self.settings.google_api_key,
            temperature=0.0,
        )

        # Bind tools if provided
        if self.tools:
            self._model = self._model.bind_tools(self.tools)
        
        logger.info(
            "gemini_initialized",
            model=self.model_name,
            tool_count=len(self.tools),
        )

    def get_model(self) -> BaseChatModel:
        """Get the Gemini model instance."""
        if not self._model:
            raise LLMClientError("Model not initialized")
        return self._model

    async def ainvoke(self, messages: list[dict]) -> Any:
        """
        Invoke the Gemini model asynchronously.
        
        Args:
            messages: List of message dicts to send to the LLM.
            
        Returns:
            LLM response.
        """
        if not self._model:
            raise LLMClientError("Model not initialized")
        
        logger.debug(
            "invoking_gemini",
            message_count=len(messages),
        )
        
        result = await self._model.ainvoke(messages)
        
        logger.debug(
            "gemini_response_received",
            has_tool_calls=bool(getattr(result, "tool_calls", None)),
        )
        
        return result

    def invoke(self, messages: list[dict]) -> Any:
        """
        Invoke the Gemini model synchronously.
        
        Args:
            messages: List of message dicts to send to the LLM.
            
        Returns:
            LLM response.
        """
        if not self._model:
            raise LLMClientError("Model not initialized")
        
        logger.debug(
            "invoking_gemini_sync",
            message_count=len(messages),
        )
        
        result = self._model.invoke(messages)
        
        logger.debug(
            "gemini_response_received",
            has_tool_calls=bool(getattr(result, "tool_calls", None)),
        )
        
        return result


# Alias for backward compatibility
MultiModelClient = GeminiClient
