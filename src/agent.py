"""
LangGraph-based quiz-solving agent with per-problem chat isolation and parallel retry.

Architecture:
- ProblemSolver: Handles a single problem URL with its own fresh chat state
- QuizOrchestrator: Manages the quiz chain, spawning new solvers per problem
  with parallel retry queue for failed problems (3-min timeout per problem)
"""

import asyncio
import json
import re
import time
from dataclasses import dataclass, field
from typing import Annotated, Any, List, TypedDict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from .config import get_settings
from .llm_client import MultiModelClient
from .logging_config import get_logger
from .tools import TOOLS
from .tools.download_file import set_workspace_path as set_download_path
from .tools.run_code import set_workspace_path as set_code_path
from .workspace import TaskWorkspace

logger = get_logger(__name__)


# -------------------------------------------------
# CONSTANTS
# -------------------------------------------------
PROBLEM_TIMEOUT_SECONDS = 180  # 3 minutes per problem (hard timeout)
SOFT_TIMEOUT_SECONDS = 160  # Soft timeout - submit dummy answer to get next URL


# -------------------------------------------------
# SUBMIT URL EXTRACTION FROM HTML
# -------------------------------------------------
SUBMIT_URL_PATTERNS = [
    # Common patterns in HTML for submit endpoints
    re.compile(r'action=["\']([^"\']*submit[^"\']*)["\']', re.IGNORECASE),
    re.compile(r'href=["\']([^"\']*submit[^"\']*)["\']', re.IGNORECASE),
    re.compile(r'post_request\(["\']([^"\']*submit[^"\']*)["\']', re.IGNORECASE),
    re.compile(r'/submit/\d+', re.IGNORECASE),
    re.compile(r'/submit', re.IGNORECASE),
]


def extract_submit_url_from_html(html_content: str, base_url: str = "") -> str | None:
    """Extract submit URL from HTML content."""
    for pattern in SUBMIT_URL_PATTERNS:
        match = pattern.search(html_content)
        if match:
            url = match.group(0) if match.lastindex is None else match.group(1)
            # Handle relative URLs
            if url.startswith('/') and base_url:
                from urllib.parse import urlparse
                parsed = urlparse(base_url)
                url = f"{parsed.scheme}://{parsed.netloc}{url}"
            elif url.startswith('http'):
                return url
            elif base_url:
                # Relative path
                from urllib.parse import urljoin
                url = urljoin(base_url, url)
            return url
    return None


# -------------------------------------------------
# MEDIA DATA EXTRACTION (audio, image, video, PDF)
# -------------------------------------------------
MEDIA_DATA_PATTERN = re.compile(
    r"MEDIA_DATA_FOR_GEMINI:\s*category:\s*(\S+)\s*mime_type:\s*(\S+)\s*base64_data:\s*(\S+)",
    re.DOTALL
)

AUDIO_DATA_PATTERN = re.compile(
    r"AUDIO_DATA_FOR_GEMINI:\s*mime_type:\s*(\S+)\s*base64_data:\s*(\S+)",
    re.DOTALL
)


def extract_media_data(content: str) -> tuple[str, str, str] | None:
    """Extract media data from tool response if present."""
    match = MEDIA_DATA_PATTERN.search(content)
    if match:
        return match.group(1), match.group(2), match.group(3)
    
    match = AUDIO_DATA_PATTERN.search(content)
    if match:
        return "audio", match.group(1), match.group(2)
    
    return None


# -------------------------------------------------
# STATE FOR SINGLE PROBLEM
# -------------------------------------------------
class ProblemState(TypedDict):
    """State for solving a single problem."""
    messages: Annotated[List, add_messages]
    problem_start_time: float
    problem_url: str
    workspace_path: str
    # Track submission result directly from tool responses
    submission_made: bool
    submission_correct: bool | None
    submission_next_url: str | None
    # Track detected submit URL from page content
    detected_submit_url: str | None
    # Track the last submitted answer (for retry feedback)
    last_submitted_answer: str | None
    # Track consecutive empty responses to break infinite loops
    consecutive_empty_responses: int


# -------------------------------------------------
# PROBLEM RESULT
# -------------------------------------------------
@dataclass
class ProblemResult:
    """Result from solving a single problem."""
    url: str
    correct: bool
    next_url: str | None = None
    answer: Any = None
    error: str | None = None
    elapsed_seconds: float = 0.0
    attempt_number: int = 1


# -------------------------------------------------
# SYSTEM PROMPT FOR SINGLE PROBLEM
# -------------------------------------------------
def get_problem_system_prompt(email: str, secret: str) -> str:
    """Generate system prompt for solving a single problem."""
    return f"""You are an autonomous quiz-solving agent designed to handle complex data tasks.

YOUR MISSION:
1. Load the quiz page from the provided URL using get_rendered_html
2. IMMEDIATELY after loading, extract and report the SUBMIT_URL like this:
   SUBMIT_URL: https://example.com/submit
   (This is CRITICAL - always report the exact submission endpoint URL right after loading the page)
3. Extract ALL instructions, required parameters, and submission rules
4. Solve the task exactly as required using the available tools
5. Submit the answer ONLY to the endpoint specified on the current page
6. After submitting, report the result including:
   - Whether 'correct' was true or false
   - The 'url' field if present (for next problem)
   - Any error messages

AVAILABLE TOOLS:
- get_rendered_html(url): Fetch JavaScript-rendered HTML pages
- download_file(url, filename): Download files (CSV, Excel, etc.)
- run_code(code): Execute Python code. Use print() to output results
- post_request(url, payload, headers): Submit answers
- get_request(url, headers, params): Fetch JSON data from APIs
- add_dependencies(dependencies): Install Python packages
- fetch_media(media_url): Fetch media for Gemini to analyze (audio, images, video, PDF)

STRICT RULES:

1. NEVER HALLUCINATE:
   - Never make up URLs, endpoints, or data values
   - Always extract exact values from the page or downloaded files

2. FILE HANDLING:
   - Use download_file for CSV, Excel, data files
   - Use fetch_media for images, audio, video, PDFs (Gemini analyzes directly)
   - Use get_rendered_html ONLY for HTML pages

3. AUDIO/MEDIA ANALYSIS (CRITICAL):
   - When you receive audio via fetch_media, FIRST TRANSCRIBE IT WORD-FOR-WORD
   - The audio contains TASK INSTRUCTIONS telling you what calculation to perform
   - Common audio instructions include things like:
     * "Find the sum of values where [column] is greater than [threshold]"
     * "Count records matching [condition]"
     * "Calculate [metric] for rows where [filter]"
   - The audio NEVER contains the answer directly - it tells you HOW to calculate
   - After transcribing:
     a) Parse the HTML page to get any thresholds/parameters mentioned
     b) Download referenced data files (CSV, JSON, etc.)
     c) Write Python code that implements the exact calculation from the audio
     d) Print and submit the numeric result

4. CODE EXECUTION:
   - Always use print() to output the final answer
   - When processing CSV files, read ALL rows (not just a sample)
   - Handle errors gracefully

5. MATPLOTLIB/VISUALIZATION PROBLEMS (CRITICAL):
   - If the problem asks about a chart/plot (e.g., "what value is at X position"):
     a) Generate the data for the plot programmatically
     b) Instead of actually creating plt.show() or saving images, CALCULATE the answer directly from the data
     c) For example, if asked "what is the Y value when X=10", compute it from the data, don't try to read it from an image
     d) Print the numerical answer directly
   - DO NOT use plt.show() - it will hang
   - DO NOT save images unless explicitly required
   - Focus on COMPUTING the answer from data, not visualizing it

6. ANSWER SUBMISSION:
   - Use the SUBMIT_URL you extracted earlier from the quiz page
   - Include all required fields in the payload
   - After POST, tell me the EXACT response including 'correct' and 'url' fields

7. COMPLETION:
   - After submitting and receiving a response, say "PROBLEM_DONE" followed by:
     * correct: true/false
     * next_url: the URL from response (or "none")
     * Example: "PROBLEM_DONE correct:true next_url:https://example.com/quiz/2"
     * Example: "PROBLEM_DONE correct:false next_url:none"

CREDENTIALS:
- Email: {email}
- Secret: {secret}
"""


# -------------------------------------------------
# SINGLE PROBLEM SOLVER
# -------------------------------------------------
class ProblemSolver:
    """
    Solves a single problem URL with its own fresh chat state.
    Each problem gets isolated context - no carryover from previous problems.
    """

    def __init__(self, workspace: TaskWorkspace | None = None):
        self.settings = get_settings()
        self.workspace = workspace
        
        if workspace:
            set_download_path(workspace.root)
            set_code_path(workspace.root)
        
        # Fresh LLM client for each problem
        self.llm_client = MultiModelClient(tools=TOOLS)
        self.graph = self._build_graph()
        self.app = self.graph.compile()

    def _build_graph(self) -> StateGraph:
        """Build the LangGraph state machine for single problem."""
        graph = StateGraph(ProblemState)
        graph.add_node("agent", self._agent_node)
        graph.add_node("tools", self._tools_node)  # Custom tools node to intercept results
        graph.add_edge(START, "agent")
        graph.add_edge("tools", "agent")
        graph.add_conditional_edges("agent", self._route)
        return graph

    def _extract_submit_url_from_llm(self, content: str) -> str | None:
        """Extract submit URL from LLM response text.
        
        The LLM is instructed to report: SUBMIT_URL: https://...
        """
        # Look for SUBMIT_URL: followed by a URL
        # Find the marker first
        marker = "SUBMIT_URL:"
        idx = content.upper().find(marker.upper())
        if idx == -1:
            return None
        
        # Get the text after the marker
        after_marker = content[idx + len(marker):].strip()
        
        # Extract the URL (take until whitespace or end)
        url_parts = after_marker.split()
        if not url_parts:
            return None
        
        url = url_parts[0].rstrip('.,;:"\'>)]')
        
        # Validate it looks like a URL
        if url.startswith("http://") or url.startswith("https://"):
            return url
        
        return None

    async def _tools_node(self, state: ProblemState) -> dict:
        """
        Custom tools node that intercepts post_request results to detect submissions.
        Also extracts submit URL from HTML responses for fallback.
        Uses async invocation to support async tools like get_rendered_html.
        """
        # Run the standard ToolNode asynchronously
        tool_node = ToolNode(TOOLS)
        result = await tool_node.ainvoke(state)
        
        # Check if any tool response contains submission result or HTML with submit URL
        messages = result.get("messages", [])
        updates = {}
        problem_url = state.get("problem_url", "")
        
        for msg in messages:
            if isinstance(msg, ToolMessage):
                content = getattr(msg, "content", "")
                
                # Try to extract submit URL from HTML content (for fallback)
                if isinstance(content, str) and not state.get("detected_submit_url"):
                    # Check if this looks like HTML (from get_rendered_html)
                    if "<" in content and ("submit" in content.lower() or "form" in content.lower()):
                        submit_url = extract_submit_url_from_html(content, problem_url)
                        if submit_url:
                            updates["detected_submit_url"] = submit_url
                            logger.info("submit_url_extracted_from_html", url=submit_url)
                
                try:
                    # Try to parse as JSON
                    if isinstance(content, str):
                        data = json.loads(content)
                    elif isinstance(content, dict):
                        data = content
                    else:
                        continue
                    
                    # Check if this is a submission response (has 'correct' field)
                    if "correct" in data:
                        updates["submission_made"] = True
                        updates["submission_correct"] = data.get("correct")
                        updates["submission_next_url"] = data.get("url")
                        
                        logger.info(
                            "submission_detected",
                            correct=data.get("correct"),
                            next_url=data.get("url"),
                        )
                except (json.JSONDecodeError, TypeError):
                    pass
        
        # Also capture the answer from tool calls (before they execute)
        # Look at the last AI message for post_request tool calls
        state_messages = state.get("messages", [])
        for msg in reversed(state_messages):
            if isinstance(msg, AIMessage) and hasattr(msg, "tool_calls"):
                for tc in msg.tool_calls or []:
                    if tc.get("name") == "post_request":
                        args = tc.get("args", {})
                        payload = args.get("payload", {})
                        if "answer" in payload:
                            updates["last_submitted_answer"] = str(payload["answer"])
                            break
                break
        
        result.update(updates)
        return result

    def _preprocess_messages(self, messages: list) -> list:
        """Preprocess messages to convert media data to multimodal format.
        
        Handles media data embedded in ToolMessages by extracting and converting
        them to multimodal HumanMessages that Gemini can understand.
        
        To prevent duplicate media messages when the agent loops, we track which
        tool_call_ids have already been converted by checking if a HumanMessage
        with media content follows the corresponding ToolMessage.
        """
        import base64
        
        media_prompts = {
            "audio": """CRITICAL AUDIO INSTRUCTION - Listen VERY carefully to this audio file:

1. FIRST: Transcribe EXACTLY what is said in the audio, word for word
2. The audio contains a TASK INSTRUCTION (e.g., "find the sum of values where column X is greater than Y")
3. After transcription, identify:
   - What operation to perform (sum, count, filter, etc.)
   - What column/field to use
   - What condition/threshold to apply
4. Then execute that calculation using downloaded data files and run_code
5. The audio does NOT contain the answer - it tells you WHAT CALCULATION to do

Transcribe the audio now, then follow the instructions.""",
            "image": "Look at this image carefully. Extract any relevant information needed to answer the quiz question.",
            "video": "Watch this video carefully. Extract any relevant information needed to answer the quiz question.",
            "document": "Read this PDF document carefully. Extract any relevant information needed to answer the quiz question.",
        }
        
        # First pass: identify which tool_call_ids already have media HumanMessages following them
        processed_tool_ids = set()
        for i, msg in enumerate(messages):
            if isinstance(msg, HumanMessage):
                content = getattr(msg, "content", "")
                if isinstance(content, list) and any(
                    isinstance(item, dict) and item.get("type") == "media" 
                    for item in content
                ):
                    # This is a media HumanMessage - find the preceding ToolMessage
                    # and mark its tool_call_id as processed
                    for j in range(i - 1, -1, -1):
                        if isinstance(messages[j], ToolMessage):
                            processed_tool_ids.add(messages[j].tool_call_id)
                            break
        
        # Second pass: process messages, skipping already-processed media
        processed = []
        for msg in messages:
            if isinstance(msg, ToolMessage):
                content = getattr(msg, "content", "")
                media_data = extract_media_data(content)
                
                if media_data:
                    tool_call_id = msg.tool_call_id
                    
                    # Check if this media has already been processed
                    if tool_call_id in processed_tool_ids:
                        # Already processed - just add a simple placeholder
                        processed.append(ToolMessage(
                            content=f"Media data already provided above.",
                            tool_call_id=tool_call_id,
                        ))
                        continue
                    
                    category, mime_type, base64_data = media_data
                    logger.info("converting_media_to_multimodal", category=category, mime_type=mime_type)
                    
                    processed.append(ToolMessage(
                        content=f"{category.capitalize()} data loaded. Analyze it to answer the quiz.",
                        tool_call_id=tool_call_id,
                    ))
                    
                    # Decode base64 to raw bytes - LangChain's media type expects bytes, not base64 string
                    try:
                        media_bytes = base64.b64decode(base64_data)
                        logger.info("media_bytes_decoded", size_bytes=len(media_bytes), mime_type=mime_type)
                    except Exception as e:
                        logger.error("failed_to_decode_media", error=str(e))
                        processed.append(msg)
                        continue
                    
                    prompt = media_prompts.get(category, f"Analyze this {category} to answer the quiz.")
                    multimodal_content = [
                        {"type": "text", "text": prompt},
                        {"type": "media", "data": media_bytes, "mime_type": mime_type},
                    ]
                    processed.append(HumanMessage(content=multimodal_content))
                else:
                    processed.append(msg)
            else:
                processed.append(msg)
        
        return processed

    def _agent_node(self, state: ProblemState) -> dict:
        """Process state and generate response."""
        messages = state["messages"]
        problem_url = state.get("problem_url", "")
        
        logger.info("problem_solver_invoked", message_count=len(messages), url=problem_url)
        
        # Check timeout
        elapsed = time.time() - state.get("problem_start_time", time.time())
        
        # Hard timeout
        if elapsed > PROBLEM_TIMEOUT_SECONDS:
            logger.warning("problem_hard_timeout", elapsed=elapsed, url=problem_url)
            return {"messages": [AIMessage(content="PROBLEM_DONE correct:false next_url:none error:timeout")]}
        
        # Soft timeout - submit dummy answer to get next URL
        if elapsed > SOFT_TIMEOUT_SECONDS and not state.get("submission_made"):
            detected_url = state.get("detected_submit_url")
            logger.warning("soft_timeout_triggered", elapsed=elapsed, url=problem_url, detected_submit_url=detected_url)
            
            if detected_url:
                # Create a tool call to submit a dummy answer
                logger.info("submitting_dummy_answer", submit_url=detected_url)
                dummy_payload = {
                    "email": self.settings.email,
                    "secret": self.settings.secret,
                    "url": problem_url,
                    "answer": "TIMEOUT_DUMMY_ANSWER",
                }
                # Return an AI message with tool call to submit
                tool_call_msg = AIMessage(
                    content="Soft timeout reached. Submitting dummy answer to get next URL.",
                    tool_calls=[{
                        "id": "soft_timeout_submit",
                        "name": "post_request",
                        "args": {
                            "url": detected_url,
                            "payload": dummy_payload,
                        },
                    }],
                )
                return {"messages": [tool_call_msg]}
            else:
                # No submit URL detected, just timeout
                logger.warning("soft_timeout_no_submit_url", elapsed=elapsed, url=problem_url)
                return {"messages": [AIMessage(content="PROBLEM_DONE correct:false next_url:none error:soft_timeout_no_submit_url")]}
        
        system_prompt = get_problem_system_prompt(self.settings.email, self.settings.secret)
        processed_messages = self._preprocess_messages(messages)
        
        llm_messages = [
            SystemMessage(content=system_prompt),
            *processed_messages,
        ]
        
        logger.info("invoking_llm_for_problem", message_count=len(llm_messages))
        
        try:
            result = self.llm_client.invoke(llm_messages)
            has_tool_calls = bool(getattr(result, "tool_calls", None))
            content = getattr(result, "content", "")
            content_length = len(str(content)) if content else 0
            
            logger.info("problem_solver_response", 
                       has_tool_calls=has_tool_calls,
                       content_length=content_length)
            
            # Track consecutive empty responses
            updates = {"messages": [result]}
            
            if not has_tool_calls and content_length == 0:
                # Empty response - increment counter
                empty_count = state.get("consecutive_empty_responses", 0) + 1
                updates["consecutive_empty_responses"] = empty_count
                logger.warning("empty_llm_response", count=empty_count)
                
                # After 3 empty responses, inject a nudge message
                if empty_count >= 3:
                    logger.warning("injecting_nudge_after_empty_responses", count=empty_count)
                    nudge = HumanMessage(content="""You have not responded. Please continue working on the problem.

If you have the data needed, use run_code to calculate the answer.
If you need to submit, use post_request with the submit URL.
If you're stuck, say "PROBLEM_DONE correct:false next_url:none" to move on.""")
                    updates["messages"] = [result, nudge]
                    updates["consecutive_empty_responses"] = 0  # Reset after nudge
            else:
                # Non-empty response - reset counter
                updates["consecutive_empty_responses"] = 0
            
            # Extract SUBMIT_URL from AI response if present - LLM URL always takes priority
            if isinstance(content, str):
                submit_url = self._extract_submit_url_from_llm(content)
                if submit_url:
                    updates["detected_submit_url"] = submit_url
                    logger.info("submit_url_from_llm", url=submit_url)
            
            return updates
        except Exception as e:
            logger.error("problem_solver_error", error=str(e))
            return {"messages": [AIMessage(content=f"PROBLEM_DONE correct:false next_url:none error:{str(e)}")]}

    def _route(self, state: ProblemState) -> str:
        """Determine next node based on last message."""
        # Check if submission was made - end immediately
        if state.get("submission_made"):
            logger.info("routing_to_end_submission_made", 
                       correct=state.get("submission_correct"),
                       next_url=state.get("submission_next_url"))
            return END
        
        # Check message count to prevent infinite loops (max 50 messages)
        if len(state["messages"]) > 50:
            logger.warning("max_messages_reached", count=len(state["messages"]))
            return END
        
        last = state["messages"][-1]
        
        tool_calls = getattr(last, "tool_calls", None)
        if isinstance(last, dict):
            tool_calls = last.get("tool_calls")
        if tool_calls:
            return "tools"
        
        content = getattr(last, "content", "")
        if isinstance(last, dict):
            content = last.get("content", "")
        
        # Check for completion markers
        if isinstance(content, str) and ("PROBLEM_DONE" in content or "END" in content.upper()):
            logger.info("problem_finished_marker", content=content[:100])
            return END
        
        # Handle list content format
        if isinstance(content, list):
            for item in content:
                text = item.get("text", "") if isinstance(item, dict) else str(item)
                if "PROBLEM_DONE" in text or "END" in text.upper():
                    return END
        
        return "agent"

    def _parse_result(self, content: str) -> ProblemResult:
        """Parse the PROBLEM_DONE response."""
        result = ProblemResult(url="", correct=False)
        
        if "correct:true" in content.lower():
            result.correct = True
        
        # Extract next_url
        url_match = re.search(r'next_url:(\S+)', content)
        if url_match:
            url = url_match.group(1)
            if url.lower() != "none" and url.startswith("http"):
                result.next_url = url
        
        # Extract error if present
        error_match = re.search(r'error:(.+?)(?:\s|$)', content)
        if error_match:
            result.error = error_match.group(1)
        
        return result

    async def _submit_timeout_answer(self, problem_url: str) -> str | None:
        """
        Submit a dummy answer to get the next URL when a timeout occurs.
        This ensures the quiz chain can continue even if we couldn't solve a problem.
        
        Args:
            problem_url: The URL of the problem that timed out
            
        Returns:
            The next URL if available, None otherwise
        """
        import httpx
        
        # Derive the submit URL from the problem URL
        # e.g., https://site.com/quiz/5 -> https://site.com/submit
        # or https://site.com/q5.html -> https://site.com/submit
        base_url = problem_url.rsplit('/', 1)[0]
        submit_url = f"{base_url}/submit"
        
        payload = {
            "email": self.settings.email,
            "secret": self.settings.secret,
            "url": problem_url,
            "answer": "TIMEOUT_FALLBACK",
        }
        
        logger.info("submitting_timeout_answer", 
                   problem_url=problem_url, 
                   submit_url=submit_url,
                   answer=payload["answer"])
        
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                response = await client.post(submit_url, json=payload)
                response.raise_for_status()
                data = response.json()
                
                next_url = data.get("url")
                logger.info("timeout_submission_result",
                           correct=data.get("correct"),
                           next_url=next_url,
                           response=data)
                return next_url
                
        except Exception as e:
            logger.error("timeout_submission_failed", error=str(e))
            return None

    async def solve(self, url: str, attempt_number: int = 1, previous_answer: str | None = None) -> ProblemResult:
        """
        Solve a single problem with fresh chat state.
        
        Args:
            url: Problem URL to solve
            attempt_number: Which attempt this is (for logging)
            previous_answer: The answer from the previous failed attempt (if any)
            
        Returns:
            ProblemResult with success/failure and next URL if any
        """
        logger.info("solving_problem", url=url, attempt=attempt_number, previous_answer=previous_answer)
        start_time = time.time()
        
        # Build initial message with retry context if applicable
        if attempt_number > 1 and previous_answer:
            initial_message = f"""Solve the quiz at: {url}

IMPORTANT - RETRY ATTEMPT #{attempt_number}:
Your previous answer "{previous_answer}" was INCORRECT.
You MUST try a DIFFERENT approach:
- Re-read the problem carefully
- Check if you misunderstood the audio/instructions
- Verify your calculations step by step
- Consider alternative interpretations of the question
- DO NOT submit the same answer again"""
        else:
            initial_message = f"Solve the quiz at: {url}"
        
        initial_state: ProblemState = {
            "messages": [HumanMessage(content=initial_message)],
            "problem_start_time": start_time,
            "problem_url": url,
            "workspace_path": str(self.workspace.root) if self.workspace else "",
            "submission_made": False,
            "submission_correct": None,
            "submission_next_url": None,
            "detected_submit_url": None,
            "last_submitted_answer": None,
            "consecutive_empty_responses": 0,
        }
        
        try:
            result = await asyncio.wait_for(
                self.app.ainvoke(
                    initial_state,
                    config={"recursion_limit": self.settings.max_recursion_limit},
                ),
                timeout=PROBLEM_TIMEOUT_SECONDS,
            )
            
            elapsed = time.time() - start_time
            
            # Get result from state (intercepted from tool response)
            correct = result.get("submission_correct", False)
            next_url = result.get("submission_next_url")
            answer = result.get("last_submitted_answer")
            
            # If submission wasn't detected, try parsing from messages
            if not result.get("submission_made"):
                last_msg = result.get("messages", [])[-1]
                content = getattr(last_msg, "content", "")
                if isinstance(content, list):
                    content = " ".join(str(c) for c in content)
                parsed = self._parse_result(content)
                correct = parsed.correct
                next_url = parsed.next_url
            
            problem_result = ProblemResult(
                url=url,
                correct=correct,
                next_url=next_url,
                answer=answer,
                elapsed_seconds=elapsed,
                attempt_number=attempt_number,
            )
            
            logger.info("problem_solved", 
                       url=url, 
                       correct=problem_result.correct,
                       next_url=problem_result.next_url,
                       answer=answer,
                       elapsed=elapsed,
                       attempt=attempt_number)
            
            return problem_result
            
        except asyncio.TimeoutError:
            elapsed = time.time() - start_time
            logger.error("problem_timeout", url=url, elapsed=elapsed, attempt=attempt_number)
            
            # Try to submit a dummy answer to get the next URL so the chain can continue
            next_url = await self._submit_timeout_answer(url)
            
            return ProblemResult(
                url=url,
                correct=False,
                next_url=next_url,
                error="timeout",
                elapsed_seconds=elapsed,
                attempt_number=attempt_number,
            )
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error("problem_error", url=url, error=str(e), attempt=attempt_number)
            return ProblemResult(
                url=url,
                correct=False,
                error=str(e),
                elapsed_seconds=elapsed,
                attempt_number=attempt_number,
            )


# -------------------------------------------------
# QUIZ ORCHESTRATOR
# -------------------------------------------------
MAX_RETRY_ATTEMPTS = 3


class QuizOrchestrator:
    """
    Orchestrates solving a quiz chain with simple synchronous retry.
    
    Strategy:
    1. Process problems sequentially, spawn new ProblemSolver for each
    2. On failure: retry up to MAX_RETRY_ATTEMPTS times immediately
    3. After retries exhausted, move on to next problem
    """

    def __init__(self, workspace: TaskWorkspace | None = None):
        self.workspace = workspace
        self.settings = get_settings()
        
        # Results tracking
        self.results: list[ProblemResult] = []

    async def _solve_with_retries(self, url: str, problem_number: int) -> tuple[ProblemResult, str | None]:
        """
        Solve a problem with up to MAX_RETRY_ATTEMPTS attempts within SOFT_TIMEOUT_SECONDS.
        
        Returns:
            Tuple of (final result, next_url to continue chain)
        """
        next_url = None
        last_result = None
        previous_answer = None  # Track the previous wrong answer
        problem_start_time = time.time()
        
        for attempt in range(1, MAX_RETRY_ATTEMPTS + 1):
            # Check if we've exceeded the retry time limit (160 seconds)
            elapsed = time.time() - problem_start_time
            if elapsed > SOFT_TIMEOUT_SECONDS:
                logger.info("retry_time_limit_exceeded",
                           problem_number=problem_number,
                           url=url,
                           elapsed=elapsed,
                           limit=SOFT_TIMEOUT_SECONDS)
                break
            
            logger.info("solving_attempt", 
                       problem_number=problem_number,
                       url=url, 
                       attempt=attempt,
                       max_attempts=MAX_RETRY_ATTEMPTS,
                       elapsed_seconds=elapsed,
                       previous_answer=previous_answer)
            
            # Create fresh solver for each attempt
            solver = ProblemSolver(workspace=self.workspace)
            
            try:
                # Pass previous answer to help LLM avoid repeating the same mistake
                result = await solver.solve(url, attempt_number=attempt, previous_answer=previous_answer)
                last_result = result
                
                # Always capture next_url from the response (needed to continue chain)
                if result.next_url:
                    next_url = result.next_url
                
                if result.correct:
                    logger.info("problem_correct", 
                               problem_number=problem_number,
                               url=url,
                               attempt=attempt)
                    return result, next_url
                else:
                    # Capture the wrong answer for the next retry
                    if result.answer:
                        previous_answer = str(result.answer)
                    
                    logger.info("attempt_failed", 
                               problem_number=problem_number,
                               url=url,
                               attempt=attempt,
                               answer=result.answer,
                               error=result.error)
                    
                    # If we have more attempts left and time remaining, wait a bit before retrying
                    if attempt < MAX_RETRY_ATTEMPTS:
                        await asyncio.sleep(1)
                        
            except Exception as e:
                logger.error("attempt_error", 
                            problem_number=problem_number,
                            url=url, 
                            attempt=attempt,
                            error=str(e))
                last_result = ProblemResult(
                    url=url,
                    correct=False,
                    error=str(e),
                    attempt_number=attempt,
                )
                if attempt < MAX_RETRY_ATTEMPTS:
                    await asyncio.sleep(1)
        
        # All attempts exhausted or time limit reached
        logger.info("retries_finished", 
                   problem_number=problem_number,
                   url=url,
                   elapsed_seconds=time.time() - problem_start_time)
        
        return last_result, next_url

    async def solve_chain(self, start_url: str) -> dict[str, Any]:
        """
        Solve the entire quiz chain starting from start_url.
        
        Returns:
            Summary of results including successes and failures
        """
        logger.info("orchestrator_starting", start_url=start_url)
        
        current_url = start_url
        problem_count = 0
        
        while current_url:
            problem_count += 1
            
            logger.info("processing_problem", 
                       problem_number=problem_count, 
                       url=current_url)
            
            # Solve with retries
            result, next_url = await self._solve_with_retries(current_url, problem_count)
            self.results.append(result)
            
            # Move to next problem
            current_url = next_url
        
        # Compile final results
        correct_count = sum(1 for r in self.results if r.correct)
        
        summary = {
            "total_problems": problem_count,
            "correct_first_try": correct_count,  # Simplified - just count correct
            "correct_after_retry": 0,  # Not tracking separately anymore
            "total_correct": correct_count,
            "results": self.results,
            "retry_results": [],  # No separate retry tracking
        }
        
        logger.info("orchestrator_complete",
                   total_problems=problem_count,
                   total_correct=correct_count)
        
        return summary


# -------------------------------------------------
# LEGACY INTERFACE (for backward compatibility)
# -------------------------------------------------
class QuizAgent:
    """
    Legacy interface wrapping QuizOrchestrator.
    Kept for backward compatibility with main.py
    """

    def __init__(self, workspace: TaskWorkspace | None = None):
        self.workspace = workspace
        self.settings = get_settings()

    def run(self, url: str) -> dict[str, Any]:
        """Synchronous run - not recommended, use arun instead."""
        return asyncio.run(self.arun(url))

    async def arun(self, url: str) -> dict[str, Any]:
        """
        Run the orchestrator on a quiz URL.
        """
        logger.info("agent_starting_async", url=url)
        
        orchestrator = QuizOrchestrator(workspace=self.workspace)
        result = await orchestrator.solve_chain(url)
        
        # Convert to messages format for compatibility
        messages = [
            HumanMessage(content=f"Started solving: {url}"),
            AIMessage(content=f"Completed. Total: {result['total_problems']}, "
                            f"Correct first try: {result['correct_first_try']}, "
                            f"Correct after retry: {result['correct_after_retry']}")
        ]
        
        return {
            "messages": messages,
            "summary": result,
        }


# -------------------------------------------------
# CONVENIENCE FUNCTIONS
# -------------------------------------------------
def run_agent(url: str, workspace: TaskWorkspace | None = None) -> dict[str, Any]:
    """Run the agent synchronously."""
    agent = QuizAgent(workspace=workspace)
    return agent.run(url)


async def arun_agent(url: str, workspace: TaskWorkspace | None = None) -> dict[str, Any]:
    """Run the agent asynchronously."""
    agent = QuizAgent(workspace=workspace)
    return await agent.arun(url)
