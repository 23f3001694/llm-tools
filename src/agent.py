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
# RETRY QUEUE ITEM
# -------------------------------------------------
@dataclass
class RetryItem:
    """Item in the retry queue."""
    url: str
    start_time: float  # When this problem was first received
    attempt_count: int = 0
    last_error: str | None = None


# -------------------------------------------------
# SYSTEM PROMPT FOR SINGLE PROBLEM
# -------------------------------------------------
def get_problem_system_prompt(email: str, secret: str) -> str:
    """Generate system prompt for solving a single problem."""
    return f"""You are an autonomous quiz-solving agent designed to handle complex data tasks.

YOUR MISSION:
1. Load the quiz page from the provided URL using get_rendered_html
2. Extract ALL instructions, required parameters, submission rules, and the submit endpoint
3. Solve the task exactly as required using the available tools
4. Submit the answer ONLY to the endpoint specified on the current page
5. After submitting, report the result including:
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

3. MEDIA ANALYSIS (IMPORTANT):
   - When you fetch audio/video/images using fetch_media, you will receive the actual media data
   - Listen to audio carefully and transcribe or extract the answer
   - For audio quizzes: the answer is usually SPOKEN in the audio - listen for names, numbers, words
   - Pay attention to the quiz question context when interpreting media

4. CODE EXECUTION:
   - Always use print() to output results
   - Handle errors gracefully

5. ANSWER SUBMISSION:
   - Extract the exact submission URL from the quiz page
   - Include all required fields in the payload
   - After POST, tell me the EXACT response including 'correct' and 'url' fields

6. COMPLETION:
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

    def _extract_submit_url(self, html_content: str) -> str | None:
        """Extract submit URL from HTML content."""
        # Common patterns for submit URLs
        patterns = [
            r'(?:post|submit).*?(?:to|url)[:\s]+["\']?(https?://[^\s"\'>]+)',
            r'(?:endpoint|api)[:\s]+["\']?(https?://[^\s"\'>]+/submit[^\s"\'>]*)',
            r'(https?://[^\s"\'>]+/submit[^\s"\'>]*)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, html_content, re.IGNORECASE)
            if match:
                return match.group(1)
        
        return None

    async def _tools_node(self, state: ProblemState) -> dict:
        """
        Custom tools node that intercepts post_request results to detect submissions.
        Also extracts submit URL from HTML content for soft timeout fallback.
        Uses async invocation to support async tools like get_rendered_html.
        """
        # Run the standard ToolNode asynchronously
        tool_node = ToolNode(TOOLS)
        result = await tool_node.ainvoke(state)
        
        # Check if any tool response contains submission result or HTML with submit URL
        messages = result.get("messages", [])
        updates = {}
        
        for msg in messages:
            if isinstance(msg, ToolMessage):
                content = getattr(msg, "content", "")
                
                # Check for submit URL in HTML content (for soft timeout fallback)
                if isinstance(content, str) and "<html" in content.lower():
                    submit_url = self._extract_submit_url(content)
                    if submit_url and not state.get("detected_submit_url"):
                        updates["detected_submit_url"] = submit_url
                        logger.info("submit_url_detected", url=submit_url)
                
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
        
        result.update(updates)
        return result

    def _preprocess_messages(self, messages: list) -> list:
        """Preprocess messages to convert media data to multimodal format."""
        processed = []
        
        media_prompts = {
            "audio": "Listen carefully to this audio. The answer to the quiz question is in this audio. Transcribe or extract the relevant information.",
            "image": "Look at this image carefully. Extract any relevant information needed to answer the quiz question.",
            "video": "Watch this video carefully. Extract any relevant information needed to answer the quiz question.",
            "document": "Read this PDF document carefully. Extract any relevant information needed to answer the quiz question.",
        }
        
        for msg in messages:
            if isinstance(msg, ToolMessage):
                content = getattr(msg, "content", "")
                media_data = extract_media_data(content)
                
                if media_data:
                    category, mime_type, base64_data = media_data
                    logger.info("converting_media_to_multimodal", category=category, mime_type=mime_type)
                    
                    processed.append(ToolMessage(
                        content=f"{category.capitalize()} data loaded. Analyze it to answer the quiz.",
                        tool_call_id=msg.tool_call_id,
                    ))
                    
                    prompt = media_prompts.get(category, f"Analyze this {category} to answer the quiz.")
                    multimodal_content = [
                        {"type": "text", "text": prompt},
                        {"type": "media", "data": base64_data, "mime_type": mime_type},
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
            logger.info("problem_solver_response", 
                       has_tool_calls=bool(getattr(result, "tool_calls", None)),
                       content_length=len(str(getattr(result, "content", ""))))
            return {"messages": [result]}
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

    async def solve(self, url: str, attempt_number: int = 1) -> ProblemResult:
        """
        Solve a single problem with fresh chat state.
        
        Args:
            url: Problem URL to solve
            attempt_number: Which attempt this is (for logging)
            
        Returns:
            ProblemResult with success/failure and next URL if any
        """
        logger.info("solving_problem", url=url, attempt=attempt_number)
        start_time = time.time()
        
        initial_state: ProblemState = {
            "messages": [HumanMessage(content=f"Solve the quiz at: {url}")],
            "problem_start_time": start_time,
            "problem_url": url,
            "workspace_path": str(self.workspace.root) if self.workspace else "",
            "submission_made": False,
            "submission_correct": None,
            "submission_next_url": None,
            "detected_submit_url": None,
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
                elapsed_seconds=elapsed,
                attempt_number=attempt_number,
            )
            
            logger.info("problem_solved", 
                       url=url, 
                       correct=problem_result.correct,
                       next_url=problem_result.next_url,
                       elapsed=elapsed,
                       attempt=attempt_number)
            
            return problem_result
            
        except asyncio.TimeoutError:
            elapsed = time.time() - start_time
            logger.error("problem_timeout", url=url, elapsed=elapsed, attempt=attempt_number)
            return ProblemResult(
                url=url,
                correct=False,
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
class QuizOrchestrator:
    """
    Orchestrates solving a quiz chain with parallel retry for failed problems.
    
    Strategy:
    1. Main thread: Process problems sequentially, spawn new ProblemSolver for each
    2. On failure: Add to retry queue with timestamp
    3. Retry thread: Run failed problems in parallel (each with 3-min timeout from first receive)
    4. Continue main thread while retries happen in background
    """

    def __init__(self, workspace: TaskWorkspace | None = None):
        self.workspace = workspace
        self.settings = get_settings()
        
        # Results tracking
        self.results: list[ProblemResult] = []
        self.retry_queue: asyncio.Queue[RetryItem] = asyncio.Queue()
        self.retry_results: list[ProblemResult] = []
        
        # Synchronization
        self.main_thread_done = asyncio.Event()
        self.retry_tasks: list[asyncio.Task] = []

    async def _retry_worker(self, item: RetryItem) -> ProblemResult | None:
        """
        Worker that retries a failed problem until 3-min timeout from first receive.
        """
        while True:
            # Check if we still have time (3 min from when problem was first received)
            elapsed = time.time() - item.start_time
            remaining = PROBLEM_TIMEOUT_SECONDS - elapsed
            
            if remaining <= 10:  # Less than 10 seconds left, not worth retrying
                logger.info("retry_time_expired", url=item.url, elapsed=elapsed)
                return None
            
            item.attempt_count += 1
            logger.info("retrying_problem", 
                       url=item.url, 
                       attempt=item.attempt_count,
                       remaining_seconds=remaining)
            
            # Create fresh solver for retry
            solver = ProblemSolver(workspace=self.workspace)
            
            try:
                # Solve with remaining time as timeout
                result = await asyncio.wait_for(
                    solver.solve(item.url, attempt_number=item.attempt_count),
                    timeout=remaining,
                )
                
                if result.correct:
                    logger.info("retry_succeeded", url=item.url, attempt=item.attempt_count)
                    return result
                else:
                    item.last_error = result.error
                    logger.info("retry_failed", url=item.url, attempt=item.attempt_count, error=result.error)
                    # Small delay before next retry
                    await asyncio.sleep(2)
                    
            except asyncio.TimeoutError:
                logger.info("retry_timeout", url=item.url, attempt=item.attempt_count)
                return None
            except Exception as e:
                logger.error("retry_error", url=item.url, error=str(e))
                item.last_error = str(e)
                await asyncio.sleep(2)
        
        return None

    async def _process_retry_queue(self):
        """Process the retry queue, running retries in parallel."""
        active_retries: dict[str, asyncio.Task] = {}
        
        while True:
            # Check if main thread is done and queue is empty
            if self.main_thread_done.is_set() and self.retry_queue.empty() and not active_retries:
                break
            
            # Start new retry tasks from queue
            try:
                while not self.retry_queue.empty():
                    item = self.retry_queue.get_nowait()
                    if item.url not in active_retries:
                        task = asyncio.create_task(self._retry_worker(item))
                        active_retries[item.url] = task
                        logger.info("started_retry_task", url=item.url)
            except asyncio.QueueEmpty:
                pass
            
            # Check completed retry tasks
            completed = []
            for url, task in active_retries.items():
                if task.done():
                    completed.append(url)
                    try:
                        result = task.result()
                        if result:
                            self.retry_results.append(result)
                    except Exception as e:
                        logger.error("retry_task_error", url=url, error=str(e))
            
            for url in completed:
                del active_retries[url]
            
            # Small sleep to prevent busy loop
            await asyncio.sleep(0.5)

    async def solve_chain(self, start_url: str) -> dict[str, Any]:
        """
        Solve the entire quiz chain starting from start_url.
        
        Returns:
            Summary of results including successes, failures, and retry results
        """
        logger.info("orchestrator_starting", start_url=start_url)
        
        # Start retry processor in background
        retry_processor = asyncio.create_task(self._process_retry_queue())
        
        current_url = start_url
        problem_count = 0
        
        try:
            while current_url:
                problem_count += 1
                problem_start_time = time.time()
                
                logger.info("processing_problem", 
                           problem_number=problem_count, 
                           url=current_url)
                
                # Create fresh solver for this problem
                solver = ProblemSolver(workspace=self.workspace)
                result = await solver.solve(current_url, attempt_number=1)
                self.results.append(result)
                
                if result.correct:
                    logger.info("problem_correct", 
                               problem_number=problem_count,
                               url=current_url)
                    current_url = result.next_url
                else:
                    logger.info("problem_incorrect_queuing_retry",
                               problem_number=problem_count,
                               url=current_url,
                               error=result.error)
                    
                    # Queue for retry with the time when we first received this problem
                    retry_item = RetryItem(
                        url=current_url,
                        start_time=problem_start_time,
                        attempt_count=1,
                        last_error=result.error,
                    )
                    await self.retry_queue.put(retry_item)
                    
                    # Continue with next URL if available
                    current_url = result.next_url
            
            logger.info("main_chain_complete", 
                       total_problems=problem_count,
                       correct_first_try=sum(1 for r in self.results if r.correct))
            
        finally:
            # Signal retry processor that main thread is done
            self.main_thread_done.set()
            
            # Wait for retry processor to finish
            await retry_processor
        
        # Compile final results
        first_try_correct = sum(1 for r in self.results if r.correct)
        retry_correct = len(self.retry_results)
        total_correct = first_try_correct + retry_correct
        
        summary = {
            "total_problems": problem_count,
            "correct_first_try": first_try_correct,
            "correct_after_retry": retry_correct,
            "total_correct": total_correct,
            "results": self.results,
            "retry_results": self.retry_results,
        }
        
        logger.info("orchestrator_complete",
                   total_problems=problem_count,
                   correct_first_try=first_try_correct,
                   correct_after_retry=retry_correct,
                   total_correct=total_correct)
        
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
