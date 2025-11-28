# Project Plan: TDS Project 2 - Automatic Quiz Solver

## 1. Overview
Build an API endpoint that receives a task trigger, verifies identity, and autonomously solves data-related quizzes presented on web pages. The system must handle dynamic content (JS-rendered), perform data analysis/scraping, and submit answers within a strict time limit (3 minutes).

## 2. Architecture

### 2.1 Tech Stack
- **Language**: Python 3.x
- **Web Framework**: FastAPI (for high-performance async handling)
- **Server**: Uvicorn
- **Headless Browser**: Playwright (for rendering JS and scraping)
- **LLM Integration**: Multi-provider setup (OpenRouter, Gemini, Grok, etc.) for consensus-based reasoning.
- **Data Tools**: Pandas, NumPy, Requests/HTTPX, BeautifulSoup.
- **Environment Management**: `uv` (as seen in context) or `pip`.

### 2.2 System Components
1.  **API Gateway (FastAPI)**:
    -   Endpoint to receive the POST trigger.
    -   Authentication middleware (Secret verification).
    -   Background task dispatcher.
2.  **Task Orchestrator**:
    -   Manages the lifecycle of a quiz session.
    -   Handles the loop of: `Fetch Question` -> `Solve` -> `Submit` -> `Handle Response (Next URL/Retry)`.
3.  **Browser Agent**:
    -   Uses Playwright to navigate to the quiz URL.
    -   Extracts the DOM content after rendering.
    -   Identifies the question and submission URL.
4.  **Multi-Model Reasoning Engine**:
    -   Queries multiple models in parallel (e.g., GPT-4o, Gemini 1.5 Pro, Claude 3.5 Sonnet, Grok).
    -   **Consensus Mechanism**:
        -   Collects code/answers from all models.
        -   Executes code from each model to get a result.
        -   Determines the "Most Common Answer" (Mode).
        -   **Fallback**: If no consensus, defaults to the result from the "Most Powerful Model" (e.g., Gemini 1.5 Pro).
5.  **Execution Sandbox**:
    -   Executes generated Python code safely.
    -   Returns the result (answer) to the Orchestrator.

## 3. Implementation Steps

### Phase 1: Setup & API Skeleton
- [ ] Initialize project with `fastapi`, `uvicorn`, `playwright`, `httpx`, `python-dotenv`.
- [ ] Create `main.py` with a POST endpoint (e.g., `/run` or root `/`).
- [ ] Implement `Secret` verification logic.
    -   Return 200 OK for valid secret.
    -   Return 403 for invalid secret.
    -   Return 400 for bad JSON.
- [ ] Setup `BackgroundTasks` to trigger the solver asynchronously after sending the 200 OK response.

### Phase 2: Browser & Scraping Logic
- [ ] Install Playwright browsers (`playwright install`).
- [ ] Implement `BrowserAgent` class.
    -   Method `get_page_content(url)`: Returns text/HTML of the question.
    -   Must handle `atob` decoding if the question is obfuscated (as seen in the example).
    -   Extract the "Submit" URL and payload structure.

### Phase 2.5: Task Workspace Management (Isolation & Race Conditions)
- [ ] Implement per-task workspace creation and management.
    -   Implement `cleanup` on completion or a background sweeper that deletes task folders older than a retention window (e.g., 24–72 hours).
    -   Ensure every job runs in an isolated environment — the `BrowserAgent` (Playwright context), `CodeExecutor` (working dir), and temporary file downloads are all constrained to that directory.
    -   Prompt engineering: Standardized prompt for all models to ensure consistent output format.
- [ ] Implement `ConsensusManager`.
    -   **Result Aggregation**: Run the generated code from each model in the `CodeExecutor`.
    -   **Voting Logic**:
        -   Compare final answers (handling slight formatting differences).
        -   Select the majority winner.
        -   If tie/divergence, select the answer from the primary model (Gemini 1.5 Pro).
- [ ] Implement `CodeExecutor`.
    -   Run the LLM-generated code safely.
    -   Capture the output variable (the answer).
### Model output contract & normalization
- Standardize the prompt so every model returns a JSON object containing:
    - `code`: A Python code snippet that computes the answer (as a string). If code is unnecessary, set to `null`.
    - `answer`: The final parsed answer (number, string, boolean, JSON object, or base64 URI), if known.
    - `explanation`: Short human-readable explanation (optional).
    - `language`: Programming language for code (default `python`).
- The `SolverAgent` must validate the JSON before execution. If invalid, ask the model to reformat the response.
- Normalization steps:
    - Run model `code` in sandbox and capture the explicit `answer` output.
    - If model returns `answer` but no `code`, treat that as the model's asserted answer but still verify it if possible.
    - Normalize numeric representations (floats/ints), string trimming, case normalization for booleans and string outputs, JSON canonicalization.
    - Answers that are arrays or objects should be normalized by stable JSON serialization (sorted keys) before comparison.

    ### ModelClient interface & feedback prompt skeleton
    - `ModelClient` should have a small request wrapper API: `client.request(prompt, timeout=10)` returning parsed JSON.
    - Prompt skeleton for reconciliation (JSON text block):
    ```
    {
        "context": "<question and extracted context goes here>",
        "model_outputs": [{"model":"gpt4o","claimed_answer":...,"code":"...","explanation":"...","exec_result":{...}}, ...],
        "instructions": "Based on the provided execution results and other models' outputs, provide a final normalized answer, your confidence (0-1) and a short explanation in JSON format. Optionally include corrected code if needed."
    }
    ```
    The `SolverAgent` should perform a JSON-schema validation on the returned structure. If a model returns invalid JSON, ask it to reformat the response (once) before discarding.

### Aggregation / Voting logic
- After code-execution, collect `answer` from each model's output.
- Compute a frequency map (normalized answer -> count).
- If a single answer has a majority, use that answer for submission.
- If no majority exists, use the primary model (configurable; set `PRIMARY_MODEL = Gemini-1.5-Pro`) as the tiebreaker.
 - If no majority or if models disagree significantly (e.g., different data types or results), trigger the Model Reconciliation feedback loop before final selection.
 - Use `final_answer` and `confidence` returned from the reconciliation step; apply majority on `final_answer` or break ties using weighted confidence.
 - Log which models contributed to final answer and why (i.e., majority, confidence-weighted, or primary fallback).


### Implementation notes for sandbox & safety
- Prefer running generated code in a separate process (subprocess or multiprocessing) with resource limits and timeouts.
- Use `resource.setrlimit` (Unix) to limit CPU time and memory for executed code. On macOS, RLIMIT_AS or RLIMIT_DATA can be used to prevent OOM.
- Ensure the working directory of the executor is the per-task folder. Avoid specifying absolute paths that escape the workspace.

### Phase 3.5: Execution, Logging, and Cleanup
- [ ] Implement per-task logging: all console output and exceptions are captured into `task.log` within the task folder.
- [ ] Write the runtime result as `result.json` to the task folder for debugging and post-mortem.
- [ ] On success or final failure, call `cleanup_task_workspace(folder)` to free storage unless `DEBUG_KEEP_WORKSPACE` is set.

### Phase 3.6: Model Reconciliation (Feedback Loop)
- [ ] Implement a model reconciliation step where execution results and intermediate outputs are sent back to each model for a final consolidated answer.
    -   When to use: always run reconciliation when models disagree, or optionally always run it to gather a final confidence-weighted answer.
    -   Inputs to reconciliation:
        -   The original question and scraped/quasi-structured context.
        -   Each model's original output: the `code`, the model `answer` claim, and the model `explanation`.
        -   Execution results for each model's code (stdout, captured `answer`, exceptions, runtime) and the `result.json` from the task workspace.
        -   Normalized list of candidate answers along with counts.
    -   Behavior: Query all models (or a configurable subset) with a standardized feedback prompt; ask them to consider the execution results and produce a final answer + confidence and optional corrected code.
    -   Constraints: The feedback loop must be time limited so it does not exceed the 3-minute deadline.
    -   Output: Each model returns:
        -   `final_answer` (normalized scalar, string, or JSON),
        -   `confidence` (0-1 or percentage),
        -   `explanation` (short),
        -   `corrected_code` (optional) if it changed
    -   Use these final answers in the `ConsensusManager` to pick the submission answer (majority with confidence-weighting as a tiebreaker).
    -   In the event of a tie or lack of consensus, use the `PRIMARY_MODEL`'s `final_answer` as the fallback.

### Phase 5: Concurrency & Optimization

### Phase 4: Submission Loop
- [ ] Implement the submission logic.
    -   POST to the extracted submission URL.
    -   Payload: `email`, `secret`, `url`, `answer`.
- [ ] Handle the response:
    -   If `correct: true` and `url` present: **Recurse** (Start Phase 2 with new URL).
    -   If `correct: false`: **Retry** (Ask LLM to fix/re-evaluate) if time permits.
    -   If `correct: true` and no URL: **Finish**.

### Phase 5: Concurrency & Optimization
- [ ] Ensure the API is `async`.
- [ ] Use `asyncio` for non-blocking I/O (network requests, browser interaction).
- [ ] Ensure Playwright runs efficiently (reuse browser context if possible, or manage instances carefully).
- [ ] Implement timeout management (stop trying after 3 minutes).
- [ ] Implement a per-task folder policy to avoid collisions and cap per-task disk usage.
- [ ] Use an `asyncio.Semaphore` to limit the number of concurrent browsers and model queries.
- [ ] Consider reusing a single Playwright browser instance with distinct contexts per task to reduce overhead while ensuring isolation.

### Phase 6: Testing
- [ ] Unit tests for the API endpoint.
- [ ] Mock tests for the Solver (using the provided demo URL).
- [ ] Test with concurrent requests to ensure the server doesn't block.

## 4. Handling Multiple Requests
-   FastAPI is inherently asynchronous.
-   The `/run` endpoint will validate the request and immediately spawn a `BackgroundTask`.
-   This allows the server to respond 200 OK instantly and free up the worker to accept new requests.
-   The heavy processing (Browser, LLM) will run in the background.
-   We will use `async` libraries (`httpx`, `async_playwright`) to ensure the event loop isn't blocked.
-   For CPU-bound tasks (Pandas processing), we will run them in a thread pool or separate process if necessary (though for small datasets, standard execution is likely fine).

## 5. Configuration
-   `.env` file for:
    -   `AIPROXY_TOKEN` (for LLM).
    -   `USER_EMAIL`.
    -   `USER_SECRET`.
    -   `TASK_BASEDIR` (default `/tmp/tds_tasks`)
    -   `TASK_RETENTION_HOURS` (default `24`)
    -   `MAX_CONCURRENT_TASKS` (e.g., `4`)
    -   `DEBUG_KEEP_WORKSPACES` (true/false)

---

## Appendix: Recommended snippets

### Task workspace helpers (example)
```python
from pathlib import Path
import uuid
import shutil
import tempfile

def create_task_workspace(base_dir: str = '/tmp/tds_tasks', task_id: str | None = None) -> Path:
    if task_id is None:
        task_id = uuid.uuid4().hex
    base = Path(base_dir)
    base.mkdir(parents=True, exist_ok=True)
    folder = base / task_id
    folder.mkdir(parents=True, exist_ok=False)
    return folder

def cleanup_task_workspace(folder: Path, keep_on_debug: bool = False):
    if keep_on_debug:
        return
    shutil.rmtree(folder, ignore_errors=True)
```

### Running generated code in an isolated process (Unix sample)
```python
import subprocess
import resource
import os

def _limit_resources():
    # 1 second CPU, 200MB virtual memory
    resource.setrlimit(resource.RLIMIT_CPU, (10, 10))
    resource.setrlimit(resource.RLIMIT_AS, (200 * 1024 * 1024, 200 * 1024 * 1024))

def run_code_in_sandbox(code_path: str, cwd: str, timeout: int = 30):
    return subprocess.run(
        ["python3", code_path],
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=timeout,
        preexec_fn=_limit_resources,
    )
```

### Playwright contexts per task
```python
from playwright.async_api import async_playwright

async def run_task_with_browser(work_dir: Path, url: str):
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context()
        page = await context.new_page()
        await page.goto(url)
        # ... scrape, save artifacts to work_dir
        await context.close()
        await browser.close()
```

