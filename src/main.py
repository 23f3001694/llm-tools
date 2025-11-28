"""
FastAPI server for the quiz-solving agent.
Handles incoming requests with proper validation, timeouts, and error handling.
"""

import asyncio
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import BackgroundTasks, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, EmailStr, Field

from .agent import QuizAgent
from .config import get_settings
from .logging_config import get_logger, setup_logging
from .tools.web_scraper import cleanup_browser
from .workspace import TaskWorkspace, cleanup_old_workspaces, create_task_workspace

# Initialize logging
setup_logging()
logger = get_logger(__name__)


# -------------------------------------------------
# LIFESPAN MANAGEMENT
# -------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application startup and shutdown."""
    # Startup
    logger.info("server_starting")
    
    # Clean up old workspaces from previous runs
    cleaned = cleanup_old_workspaces(max_age_hours=24)
    if cleaned:
        logger.info("cleaned_old_workspaces", count=cleaned)
    
    yield
    
    # Shutdown
    logger.info("server_shutting_down")
    await cleanup_browser()
    logger.info("server_stopped")


# -------------------------------------------------
# FASTAPI APP
# -------------------------------------------------
app = FastAPI(
    title="TDS Quiz Solver",
    description="Automated quiz-solving agent with multi-model LLM support",
    version="0.2.0",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -------------------------------------------------
# REQUEST/RESPONSE MODELS
# -------------------------------------------------
class SolveRequest(BaseModel):
    """Request model for the /solve endpoint."""
    email: EmailStr = Field(..., description="Student email ID")
    secret: str = Field(..., min_length=1, description="Student-provided secret")
    url: str = Field(..., min_length=1, description="Quiz URL to solve")


class SolveResponse(BaseModel):
    """Response model for the /solve endpoint."""
    status: str
    task_id: str
    message: str


class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str
    version: str
    providers: list[str]


# -------------------------------------------------
# TASK TRACKING
# -------------------------------------------------
# Track active tasks for status queries
active_tasks: dict[str, dict[str, Any]] = {}


async def run_task(task_id: str, url: str, workspace: TaskWorkspace) -> None:
    """
    Run the quiz-solving task with timeout and error handling.
    
    Args:
        task_id: Unique task identifier
        url: Quiz URL to solve
        workspace: Task workspace for file isolation
    """
    settings = get_settings()
    
    active_tasks[task_id] = {
        "status": "running",
        "url": url,
        "error": None,
    }
    
    try:
        # Create agent with workspace
        agent = QuizAgent(workspace=workspace)
        
        # Run with timeout
        logger.info("task_starting", task_id=task_id, url=url)
        
        result = await asyncio.wait_for(
            agent.arun(url),
            timeout=settings.task_timeout_seconds,
        )
        
        active_tasks[task_id] = {
            "status": "completed",
            "url": url,
            "error": None,
            "message_count": len(result.get("messages", [])),
        }
        
        logger.info(
            "task_completed",
            task_id=task_id,
            message_count=len(result.get("messages", [])),
        )
        
    except asyncio.TimeoutError:
        active_tasks[task_id] = {
            "status": "timeout",
            "url": url,
            "error": f"Task timed out after {settings.task_timeout_seconds} seconds",
        }
        logger.error(
            "task_timeout",
            task_id=task_id,
            timeout=settings.task_timeout_seconds,
        )
        
    except Exception as e:
        active_tasks[task_id] = {
            "status": "failed",
            "url": url,
            "error": str(e),
        }
        logger.error(
            "task_failed",
            task_id=task_id,
            error=str(e),
        )
        
    finally:
        # Clean up workspace
        try:
            workspace.cleanup()
        except Exception as e:
            logger.warning(
                "workspace_cleanup_failed",
                task_id=task_id,
                error=str(e),
            )


# -------------------------------------------------
# ENDPOINTS
# -------------------------------------------------
@app.get("/healthz", response_model=HealthResponse)
async def healthz() -> HealthResponse:
    """Health check endpoint."""
    return HealthResponse(
        status="ok",
        version="0.2.0",
        providers=["gemini-2.5-flash"],
    )


@app.post("/solve", response_model=SolveResponse)
async def solve(
    request: Request,
    background_tasks: BackgroundTasks,
) -> SolveResponse:
    """
    Start solving a quiz.
    
    Validates the request, creates an isolated workspace, and starts
    the agent in a background task.
    """
    settings = get_settings()
    
    # Parse and validate request
    try:
        data = await request.json()
    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Invalid JSON in request body",
        )
    
    # Validate required fields
    try:
        solve_request = SolveRequest(**data)
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid request: {str(e)}",
        )
    
    # Verify secret
    if solve_request.secret != settings.secret:
        logger.warning(
            "invalid_secret",
            email=solve_request.email,
        )
        raise HTTPException(
            status_code=403,
            detail="Invalid secret",
        )
    
    # Generate task ID and create workspace
    task_id = str(uuid.uuid4())
    
    logger.info(
        "solve_request_received",
        task_id=task_id,
        email=solve_request.email,
        url=solve_request.url,
    )
    
    # Create workspace (don't use context manager since background task manages lifecycle)
    workspace = TaskWorkspace(
        task_id=task_id,
        base_path=Path(settings.task_base_dir),
    )
    workspace.setup()
    
    # Start background task
    background_tasks.add_task(
        run_task,
        task_id=task_id,
        url=solve_request.url,
        workspace=workspace,
    )
    
    return SolveResponse(
        status="ok",
        task_id=task_id,
        message="Task started successfully",
    )


@app.get("/task/{task_id}")
async def get_task_status(task_id: str) -> dict[str, Any]:
    """Get the status of a running task."""
    if task_id not in active_tasks:
        raise HTTPException(
            status_code=404,
            detail=f"Task {task_id} not found",
        )
    
    return active_tasks[task_id]


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handle uncaught exceptions."""
    logger.error(
        "unhandled_exception",
        path=request.url.path,
        method=request.method,
        error=str(exc),
    )
    
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"},
    )


# -------------------------------------------------
# MAIN
# -------------------------------------------------
def main() -> None:
    """Run the server."""
    settings = get_settings()
    
    logger.info(
        "starting_server",
        host=settings.host,
        port=settings.port,
    )
    
    uvicorn.run(
        "src.main:app",
        host=settings.host,
        port=settings.port,
        reload=False,
        log_level=settings.log_level.lower(),
    )


if __name__ == "__main__":
    main()
