"""
Sandboxed Python code execution with timeouts and resource limits.
"""

import os
import resource
import subprocess
import sys
import uuid
from pathlib import Path
from typing import TypedDict

from langchain_core.tools import tool

from ..config import get_settings
from ..logging_config import get_logger

logger = get_logger(__name__)

# Global workspace path - set by the agent when starting a task
_current_workspace_path: Path | None = None


def set_workspace_path(path: Path) -> None:
    """Set the current workspace path for code execution."""
    global _current_workspace_path
    _current_workspace_path = path


def get_workspace_path() -> Path:
    """Get the current workspace path, or fallback to LLMFiles."""
    if _current_workspace_path:
        return _current_workspace_path / "code"
    # Fallback for backwards compatibility
    fallback = Path("LLMFiles")
    fallback.mkdir(exist_ok=True)
    return fallback


class CodeExecutionResult(TypedDict):
    """Result of code execution."""
    stdout: str
    stderr: str
    return_code: int
    execution_time_seconds: float
    error: str | None


def _set_resource_limits() -> None:
    """
    Set resource limits for the subprocess.
    Called in the subprocess before exec.
    Note: May fail on macOS due to SIP, so we catch exceptions.
    """
    try:
        # Memory limit: 512MB
        memory_limit = 512 * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (memory_limit, memory_limit))
    except (ValueError, OSError):
        pass  # macOS may not support this
    
    try:
        # CPU time limit: 60 seconds
        cpu_limit = 60
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_limit, cpu_limit))
    except (ValueError, OSError):
        pass
    
    try:
        # Max file size: 50MB
        file_limit = 50 * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_FSIZE, (file_limit, file_limit))
    except (ValueError, OSError):
        pass
    
    try:
        # Max open files: 100
        resource.setrlimit(resource.RLIMIT_NOFILE, (100, 100))
    except (ValueError, OSError):
        pass


def _strip_code_fences(code: str) -> str:
    """Remove markdown code fences from code."""
    code = code.strip()
    
    # Remove ```python ... ``` or ``` ... ```
    if code.startswith("```"):
        lines = code.split("\n")
        # Remove first line (```python or ```)
        lines = lines[1:]
        # Remove last line if it's ```
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        code = "\n".join(lines)
    
    return code.strip()


@tool
def run_code(code: str) -> dict:
    """
    Execute Python code in a sandboxed environment with resource limits.

    This tool safely executes Python code by:
    1. Writing the code to a unique temporary file
    2. Running it in a subprocess with memory/CPU limits
    3. Returning stdout, stderr, and return code

    The code runs with access to the workspace's downloads directory,
    so it can read files downloaded by the download_file tool.

    IMPORTANT:
    - Code execution has a timeout (default 30 seconds)
    - Memory is limited to 512MB
    - Use print() to output results - the stdout will be returned
    - If you need additional packages, use the add_dependencies tool first

    Parameters
    ----------
    code : str
        Python source code to execute. Can include markdown code fences.

    Returns
    -------
    dict
        {
            "stdout": <program output>,
            "stderr": <errors if any>,
            "return_code": <exit code, 0 = success>,
            "execution_time_seconds": <time taken>,
            "error": <error message if execution failed, else None>
        }
    """
    settings = get_settings()
    timeout = settings.code_execution_timeout_seconds
    
    # Clean the code
    code = _strip_code_fences(code)
    
    if not code.strip():
        return {
            "stdout": "",
            "stderr": "Error: Empty code provided",
            "return_code": -1,
            "execution_time_seconds": 0,
            "error": "Empty code provided",
        }
    
    # Generate unique filename to avoid race conditions
    unique_id = uuid.uuid4().hex[:8]
    filename = f"runner_{unique_id}.py"
    
    # Get workspace directory
    code_dir = get_workspace_path()
    code_dir.mkdir(parents=True, exist_ok=True)
    filepath = code_dir / filename
    
    logger.info(
        "executing_code",
        filename=filename,
        code_length=len(code),
        timeout=timeout,
    )
    
    try:
        # Write code to file
        with open(filepath, "w") as f:
            f.write(code)
        
        # Determine the working directory
        # Use downloads dir if it exists (so code can access downloaded files)
        if _current_workspace_path:
            cwd = _current_workspace_path / "downloads"
            if not cwd.exists():
                cwd = _current_workspace_path
        else:
            cwd = code_dir
        
        # Prepare environment
        env = os.environ.copy()
        # Add the workspace to PYTHONPATH so imports work
        env["PYTHONPATH"] = str(code_dir) + ":" + env.get("PYTHONPATH", "")
        
        import time
        start_time = time.time()
        
        # Run the code with subprocess
        # Use 'uv run' if available, otherwise fall back to python
        # Note: preexec_fn can cause issues on some systems, so we skip it
        try:
            # Check if we're in a uv project
            proc = subprocess.Popen(
                ["uv", "run", "python", str(filepath)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=str(cwd),
                env=env,
            )
        except FileNotFoundError:
            # uv not available, use python directly
            proc = subprocess.Popen(
                [sys.executable, str(filepath)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=str(cwd),
                env=env,
            )
        
        try:
            stdout, stderr = proc.communicate(timeout=timeout)
            execution_time = time.time() - start_time
            
            logger.info(
                "code_executed",
                filename=filename,
                return_code=proc.returncode,
                execution_time=execution_time,
            )
            
            return {
                "stdout": stdout,
                "stderr": stderr,
                "return_code": proc.returncode,
                "execution_time_seconds": round(execution_time, 2),
                "error": None if proc.returncode == 0 else "Code exited with non-zero status",
            }
            
        except subprocess.TimeoutExpired:
            proc.kill()
            stdout, stderr = proc.communicate()
            execution_time = time.time() - start_time
            
            logger.warning(
                "code_timeout",
                filename=filename,
                timeout=timeout,
            )
            
            return {
                "stdout": stdout or "",
                "stderr": f"Execution timed out after {timeout} seconds.\n{stderr or ''}",
                "return_code": -1,
                "execution_time_seconds": round(execution_time, 2),
                "error": f"Execution timed out after {timeout} seconds",
            }
            
    except Exception as e:
        logger.error(
            "code_execution_failed",
            filename=filename,
            error=str(e),
        )
        return {
            "stdout": "",
            "stderr": str(e),
            "return_code": -1,
            "execution_time_seconds": 0,
            "error": str(e),
        }
    finally:
        # Clean up the temporary file
        try:
            filepath.unlink(missing_ok=True)
        except Exception:
            pass
