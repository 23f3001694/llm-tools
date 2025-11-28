"""
Task workspace management.
Creates isolated directories for each task with automatic cleanup.
"""

import shutil
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Generator

from .config import get_settings
from .logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class TaskWorkspace:
    """Represents an isolated workspace for a single task."""

    task_id: str
    base_path: Path
    created_at: datetime = field(default_factory=datetime.now)

    @property
    def root(self) -> Path:
        """Root directory for this task."""
        return self.base_path / self.task_id

    @property
    def downloads_dir(self) -> Path:
        """Directory for downloaded files."""
        return self.root / "downloads"

    @property
    def code_dir(self) -> Path:
        """Directory for generated code files."""
        return self.root / "code"

    @property
    def output_dir(self) -> Path:
        """Directory for output files (charts, etc.)."""
        return self.root / "output"

    @property
    def log_file(self) -> Path:
        """Path to the task-specific log file."""
        return self.root / "task.log"

    def setup(self) -> None:
        """Create all workspace directories."""
        self.root.mkdir(parents=True, exist_ok=True)
        self.downloads_dir.mkdir(exist_ok=True)
        self.code_dir.mkdir(exist_ok=True)
        self.output_dir.mkdir(exist_ok=True)
        logger.info("workspace_created", path=str(self.root))

    def cleanup(self) -> None:
        """Remove the workspace and all its contents."""
        if self.root.exists():
            shutil.rmtree(self.root, ignore_errors=True)
            logger.info("workspace_cleaned", path=str(self.root))

    def get_unique_filename(self, prefix: str, extension: str) -> Path:
        """Generate a unique filename in the downloads directory."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = uuid.uuid4().hex[:8]
        filename = f"{prefix}_{timestamp}_{unique_id}.{extension}"
        return self.downloads_dir / filename


@contextmanager
def create_task_workspace(
    task_id: str | None = None,
    cleanup_on_exit: bool = True,
) -> Generator[TaskWorkspace, None, None]:
    """
    Context manager that creates and optionally cleans up a task workspace.
    
    Args:
        task_id: Optional task ID. If not provided, a UUID will be generated.
        cleanup_on_exit: Whether to clean up the workspace on exit.
        
    Yields:
        TaskWorkspace instance with all directories created.
    """
    settings = get_settings()
    
    if task_id is None:
        task_id = str(uuid.uuid4())
    
    workspace = TaskWorkspace(
        task_id=task_id,
        base_path=Path(settings.task_base_dir),
    )
    
    try:
        workspace.setup()
        yield workspace
    finally:
        if cleanup_on_exit:
            workspace.cleanup()
        else:
            logger.info(
                "workspace_preserved",
                path=str(workspace.root),
                reason="cleanup_disabled",
            )


def cleanup_old_workspaces(max_age_hours: int = 24) -> int:
    """
    Clean up workspaces older than the specified age.
    
    Args:
        max_age_hours: Maximum age in hours before cleanup.
        
    Returns:
        Number of workspaces cleaned up.
    """
    settings = get_settings()
    base_path = Path(settings.task_base_dir)
    
    if not base_path.exists():
        return 0
    
    cleaned = 0
    cutoff = datetime.now().timestamp() - (max_age_hours * 3600)
    
    for workspace_dir in base_path.iterdir():
        if workspace_dir.is_dir():
            try:
                mtime = workspace_dir.stat().st_mtime
                if mtime < cutoff:
                    shutil.rmtree(workspace_dir, ignore_errors=True)
                    cleaned += 1
                    logger.info("old_workspace_cleaned", path=str(workspace_dir))
            except Exception as e:
                logger.warning(
                    "cleanup_failed",
                    path=str(workspace_dir),
                    error=str(e),
                )
    
    return cleaned
