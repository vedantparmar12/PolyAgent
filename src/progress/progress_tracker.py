"""Progress tracking system for agent operations"""

from typing import Dict, Any, List, Optional, Callable
from datetime import datetime, timedelta
from enum import Enum
from pydantic import BaseModel, Field
import asyncio
from collections import defaultdict
import uuid
import logfire


class ProgressStatus(str, Enum):
    """Progress status types"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"


class ProgressEvent(BaseModel):
    """Progress event model"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    task_id: str = Field(description="Task identifier")
    status: ProgressStatus = Field(description="Current status")
    progress: float = Field(default=0.0, ge=0.0, le=100.0, description="Progress percentage")
    message: str = Field(default="", description="Progress message")
    details: Dict[str, Any] = Field(default_factory=dict, description="Additional details")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    duration: Optional[timedelta] = Field(default=None, description="Duration so far")
    error: Optional[str] = Field(default=None, description="Error message if failed")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Event metadata")


class TaskProgress(BaseModel):
    """Task progress tracking"""
    task_id: str
    name: str
    status: ProgressStatus
    progress: float = 0.0
    start_time: datetime
    end_time: Optional[datetime] = None
    parent_id: Optional[str] = None
    children: List[str] = Field(default_factory=list)
    events: List[ProgressEvent] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    @property
    def duration(self) -> timedelta:
        """Calculate task duration"""
        end = self.end_time or datetime.utcnow()
        return end - self.start_time
    
    @property
    def is_complete(self) -> bool:
        """Check if task is complete"""
        return self.status in [ProgressStatus.COMPLETED, ProgressStatus.FAILED, ProgressStatus.CANCELLED]


class ProgressTracker:
    """Central progress tracking system"""
    
    def __init__(self):
        """Initialize progress tracker"""
        self._tasks: Dict[str, TaskProgress] = {}
        self._listeners: List[Callable[[ProgressEvent], None]] = []
        self._logger = logfire.span("progress_tracker")
        self._lock = asyncio.Lock()
        
        # Analytics data
        self._task_stats = defaultdict(lambda: {
            "total": 0,
            "completed": 0,
            "failed": 0,
            "cancelled": 0,
            "total_duration": timedelta(),
            "avg_duration": timedelta()
        })
    
    async def create_task(
        self,
        task_id: str,
        name: str,
        parent_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> TaskProgress:
        """Create a new task
        
        Args:
            task_id: Unique task identifier
            name: Task name
            parent_id: Parent task ID for subtasks
            metadata: Task metadata
            
        Returns:
            Created task progress
        """
        async with self._lock:
            if task_id in self._tasks:
                raise ValueError(f"Task {task_id} already exists")
            
            task = TaskProgress(
                task_id=task_id,
                name=name,
                status=ProgressStatus.PENDING,
                start_time=datetime.utcnow(),
                parent_id=parent_id,
                metadata=metadata or {}
            )
            
            self._tasks[task_id] = task
            
            # Update parent if exists
            if parent_id and parent_id in self._tasks:
                self._tasks[parent_id].children.append(task_id)
            
            # Create initial event
            event = ProgressEvent(
                task_id=task_id,
                status=ProgressStatus.PENDING,
                message=f"Task '{name}' created"
            )
            task.events.append(event)
            
            # Notify listeners
            await self._notify_listeners(event)
            
            self._logger.info(
                "Task created",
                task_id=task_id,
                name=name,
                parent_id=parent_id
            )
            
            return task
    
    async def update_progress(
        self,
        task_id: str,
        progress: float,
        message: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ) -> ProgressEvent:
        """Update task progress
        
        Args:
            task_id: Task identifier
            progress: Progress percentage (0-100)
            message: Progress message
            details: Additional details
            
        Returns:
            Progress event
        """
        async with self._lock:
            if task_id not in self._tasks:
                raise ValueError(f"Task {task_id} not found")
            
            task = self._tasks[task_id]
            
            # Update task progress
            task.progress = progress
            if task.status == ProgressStatus.PENDING:
                task.status = ProgressStatus.RUNNING
            
            # Create event
            event = ProgressEvent(
                task_id=task_id,
                status=task.status,
                progress=progress,
                message=message or f"Progress: {progress:.1f}%",
                details=details or {},
                duration=task.duration
            )
            task.events.append(event)
            
            # Update parent progress
            if task.parent_id:
                await self._update_parent_progress(task.parent_id)
            
            # Notify listeners
            await self._notify_listeners(event)
            
            return event
    
    async def complete_task(
        self,
        task_id: str,
        message: Optional[str] = None,
        result: Optional[Dict[str, Any]] = None
    ) -> ProgressEvent:
        """Mark task as completed
        
        Args:
            task_id: Task identifier
            message: Completion message
            result: Task result
            
        Returns:
            Progress event
        """
        async with self._lock:
            if task_id not in self._tasks:
                raise ValueError(f"Task {task_id} not found")
            
            task = self._tasks[task_id]
            task.status = ProgressStatus.COMPLETED
            task.progress = 100.0
            task.end_time = datetime.utcnow()
            
            # Update statistics
            self._update_task_stats(task)
            
            # Create event
            event = ProgressEvent(
                task_id=task_id,
                status=ProgressStatus.COMPLETED,
                progress=100.0,
                message=message or f"Task '{task.name}' completed",
                details=result or {},
                duration=task.duration
            )
            task.events.append(event)
            
            # Update parent progress
            if task.parent_id:
                await self._update_parent_progress(task.parent_id)
            
            # Notify listeners
            await self._notify_listeners(event)
            
            self._logger.info(
                "Task completed",
                task_id=task_id,
                duration=str(task.duration)
            )
            
            return event
    
    async def fail_task(
        self,
        task_id: str,
        error: str,
        details: Optional[Dict[str, Any]] = None
    ) -> ProgressEvent:
        """Mark task as failed
        
        Args:
            task_id: Task identifier
            error: Error message
            details: Error details
            
        Returns:
            Progress event
        """
        async with self._lock:
            if task_id not in self._tasks:
                raise ValueError(f"Task {task_id} not found")
            
            task = self._tasks[task_id]
            task.status = ProgressStatus.FAILED
            task.end_time = datetime.utcnow()
            
            # Update statistics
            self._update_task_stats(task)
            
            # Create event
            event = ProgressEvent(
                task_id=task_id,
                status=ProgressStatus.FAILED,
                progress=task.progress,
                message=f"Task '{task.name}' failed",
                error=error,
                details=details or {},
                duration=task.duration
            )
            task.events.append(event)
            
            # Fail children tasks
            for child_id in task.children:
                if child_id in self._tasks and not self._tasks[child_id].is_complete:
                    await self.fail_task(
                        child_id,
                        f"Parent task {task_id} failed",
                        {"parent_error": error}
                    )
            
            # Update parent
            if task.parent_id:
                await self._update_parent_progress(task.parent_id)
            
            # Notify listeners
            await self._notify_listeners(event)
            
            self._logger.error(
                "Task failed",
                task_id=task_id,
                error=error
            )
            
            return event
    
    async def cancel_task(
        self,
        task_id: str,
        reason: Optional[str] = None
    ) -> ProgressEvent:
        """Cancel a task
        
        Args:
            task_id: Task identifier
            reason: Cancellation reason
            
        Returns:
            Progress event
        """
        async with self._lock:
            if task_id not in self._tasks:
                raise ValueError(f"Task {task_id} not found")
            
            task = self._tasks[task_id]
            
            if task.is_complete:
                raise ValueError(f"Cannot cancel completed task {task_id}")
            
            task.status = ProgressStatus.CANCELLED
            task.end_time = datetime.utcnow()
            
            # Update statistics
            self._update_task_stats(task)
            
            # Create event
            event = ProgressEvent(
                task_id=task_id,
                status=ProgressStatus.CANCELLED,
                progress=task.progress,
                message=f"Task '{task.name}' cancelled",
                details={"reason": reason} if reason else {},
                duration=task.duration
            )
            task.events.append(event)
            
            # Cancel children tasks
            for child_id in task.children:
                if child_id in self._tasks and not self._tasks[child_id].is_complete:
                    await self.cancel_task(child_id, f"Parent task {task_id} cancelled")
            
            # Update parent
            if task.parent_id:
                await self._update_parent_progress(task.parent_id)
            
            # Notify listeners
            await self._notify_listeners(event)
            
            return event
    
    async def pause_task(self, task_id: str) -> ProgressEvent:
        """Pause a running task
        
        Args:
            task_id: Task identifier
            
        Returns:
            Progress event
        """
        async with self._lock:
            if task_id not in self._tasks:
                raise ValueError(f"Task {task_id} not found")
            
            task = self._tasks[task_id]
            
            if task.status != ProgressStatus.RUNNING:
                raise ValueError(f"Can only pause running tasks")
            
            task.status = ProgressStatus.PAUSED
            
            # Create event
            event = ProgressEvent(
                task_id=task_id,
                status=ProgressStatus.PAUSED,
                progress=task.progress,
                message=f"Task '{task.name}' paused",
                duration=task.duration
            )
            task.events.append(event)
            
            # Notify listeners
            await self._notify_listeners(event)
            
            return event
    
    async def resume_task(self, task_id: str) -> ProgressEvent:
        """Resume a paused task
        
        Args:
            task_id: Task identifier
            
        Returns:
            Progress event
        """
        async with self._lock:
            if task_id not in self._tasks:
                raise ValueError(f"Task {task_id} not found")
            
            task = self._tasks[task_id]
            
            if task.status != ProgressStatus.PAUSED:
                raise ValueError(f"Can only resume paused tasks")
            
            task.status = ProgressStatus.RUNNING
            
            # Create event
            event = ProgressEvent(
                task_id=task_id,
                status=ProgressStatus.RUNNING,
                progress=task.progress,
                message=f"Task '{task.name}' resumed",
                duration=task.duration
            )
            task.events.append(event)
            
            # Notify listeners
            await self._notify_listeners(event)
            
            return event
    
    def get_task(self, task_id: str) -> Optional[TaskProgress]:
        """Get task by ID
        
        Args:
            task_id: Task identifier
            
        Returns:
            Task progress or None
        """
        return self._tasks.get(task_id)
    
    def get_all_tasks(self) -> List[TaskProgress]:
        """Get all tasks
        
        Returns:
            List of all tasks
        """
        return list(self._tasks.values())
    
    def get_active_tasks(self) -> List[TaskProgress]:
        """Get active tasks
        
        Returns:
            List of active tasks
        """
        return [
            task for task in self._tasks.values()
            if task.status in [ProgressStatus.RUNNING, ProgressStatus.PAUSED]
        ]
    
    def get_task_tree(self, root_id: Optional[str] = None) -> Dict[str, Any]:
        """Get task hierarchy tree
        
        Args:
            root_id: Root task ID (None for all roots)
            
        Returns:
            Task tree structure
        """
        def build_tree(task_id: str) -> Dict[str, Any]:
            task = self._tasks[task_id]
            return {
                "id": task.task_id,
                "name": task.name,
                "status": task.status,
                "progress": task.progress,
                "duration": str(task.duration),
                "children": [build_tree(child_id) for child_id in task.children]
            }
        
        if root_id:
            return build_tree(root_id)
        
        # Find root tasks (no parent)
        roots = [
            task for task in self._tasks.values()
            if task.parent_id is None
        ]
        
        return {
            "roots": [build_tree(task.task_id) for task in roots]
        }
    
    def add_listener(self, listener: Callable[[ProgressEvent], None]) -> None:
        """Add progress event listener
        
        Args:
            listener: Event listener function
        """
        self._listeners.append(listener)
    
    def remove_listener(self, listener: Callable[[ProgressEvent], None]) -> None:
        """Remove progress event listener
        
        Args:
            listener: Event listener function
        """
        if listener in self._listeners:
            self._listeners.remove(listener)
    
    async def _notify_listeners(self, event: ProgressEvent) -> None:
        """Notify all listeners of an event"""
        for listener in self._listeners:
            try:
                if asyncio.iscoroutinefunction(listener):
                    await listener(event)
                else:
                    listener(event)
            except Exception as e:
                self._logger.error(f"Listener error: {e}")
    
    async def _update_parent_progress(self, parent_id: str) -> None:
        """Update parent task progress based on children"""
        parent = self._tasks.get(parent_id)
        if not parent:
            return
        
        if not parent.children:
            return
        
        # Calculate aggregate progress
        total_progress = 0.0
        completed_count = 0
        
        for child_id in parent.children:
            child = self._tasks.get(child_id)
            if child:
                total_progress += child.progress
                if child.is_complete:
                    completed_count += 1
        
        # Update parent progress
        parent.progress = total_progress / len(parent.children)
        
        # Update parent status
        if completed_count == len(parent.children):
            # All children complete
            failed_children = [
                self._tasks[cid] for cid in parent.children
                if self._tasks[cid].status == ProgressStatus.FAILED
            ]
            
            if failed_children:
                parent.status = ProgressStatus.FAILED
            else:
                parent.status = ProgressStatus.COMPLETED
            
            parent.end_time = datetime.utcnow()
    
    def _update_task_stats(self, task: TaskProgress) -> None:
        """Update task statistics"""
        stats = self._task_stats[task.name]
        stats["total"] += 1
        
        if task.status == ProgressStatus.COMPLETED:
            stats["completed"] += 1
        elif task.status == ProgressStatus.FAILED:
            stats["failed"] += 1
        elif task.status == ProgressStatus.CANCELLED:
            stats["cancelled"] += 1
        
        # Update duration stats
        if task.duration:
            stats["total_duration"] += task.duration
            stats["avg_duration"] = stats["total_duration"] / stats["total"]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get progress tracking statistics
        
        Returns:
            Statistics dictionary
        """
        active_tasks = self.get_active_tasks()
        all_tasks = self.get_all_tasks()
        
        return {
            "total_tasks": len(all_tasks),
            "active_tasks": len(active_tasks),
            "completed_tasks": len([t for t in all_tasks if t.status == ProgressStatus.COMPLETED]),
            "failed_tasks": len([t for t in all_tasks if t.status == ProgressStatus.FAILED]),
            "cancelled_tasks": len([t for t in all_tasks if t.status == ProgressStatus.CANCELLED]),
            "task_stats": dict(self._task_stats),
            "active_task_names": [t.name for t in active_tasks]
        }
    
    def clear_completed_tasks(self, older_than: Optional[timedelta] = None) -> int:
        """Clear completed tasks
        
        Args:
            older_than: Only clear tasks older than this duration
            
        Returns:
            Number of tasks cleared
        """
        cutoff_time = None
        if older_than:
            cutoff_time = datetime.utcnow() - older_than
        
        tasks_to_remove = []
        
        for task_id, task in self._tasks.items():
            if task.is_complete:
                if not cutoff_time or (task.end_time and task.end_time < cutoff_time):
                    # Don't remove if it has active children
                    has_active_children = any(
                        child_id in self._tasks and not self._tasks[child_id].is_complete
                        for child_id in task.children
                    )
                    if not has_active_children:
                        tasks_to_remove.append(task_id)
        
        # Remove tasks
        for task_id in tasks_to_remove:
            task = self._tasks[task_id]
            
            # Remove from parent's children list
            if task.parent_id and task.parent_id in self._tasks:
                parent = self._tasks[task.parent_id]
                if task_id in parent.children:
                    parent.children.remove(task_id)
            
            del self._tasks[task_id]
        
        return len(tasks_to_remove)