"""Streaming progress updates system"""

from typing import Dict, Any, List, Optional, Callable, AsyncIterator
from datetime import datetime
import asyncio
import json
from abc import ABC, abstractmethod
from .progress_tracker import ProgressTracker, ProgressEvent, ProgressStatus
import logfire


class StreamHandler(ABC):
    """Abstract base class for stream handlers"""
    
    @abstractmethod
    async def handle_event(self, event: ProgressEvent) -> None:
        """Handle a progress event
        
        Args:
            event: Progress event to handle
        """
        pass
    
    @abstractmethod
    async def close(self) -> None:
        """Close the stream handler"""
        pass


class ConsoleStreamHandler(StreamHandler):
    """Console output stream handler"""
    
    def __init__(self, verbose: bool = False):
        """Initialize console handler
        
        Args:
            verbose: Enable verbose output
        """
        self.verbose = verbose
        self._last_update: Dict[str, datetime] = {}
        self._update_interval = 0.1  # Minimum seconds between updates
    
    async def handle_event(self, event: ProgressEvent) -> None:
        """Handle event by printing to console"""
        # Rate limit updates
        now = datetime.utcnow()
        last_update = self._last_update.get(event.task_id)
        
        if event.status == ProgressStatus.RUNNING and last_update:
            if (now - last_update).total_seconds() < self._update_interval:
                return
        
        self._last_update[event.task_id] = now
        
        # Format message
        if event.status == ProgressStatus.COMPLETED:
            icon = "✅"
            color = "\033[32m"  # Green
        elif event.status == ProgressStatus.FAILED:
            icon = "❌"
            color = "\033[31m"  # Red
        elif event.status == ProgressStatus.RUNNING:
            icon = "🔄"
            color = "\033[34m"  # Blue
        elif event.status == ProgressStatus.CANCELLED:
            icon = "⛔"
            color = "\033[33m"  # Yellow
        else:
            icon = "⏸️"
            color = "\033[90m"  # Gray
        
        reset = "\033[0m"
        
        # Build message
        message = f"{color}{icon} [{event.task_id[:8]}] {event.message}{reset}"
        
        if self.verbose or event.status != ProgressStatus.RUNNING:
            # Add progress for running tasks
            if event.status == ProgressStatus.RUNNING:
                message += f" ({event.progress:.1f}%)"
            
            # Add error for failed tasks
            if event.error:
                message += f"\n   {color}Error: {event.error}{reset}"
            
            print(message)
    
    async def close(self) -> None:
        """Close console handler"""
        self._last_update.clear()


class WebSocketStreamHandler(StreamHandler):
    """WebSocket stream handler for real-time updates"""
    
    def __init__(self, websocket):
        """Initialize WebSocket handler
        
        Args:
            websocket: WebSocket connection
        """
        self.websocket = websocket
        self._logger = logfire.span("websocket_handler")
    
    async def handle_event(self, event: ProgressEvent) -> None:
        """Handle event by sending via WebSocket"""
        try:
            message = {
                "type": "progress",
                "event": {
                    "id": event.id,
                    "task_id": event.task_id,
                    "status": event.status,
                    "progress": event.progress,
                    "message": event.message,
                    "timestamp": event.timestamp.isoformat(),
                    "error": event.error,
                    "details": event.details
                }
            }
            
            await self.websocket.send(json.dumps(message))
            
        except Exception as e:
            self._logger.error(f"Failed to send WebSocket message: {e}")
    
    async def close(self) -> None:
        """Close WebSocket handler"""
        try:
            await self.websocket.close()
        except Exception:
            pass


class FileStreamHandler(StreamHandler):
    """File stream handler for logging progress"""
    
    def __init__(self, file_path: str):
        """Initialize file handler
        
        Args:
            file_path: Path to log file
        """
        self.file_path = file_path
        self._file = None
        self._logger = logfire.span("file_handler")
    
    async def handle_event(self, event: ProgressEvent) -> None:
        """Handle event by writing to file"""
        try:
            if not self._file:
                self._file = open(self.file_path, 'a')
            
            log_entry = {
                "timestamp": event.timestamp.isoformat(),
                "task_id": event.task_id,
                "status": event.status,
                "progress": event.progress,
                "message": event.message,
                "error": event.error,
                "details": event.details
            }
            
            self._file.write(json.dumps(log_entry) + "\n")
            self._file.flush()
            
        except Exception as e:
            self._logger.error(f"Failed to write to file: {e}")
    
    async def close(self) -> None:
        """Close file handler"""
        if self._file:
            self._file.close()
            self._file = None


class CallbackStreamHandler(StreamHandler):
    """Callback-based stream handler"""
    
    def __init__(self, callback: Callable[[ProgressEvent], None]):
        """Initialize callback handler
        
        Args:
            callback: Callback function
        """
        self.callback = callback
    
    async def handle_event(self, event: ProgressEvent) -> None:
        """Handle event by calling callback"""
        if asyncio.iscoroutinefunction(self.callback):
            await self.callback(event)
        else:
            self.callback(event)
    
    async def close(self) -> None:
        """Close callback handler"""
        pass


class StreamingProgress:
    """Streaming progress update system"""
    
    def __init__(self, tracker: ProgressTracker):
        """Initialize streaming progress
        
        Args:
            tracker: Progress tracker instance
        """
        self.tracker = tracker
        self._handlers: List[StreamHandler] = []
        self._streaming = False
        self._logger = logfire.span("streaming_progress")
        
        # Register as listener
        self.tracker.add_listener(self._on_progress_event)
    
    def add_handler(self, handler: StreamHandler) -> None:
        """Add a stream handler
        
        Args:
            handler: Stream handler to add
        """
        self._handlers.append(handler)
        self._logger.info(f"Added handler: {type(handler).__name__}")
    
    def remove_handler(self, handler: StreamHandler) -> None:
        """Remove a stream handler
        
        Args:
            handler: Stream handler to remove
        """
        if handler in self._handlers:
            self._handlers.remove(handler)
            self._logger.info(f"Removed handler: {type(handler).__name__}")
    
    async def start_streaming(self) -> None:
        """Start streaming progress updates"""
        self._streaming = True
        self._logger.info("Started streaming")
    
    async def stop_streaming(self) -> None:
        """Stop streaming progress updates"""
        self._streaming = False
        
        # Close all handlers
        for handler in self._handlers:
            try:
                await handler.close()
            except Exception as e:
                self._logger.error(f"Error closing handler: {e}")
        
        self._logger.info("Stopped streaming")
    
    async def _on_progress_event(self, event: ProgressEvent) -> None:
        """Handle progress event from tracker"""
        if not self._streaming:
            return
        
        # Send to all handlers
        tasks = []
        for handler in self._handlers:
            tasks.append(self._handle_event_safe(handler, event))
        
        if tasks:
            await asyncio.gather(*tasks)
    
    async def _handle_event_safe(self, handler: StreamHandler, event: ProgressEvent) -> None:
        """Safely handle event with a handler"""
        try:
            await handler.handle_event(event)
        except Exception as e:
            self._logger.error(
                f"Handler error in {type(handler).__name__}: {e}",
                handler=type(handler).__name__,
                error=str(e)
            )
    
    async def stream_task_progress(
        self,
        task_id: str,
        handler: Optional[StreamHandler] = None
    ) -> AsyncIterator[ProgressEvent]:
        """Stream progress for a specific task
        
        Args:
            task_id: Task to stream
            handler: Optional additional handler
            
        Yields:
            Progress events
        """
        # Add temporary handler if provided
        if handler:
            self.add_handler(handler)
        
        try:
            # Create event queue
            event_queue = asyncio.Queue()
            
            # Create temporary handler to capture events
            async def capture_event(event: ProgressEvent):
                if event.task_id == task_id:
                    await event_queue.put(event)
            
            capture_handler = CallbackStreamHandler(capture_event)
            self.add_handler(capture_handler)
            
            # Start streaming
            await self.start_streaming()
            
            # Stream events
            task = self.tracker.get_task(task_id)
            while task and not task.is_complete:
                try:
                    # Wait for event with timeout
                    event = await asyncio.wait_for(
                        event_queue.get(),
                        timeout=1.0
                    )
                    yield event
                    
                except asyncio.TimeoutError:
                    # Check if task is still active
                    task = self.tracker.get_task(task_id)
                    
                except Exception as e:
                    self._logger.error(f"Stream error: {e}")
                    break
            
            # Yield final event
            task = self.tracker.get_task(task_id)
            if task and task.events:
                yield task.events[-1]
                
        finally:
            # Clean up
            self.remove_handler(capture_handler)
            if handler:
                self.remove_handler(handler)
    
    def create_console_handler(self, verbose: bool = False) -> ConsoleStreamHandler:
        """Create a console stream handler
        
        Args:
            verbose: Enable verbose output
            
        Returns:
            Console handler
        """
        return ConsoleStreamHandler(verbose)
    
    def create_file_handler(self, file_path: str) -> FileStreamHandler:
        """Create a file stream handler
        
        Args:
            file_path: Log file path
            
        Returns:
            File handler
        """
        return FileStreamHandler(file_path)
    
    def create_callback_handler(
        self,
        callback: Callable[[ProgressEvent], None]
    ) -> CallbackStreamHandler:
        """Create a callback stream handler
        
        Args:
            callback: Callback function
            
        Returns:
            Callback handler
        """
        return CallbackStreamHandler(callback)
    
    async def broadcast_message(
        self,
        message: str,
        task_id: Optional[str] = None,
        level: str = "info"
    ) -> None:
        """Broadcast a custom message to all handlers
        
        Args:
            message: Message to broadcast
            task_id: Associated task ID
            level: Message level (info, warning, error)
        """
        # Create a synthetic event
        event = ProgressEvent(
            task_id=task_id or "broadcast",
            status=ProgressStatus.RUNNING,
            message=message,
            metadata={"level": level, "broadcast": True}
        )
        
        await self._on_progress_event(event)