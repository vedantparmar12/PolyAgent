"""Progress visualization components"""

from typing import Dict, Any, List, Optional
from enum import Enum
from datetime import datetime
import json
from .progress_tracker import ProgressTracker, TaskProgress, ProgressStatus


class VisualizationType(str, Enum):
    """Visualization types"""
    TEXT = "text"
    JSON = "json"
    TREE = "tree"
    TIMELINE = "timeline"
    GANTT = "gantt"
    PROGRESS_BAR = "progress_bar"


class ProgressVisualizer:
    """Progress visualization system"""
    
    def __init__(self, tracker: ProgressTracker):
        """Initialize visualizer
        
        Args:
            tracker: Progress tracker instance
        """
        self.tracker = tracker
    
    def visualize(
        self,
        viz_type: VisualizationType = VisualizationType.TEXT,
        task_id: Optional[str] = None,
        include_completed: bool = True
    ) -> str:
        """Generate visualization
        
        Args:
            viz_type: Type of visualization
            task_id: Specific task to visualize (None for all)
            include_completed: Include completed tasks
            
        Returns:
            Visualization string
        """
        if viz_type == VisualizationType.TEXT:
            return self._visualize_text(task_id, include_completed)
        elif viz_type == VisualizationType.JSON:
            return self._visualize_json(task_id, include_completed)
        elif viz_type == VisualizationType.TREE:
            return self._visualize_tree(task_id, include_completed)
        elif viz_type == VisualizationType.TIMELINE:
            return self._visualize_timeline(task_id, include_completed)
        elif viz_type == VisualizationType.GANTT:
            return self._visualize_gantt(task_id, include_completed)
        elif viz_type == VisualizationType.PROGRESS_BAR:
            return self._visualize_progress_bar(task_id)
        else:
            raise ValueError(f"Unknown visualization type: {viz_type}")
    
    def _visualize_text(self, task_id: Optional[str], include_completed: bool) -> str:
        """Generate text visualization"""
        lines = []
        
        if task_id:
            task = self.tracker.get_task(task_id)
            if task:
                lines.extend(self._format_task_text(task, 0))
        else:
            # Show all root tasks
            all_tasks = self.tracker.get_all_tasks()
            root_tasks = [t for t in all_tasks if t.parent_id is None]
            
            for task in root_tasks:
                if include_completed or not task.is_complete:
                    lines.extend(self._format_task_text(task, 0))
        
        return "\n".join(lines)
    
    def _format_task_text(self, task: TaskProgress, indent: int) -> List[str]:
        """Format task as text"""
        lines = []
        prefix = "  " * indent
        
        # Status icon
        status_icon = {
            ProgressStatus.PENDING: "⏸️",
            ProgressStatus.RUNNING: "▶️",
            ProgressStatus.COMPLETED: "✅",
            ProgressStatus.FAILED: "❌",
            ProgressStatus.CANCELLED: "⛔",
            ProgressStatus.PAUSED: "⏸️"
        }.get(task.status, "❓")
        
        # Progress bar
        progress_bar = self._create_progress_bar(task.progress, 20)
        
        # Main line
        lines.append(
            f"{prefix}{status_icon} {task.name} {progress_bar} {task.progress:.1f}% "
            f"[{task.status}] ({task.duration})"
        )
        
        # Add recent event if any
        if task.events:
            recent_event = task.events[-1]
            if recent_event.message:
                lines.append(f"{prefix}  └─ {recent_event.message}")
        
        # Add children
        for child_id in task.children:
            child = self.tracker.get_task(child_id)
            if child:
                lines.extend(self._format_task_text(child, indent + 1))
        
        return lines
    
    def _visualize_json(self, task_id: Optional[str], include_completed: bool) -> str:
        """Generate JSON visualization"""
        if task_id:
            task = self.tracker.get_task(task_id)
            if task:
                data = self._task_to_dict(task, include_completed)
            else:
                data = {}
        else:
            tree = self.tracker.get_task_tree()
            if not include_completed:
                # Filter out completed tasks
                tree = self._filter_completed(tree)
            data = tree
        
        return json.dumps(data, indent=2, default=str)
    
    def _task_to_dict(self, task: TaskProgress, include_completed: bool) -> Dict[str, Any]:
        """Convert task to dictionary"""
        task_dict = {
            "id": task.task_id,
            "name": task.name,
            "status": task.status,
            "progress": task.progress,
            "start_time": task.start_time.isoformat(),
            "end_time": task.end_time.isoformat() if task.end_time else None,
            "duration": str(task.duration),
            "metadata": task.metadata,
            "events": [
                {
                    "timestamp": e.timestamp.isoformat(),
                    "status": e.status,
                    "progress": e.progress,
                    "message": e.message,
                    "error": e.error
                }
                for e in task.events[-5:]  # Last 5 events
            ]
        }
        
        # Add children
        if task.children:
            children = []
            for child_id in task.children:
                child = self.tracker.get_task(child_id)
                if child and (include_completed or not child.is_complete):
                    children.append(self._task_to_dict(child, include_completed))
            
            if children:
                task_dict["children"] = children
        
        return task_dict
    
    def _visualize_tree(self, task_id: Optional[str], include_completed: bool) -> str:
        """Generate tree visualization"""
        lines = []
        
        def add_tree_node(task: TaskProgress, prefix: str, is_last: bool):
            # Node connector
            connector = "└── " if is_last else "├── "
            
            # Status and progress
            status_mark = "✓" if task.status == ProgressStatus.COMPLETED else "●"
            color = {
                ProgressStatus.RUNNING: "\033[32m",  # Green
                ProgressStatus.COMPLETED: "\033[90m",  # Gray
                ProgressStatus.FAILED: "\033[31m",  # Red
                ProgressStatus.CANCELLED: "\033[33m",  # Yellow
            }.get(task.status, "")
            reset = "\033[0m" if color else ""
            
            lines.append(
                f"{prefix}{connector}{color}{status_mark} {task.name} "
                f"[{task.progress:.0f}%]{reset}"
            )
            
            # Add children
            child_prefix = prefix + ("    " if is_last else "│   ")
            children = [
                self.tracker.get_task(cid) for cid in task.children
                if self.tracker.get_task(cid)
            ]
            
            if not include_completed:
                children = [c for c in children if not c.is_complete]
            
            for i, child in enumerate(children):
                is_last_child = i == len(children) - 1
                add_tree_node(child, child_prefix, is_last_child)
        
        if task_id:
            task = self.tracker.get_task(task_id)
            if task:
                add_tree_node(task, "", True)
        else:
            # Get root tasks
            all_tasks = self.tracker.get_all_tasks()
            root_tasks = [t for t in all_tasks if t.parent_id is None]
            
            if not include_completed:
                root_tasks = [t for t in root_tasks if not t.is_complete]
            
            for i, task in enumerate(root_tasks):
                is_last = i == len(root_tasks) - 1
                add_tree_node(task, "", is_last)
        
        return "\n".join(lines)
    
    def _visualize_timeline(self, task_id: Optional[str], include_completed: bool) -> str:
        """Generate timeline visualization"""
        events = []
        
        if task_id:
            task = self.tracker.get_task(task_id)
            if task:
                self._collect_events(task, events, include_completed)
        else:
            for task in self.tracker.get_all_tasks():
                if task.parent_id is None:  # Root tasks
                    self._collect_events(task, events, include_completed)
        
        # Sort by timestamp
        events.sort(key=lambda e: e["timestamp"])
        
        # Format timeline
        lines = ["=== TIMELINE ==="]
        
        for event in events:
            timestamp = event["timestamp"].strftime("%H:%M:%S")
            task_name = event["task_name"]
            status = event["status"]
            message = event["message"]
            
            icon = {
                ProgressStatus.PENDING: "○",
                ProgressStatus.RUNNING: "►",
                ProgressStatus.COMPLETED: "✓",
                ProgressStatus.FAILED: "✗",
                ProgressStatus.CANCELLED: "⊗",
                ProgressStatus.PAUSED: "║"
            }.get(status, "?")
            
            lines.append(f"{timestamp} {icon} [{task_name}] {message}")
        
        return "\n".join(lines)
    
    def _visualize_gantt(self, task_id: Optional[str], include_completed: bool) -> str:
        """Generate Gantt chart visualization"""
        tasks = []
        
        if task_id:
            task = self.tracker.get_task(task_id)
            if task:
                self._collect_tasks_flat(task, tasks, include_completed)
        else:
            for task in self.tracker.get_all_tasks():
                if task.parent_id is None:
                    self._collect_tasks_flat(task, tasks, include_completed)
        
        if not tasks:
            return "No tasks to display"
        
        # Find time range
        min_time = min(t.start_time for t in tasks)
        max_time = max(
            t.end_time or datetime.utcnow()
            for t in tasks
        )
        
        total_duration = (max_time - min_time).total_seconds()
        if total_duration == 0:
            total_duration = 1
        
        # Chart settings
        chart_width = 50
        name_width = 30
        
        lines = ["=== GANTT CHART ==="]
        lines.append(f"{'Task':<{name_width}} |{'Progress':^{chart_width}}|")
        lines.append(f"{'-' * name_width}-+-{'-' * chart_width}-+")
        
        for task in tasks:
            # Calculate position
            start_offset = (task.start_time - min_time).total_seconds()
            start_pos = int((start_offset / total_duration) * chart_width)
            
            end_time = task.end_time or datetime.utcnow()
            end_offset = (end_time - min_time).total_seconds()
            end_pos = int((end_offset / total_duration) * chart_width)
            
            # Build bar
            bar = [" "] * chart_width
            
            for i in range(start_pos, min(end_pos + 1, chart_width)):
                if task.status == ProgressStatus.COMPLETED:
                    bar[i] = "█"
                elif task.status == ProgressStatus.RUNNING:
                    progress_point = start_pos + int((end_pos - start_pos) * task.progress / 100)
                    bar[i] = "█" if i <= progress_point else "░"
                elif task.status == ProgressStatus.FAILED:
                    bar[i] = "✗"
                elif task.status == ProgressStatus.CANCELLED:
                    bar[i] = "-"
                else:
                    bar[i] = "░"
            
            # Task line
            task_name = task.name[:name_width].ljust(name_width)
            lines.append(f"{task_name} |{''.join(bar)}|")
        
        return "\n".join(lines)
    
    def _visualize_progress_bar(self, task_id: Optional[str]) -> str:
        """Generate progress bar visualization"""
        if task_id:
            task = self.tracker.get_task(task_id)
            if not task:
                return "Task not found"
            
            tasks = [task]
        else:
            tasks = self.tracker.get_active_tasks()
        
        if not tasks:
            return "No active tasks"
        
        lines = []
        
        for task in tasks:
            # Create progress bar
            bar = self._create_progress_bar(task.progress, 30)
            
            # Status indicator
            if task.status == ProgressStatus.RUNNING:
                spinner = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
                spin_char = spinner[int(datetime.utcnow().timestamp() * 10) % len(spinner)]
            else:
                spin_char = " "
            
            lines.append(
                f"{spin_char} {task.name}\n"
                f"  {bar} {task.progress:.1f}% - {task.status}"
            )
            
            # Add ETA if running
            if task.status == ProgressStatus.RUNNING and task.progress > 0:
                elapsed = task.duration.total_seconds()
                if elapsed > 0:
                    estimated_total = elapsed / (task.progress / 100)
                    remaining = estimated_total - elapsed
                    
                    if remaining > 0:
                        minutes, seconds = divmod(int(remaining), 60)
                        hours, minutes = divmod(minutes, 60)
                        
                        if hours > 0:
                            eta = f"{hours}h {minutes}m"
                        elif minutes > 0:
                            eta = f"{minutes}m {seconds}s"
                        else:
                            eta = f"{seconds}s"
                        
                        lines[-1] += f" - ETA: {eta}"
        
        return "\n\n".join(lines)
    
    def _create_progress_bar(self, progress: float, width: int) -> str:
        """Create a progress bar string"""
        filled = int((progress / 100) * width)
        empty = width - filled
        
        return f"[{'█' * filled}{'░' * empty}]"
    
    def _collect_events(
        self,
        task: TaskProgress,
        events: List[Dict[str, Any]],
        include_completed: bool
    ):
        """Collect events from task and children"""
        if not include_completed and task.is_complete:
            return
        
        # Add task events
        for event in task.events:
            events.append({
                "timestamp": event.timestamp,
                "task_name": task.name,
                "status": event.status,
                "message": event.message
            })
        
        # Add children events
        for child_id in task.children:
            child = self.tracker.get_task(child_id)
            if child:
                self._collect_events(child, events, include_completed)
    
    def _collect_tasks_flat(
        self,
        task: TaskProgress,
        tasks: List[TaskProgress],
        include_completed: bool
    ):
        """Collect tasks in flat list"""
        if not include_completed and task.is_complete:
            return
        
        tasks.append(task)
        
        for child_id in task.children:
            child = self.tracker.get_task(child_id)
            if child:
                self._collect_tasks_flat(child, tasks, include_completed)
    
    def _filter_completed(self, tree: Dict[str, Any]) -> Dict[str, Any]:
        """Filter out completed tasks from tree"""
        if isinstance(tree, dict):
            if "status" in tree and tree["status"] == ProgressStatus.COMPLETED:
                return None
            
            filtered = {}
            for key, value in tree.items():
                if key == "children" and isinstance(value, list):
                    filtered_children = []
                    for child in value:
                        filtered_child = self._filter_completed(child)
                        if filtered_child:
                            filtered_children.append(filtered_child)
                    if filtered_children:
                        filtered[key] = filtered_children
                elif key == "roots" and isinstance(value, list):
                    filtered_roots = []
                    for root in value:
                        filtered_root = self._filter_completed(root)
                        if filtered_root:
                            filtered_roots.append(filtered_root)
                    filtered[key] = filtered_roots
                else:
                    filtered[key] = value
            
            return filtered
        
        return tree