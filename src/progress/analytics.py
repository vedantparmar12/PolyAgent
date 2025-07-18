"""Analytics system for progress tracking and performance monitoring"""

from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from collections import defaultdict
from pydantic import BaseModel, Field
import statistics
from .progress_tracker import ProgressTracker, TaskProgress, ProgressStatus
import logfire


class PerformanceMetrics(BaseModel):
    """Performance metrics for a task or operation"""
    task_name: str
    total_executions: int = 0
    successful_executions: int = 0
    failed_executions: int = 0
    avg_duration_seconds: float = 0.0
    min_duration_seconds: float = float('inf')
    max_duration_seconds: float = 0.0
    median_duration_seconds: float = 0.0
    p95_duration_seconds: float = 0.0
    p99_duration_seconds: float = 0.0
    success_rate: float = 0.0
    error_rate: float = 0.0
    throughput_per_minute: float = 0.0
    last_execution: Optional[datetime] = None


class TaskAnalytics(BaseModel):
    """Analytics for a specific task type"""
    task_type: str
    metrics: PerformanceMetrics
    hourly_stats: Dict[int, Dict[str, Any]] = Field(default_factory=dict)
    daily_stats: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    error_categories: Dict[str, int] = Field(default_factory=dict)
    common_patterns: List[str] = Field(default_factory=list)


class ProgressAnalytics:
    """Analytics system for progress tracking"""
    
    def __init__(self, tracker: ProgressTracker):
        """Initialize analytics
        
        Args:
            tracker: Progress tracker instance
        """
        self.tracker = tracker
        self._logger = logfire.span("progress_analytics")
        
        # Analytics data
        self._task_analytics: Dict[str, TaskAnalytics] = {}
        self._duration_history: Dict[str, List[float]] = defaultdict(list)
        self._error_history: Dict[str, List[Tuple[datetime, str]]] = defaultdict(list)
        
        # Real-time metrics
        self._active_task_count = 0
        self._tasks_per_minute = defaultdict(int)
        self._last_minute_update = datetime.utcnow().replace(second=0, microsecond=0)
    
    def analyze_task(self, task: TaskProgress) -> TaskAnalytics:
        """Analyze a completed task
        
        Args:
            task: Task to analyze
            
        Returns:
            Task analytics
        """
        task_type = task.name
        
        # Get or create analytics
        if task_type not in self._task_analytics:
            self._task_analytics[task_type] = TaskAnalytics(
                task_type=task_type,
                metrics=PerformanceMetrics(task_name=task_type)
            )
        
        analytics = self._task_analytics[task_type]
        metrics = analytics.metrics
        
        # Update execution counts
        metrics.total_executions += 1
        
        if task.status == ProgressStatus.COMPLETED:
            metrics.successful_executions += 1
        elif task.status == ProgressStatus.FAILED:
            metrics.failed_executions += 1
            
            # Track error categories
            if task.events:
                last_event = task.events[-1]
                if last_event.error:
                    error_type = self._categorize_error(last_event.error)
                    analytics.error_categories[error_type] = \
                        analytics.error_categories.get(error_type, 0) + 1
                    
                    # Store error history
                    self._error_history[task_type].append(
                        (datetime.utcnow(), last_event.error)
                    )
        
        # Update duration metrics
        if task.duration:
            duration_seconds = task.duration.total_seconds()
            self._duration_history[task_type].append(duration_seconds)
            
            # Keep only recent history (last 1000 entries)
            if len(self._duration_history[task_type]) > 1000:
                self._duration_history[task_type] = \
                    self._duration_history[task_type][-1000:]
            
            # Calculate statistics
            durations = self._duration_history[task_type]
            metrics.avg_duration_seconds = statistics.mean(durations)
            metrics.min_duration_seconds = min(durations)
            metrics.max_duration_seconds = max(durations)
            metrics.median_duration_seconds = statistics.median(durations)
            
            if len(durations) >= 20:
                sorted_durations = sorted(durations)
                metrics.p95_duration_seconds = sorted_durations[int(len(durations) * 0.95)]
                metrics.p99_duration_seconds = sorted_durations[int(len(durations) * 0.99)]
        
        # Update rates
        if metrics.total_executions > 0:
            metrics.success_rate = metrics.successful_executions / metrics.total_executions
            metrics.error_rate = metrics.failed_executions / metrics.total_executions
        
        # Update last execution time
        metrics.last_execution = task.end_time or task.start_time
        
        # Update hourly/daily stats
        self._update_time_based_stats(analytics, task)
        
        # Detect patterns
        self._detect_patterns(analytics)
        
        return analytics
    
    def get_real_time_metrics(self) -> Dict[str, Any]:
        """Get real-time system metrics
        
        Returns:
            Real-time metrics
        """
        all_tasks = self.tracker.get_all_tasks()
        active_tasks = self.tracker.get_active_tasks()
        
        # Update tasks per minute
        current_minute = datetime.utcnow().replace(second=0, microsecond=0)
        if current_minute != self._last_minute_update:
            self._last_minute_update = current_minute
            self._tasks_per_minute[current_minute] = 0
        
        # Count recent task starts
        recent_starts = 0
        for task in all_tasks:
            if task.start_time >= current_minute:
                recent_starts += 1
        
        self._tasks_per_minute[current_minute] = recent_starts
        
        # Calculate throughput (last 5 minutes)
        five_minutes_ago = current_minute - timedelta(minutes=5)
        recent_throughput = sum(
            count for time, count in self._tasks_per_minute.items()
            if time >= five_minutes_ago
        ) / 5.0
        
        # Task distribution by status
        status_distribution = defaultdict(int)
        for task in all_tasks:
            status_distribution[task.status] += 1
        
        # Average progress of active tasks
        avg_progress = 0.0
        if active_tasks:
            avg_progress = sum(t.progress for t in active_tasks) / len(active_tasks)
        
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "active_tasks": len(active_tasks),
            "total_tasks": len(all_tasks),
            "tasks_per_minute": recent_throughput,
            "average_active_progress": avg_progress,
            "status_distribution": dict(status_distribution),
            "memory_tasks": len(self.tracker._tasks),
            "analytics_cache_size": len(self._task_analytics)
        }
    
    def get_performance_summary(
        self,
        task_type: Optional[str] = None,
        time_window: Optional[timedelta] = None
    ) -> Dict[str, Any]:
        """Get performance summary
        
        Args:
            task_type: Specific task type (None for all)
            time_window: Time window to consider
            
        Returns:
            Performance summary
        """
        if task_type:
            if task_type not in self._task_analytics:
                return {"error": f"No analytics for task type: {task_type}"}
            
            analytics = self._task_analytics[task_type]
            return {
                "task_type": task_type,
                "metrics": analytics.metrics.dict(),
                "error_categories": analytics.error_categories,
                "patterns": analytics.common_patterns
            }
        
        # Overall summary
        summary = {
            "total_task_types": len(self._task_analytics),
            "overall_metrics": self._calculate_overall_metrics(),
            "top_performers": self._get_top_performers(5),
            "problem_areas": self._get_problem_areas(5),
            "recent_errors": self._get_recent_errors(10)
        }
        
        if time_window:
            summary["time_window_stats"] = self._get_time_window_stats(time_window)
        
        return summary
    
    def get_trend_analysis(
        self,
        task_type: str,
        metric: str = "duration",
        period: str = "hourly"
    ) -> Dict[str, Any]:
        """Get trend analysis for a task type
        
        Args:
            task_type: Task type to analyze
            metric: Metric to analyze (duration, success_rate, throughput)
            period: Analysis period (hourly, daily)
            
        Returns:
            Trend analysis
        """
        if task_type not in self._task_analytics:
            return {"error": f"No analytics for task type: {task_type}"}
        
        analytics = self._task_analytics[task_type]
        
        if period == "hourly":
            time_stats = analytics.hourly_stats
        else:
            time_stats = analytics.daily_stats
        
        # Extract metric values
        metric_values = []
        for time_key in sorted(time_stats.keys()):
            stats = time_stats[time_key]
            if metric in stats:
                metric_values.append({
                    "time": time_key,
                    "value": stats[metric]
                })
        
        # Calculate trend
        trend = "stable"
        if len(metric_values) >= 3:
            recent_avg = statistics.mean([v["value"] for v in metric_values[-3:]])
            older_avg = statistics.mean([v["value"] for v in metric_values[:-3]])
            
            if recent_avg > older_avg * 1.1:
                trend = "increasing"
            elif recent_avg < older_avg * 0.9:
                trend = "decreasing"
        
        return {
            "task_type": task_type,
            "metric": metric,
            "period": period,
            "trend": trend,
            "data_points": metric_values,
            "latest_value": metric_values[-1]["value"] if metric_values else None
        }
    
    def get_bottlenecks(self, threshold_percentile: float = 0.9) -> List[Dict[str, Any]]:
        """Identify performance bottlenecks
        
        Args:
            threshold_percentile: Percentile threshold for bottleneck detection
            
        Returns:
            List of bottlenecks
        """
        bottlenecks = []
        
        for task_type, analytics in self._task_analytics.items():
            metrics = analytics.metrics
            
            # Check for slow tasks
            if metrics.p95_duration_seconds > metrics.avg_duration_seconds * 2:
                bottlenecks.append({
                    "type": "slow_tail_latency",
                    "task_type": task_type,
                    "severity": "high",
                    "details": {
                        "p95_duration": metrics.p95_duration_seconds,
                        "avg_duration": metrics.avg_duration_seconds,
                        "ratio": metrics.p95_duration_seconds / max(metrics.avg_duration_seconds, 0.001)
                    }
                })
            
            # Check for high error rates
            if metrics.error_rate > 0.1:
                bottlenecks.append({
                    "type": "high_error_rate",
                    "task_type": task_type,
                    "severity": "critical" if metrics.error_rate > 0.3 else "high",
                    "details": {
                        "error_rate": metrics.error_rate,
                        "failed_count": metrics.failed_executions,
                        "total_count": metrics.total_executions
                    }
                })
            
            # Check for low throughput
            if metrics.throughput_per_minute < 0.1 and metrics.total_executions > 10:
                bottlenecks.append({
                    "type": "low_throughput",
                    "task_type": task_type,
                    "severity": "medium",
                    "details": {
                        "throughput": metrics.throughput_per_minute,
                        "avg_duration": metrics.avg_duration_seconds
                    }
                })
        
        # Sort by severity
        severity_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        bottlenecks.sort(key=lambda x: severity_order.get(x["severity"], 999))
        
        return bottlenecks
    
    def generate_report(self, format: str = "text") -> str:
        """Generate analytics report
        
        Args:
            format: Report format (text, json, markdown)
            
        Returns:
            Formatted report
        """
        data = {
            "generated_at": datetime.utcnow().isoformat(),
            "summary": self.get_performance_summary(),
            "real_time": self.get_real_time_metrics(),
            "bottlenecks": self.get_bottlenecks()
        }
        
        if format == "json":
            import json
            return json.dumps(data, indent=2, default=str)
        
        elif format == "markdown":
            return self._format_markdown_report(data)
        
        else:  # text
            return self._format_text_report(data)
    
    def _categorize_error(self, error_message: str) -> str:
        """Categorize error message"""
        error_lower = error_message.lower()
        
        if "timeout" in error_lower:
            return "timeout"
        elif "connection" in error_lower or "network" in error_lower:
            return "network"
        elif "permission" in error_lower or "access" in error_lower:
            return "permission"
        elif "memory" in error_lower or "resource" in error_lower:
            return "resource"
        elif "validation" in error_lower or "invalid" in error_lower:
            return "validation"
        else:
            return "other"
    
    def _update_time_based_stats(self, analytics: TaskAnalytics, task: TaskProgress):
        """Update hourly and daily statistics"""
        if not task.start_time:
            return
        
        # Hourly stats
        hour_key = task.start_time.hour
        if hour_key not in analytics.hourly_stats:
            analytics.hourly_stats[hour_key] = defaultdict(list)
        
        hour_stats = analytics.hourly_stats[hour_key]
        hour_stats["executions"].append(1)
        
        if task.duration:
            hour_stats["durations"].append(task.duration.total_seconds())
        
        if task.status == ProgressStatus.COMPLETED:
            hour_stats["successes"].append(1)
        elif task.status == ProgressStatus.FAILED:
            hour_stats["failures"].append(1)
        
        # Calculate aggregates
        hour_stats["total"] = len(hour_stats["executions"])
        hour_stats["success_rate"] = len(hour_stats.get("successes", [])) / hour_stats["total"]
        
        if hour_stats.get("durations"):
            hour_stats["avg_duration"] = statistics.mean(hour_stats["durations"])
        
        # Daily stats
        day_key = task.start_time.date().isoformat()
        if day_key not in analytics.daily_stats:
            analytics.daily_stats[day_key] = defaultdict(list)
        
        daily_stats = analytics.daily_stats[day_key]
        daily_stats["executions"].append(1)
        
        if task.duration:
            daily_stats["durations"].append(task.duration.total_seconds())
    
    def _detect_patterns(self, analytics: TaskAnalytics):
        """Detect common patterns in task execution"""
        patterns = []
        
        # Pattern: Consistent failures
        if analytics.metrics.error_rate > 0.5:
            patterns.append("High failure rate - needs investigation")
        
        # Pattern: Performance degradation
        if len(self._duration_history[analytics.task_type]) >= 10:
            recent = self._duration_history[analytics.task_type][-5:]
            older = self._duration_history[analytics.task_type][-10:-5]
            
            if statistics.mean(recent) > statistics.mean(older) * 1.5:
                patterns.append("Performance degradation detected")
        
        # Pattern: Time-based issues
        if analytics.hourly_stats:
            problem_hours = [
                hour for hour, stats in analytics.hourly_stats.items()
                if stats.get("success_rate", 1.0) < 0.5
            ]
            
            if len(problem_hours) >= 2:
                patterns.append(f"Issues during hours: {sorted(problem_hours)}")
        
        analytics.common_patterns = patterns
    
    def _calculate_overall_metrics(self) -> Dict[str, Any]:
        """Calculate overall system metrics"""
        all_tasks = self.tracker.get_all_tasks()
        
        total_executions = len(all_tasks)
        successful = len([t for t in all_tasks if t.status == ProgressStatus.COMPLETED])
        failed = len([t for t in all_tasks if t.status == ProgressStatus.FAILED])
        
        durations = [
            t.duration.total_seconds()
            for t in all_tasks
            if t.duration and t.is_complete
        ]
        
        return {
            "total_executions": total_executions,
            "successful_executions": successful,
            "failed_executions": failed,
            "overall_success_rate": successful / total_executions if total_executions > 0 else 0,
            "avg_duration_seconds": statistics.mean(durations) if durations else 0,
            "total_duration_seconds": sum(durations) if durations else 0
        }
    
    def _get_top_performers(self, limit: int) -> List[Dict[str, Any]]:
        """Get top performing task types"""
        performers = []
        
        for task_type, analytics in self._task_analytics.items():
            metrics = analytics.metrics
            
            # Calculate performance score
            score = (
                metrics.success_rate * 100 +
                (1 / max(metrics.avg_duration_seconds, 0.001)) * 10 +
                metrics.throughput_per_minute * 5
            )
            
            performers.append({
                "task_type": task_type,
                "performance_score": score,
                "success_rate": metrics.success_rate,
                "avg_duration": metrics.avg_duration_seconds,
                "executions": metrics.total_executions
            })
        
        performers.sort(key=lambda x: x["performance_score"], reverse=True)
        return performers[:limit]
    
    def _get_problem_areas(self, limit: int) -> List[Dict[str, Any]]:
        """Get problematic task types"""
        problems = []
        
        for task_type, analytics in self._task_analytics.items():
            metrics = analytics.metrics
            
            # Calculate problem score
            problem_score = (
                metrics.error_rate * 100 +
                metrics.avg_duration_seconds +
                len(analytics.error_categories) * 10
            )
            
            if problem_score > 0:
                problems.append({
                    "task_type": task_type,
                    "problem_score": problem_score,
                    "error_rate": metrics.error_rate,
                    "avg_duration": metrics.avg_duration_seconds,
                    "error_types": list(analytics.error_categories.keys())
                })
        
        problems.sort(key=lambda x: x["problem_score"], reverse=True)
        return problems[:limit]
    
    def _get_recent_errors(self, limit: int) -> List[Dict[str, Any]]:
        """Get recent errors across all task types"""
        all_errors = []
        
        for task_type, errors in self._error_history.items():
            for timestamp, error in errors:
                all_errors.append({
                    "task_type": task_type,
                    "timestamp": timestamp.isoformat(),
                    "error": error[:200],  # Truncate long errors
                    "category": self._categorize_error(error)
                })
        
        all_errors.sort(key=lambda x: x["timestamp"], reverse=True)
        return all_errors[:limit]
    
    def _get_time_window_stats(self, window: timedelta) -> Dict[str, Any]:
        """Get statistics for a specific time window"""
        cutoff_time = datetime.utcnow() - window
        
        all_tasks = self.tracker.get_all_tasks()
        window_tasks = [
            t for t in all_tasks
            if t.start_time and t.start_time >= cutoff_time
        ]
        
        if not window_tasks:
            return {"message": "No tasks in time window"}
        
        successful = len([t for t in window_tasks if t.status == ProgressStatus.COMPLETED])
        failed = len([t for t in window_tasks if t.status == ProgressStatus.FAILED])
        
        return {
            "window": str(window),
            "total_tasks": len(window_tasks),
            "successful": successful,
            "failed": failed,
            "success_rate": successful / len(window_tasks),
            "tasks_per_hour": len(window_tasks) / (window.total_seconds() / 3600)
        }
    
    def _format_text_report(self, data: Dict[str, Any]) -> str:
        """Format report as text"""
        lines = [
            "=== Progress Analytics Report ===",
            f"Generated: {data['generated_at']}",
            "",
            "=== Real-Time Metrics ===",
            f"Active Tasks: {data['real_time']['active_tasks']}",
            f"Total Tasks: {data['real_time']['total_tasks']}",
            f"Tasks/Minute: {data['real_time']['tasks_per_minute']:.2f}",
            f"Avg Progress: {data['real_time']['average_active_progress']:.1f}%",
            "",
            "=== Overall Performance ===",
            f"Success Rate: {data['summary']['overall_metrics']['overall_success_rate']:.1%}",
            f"Avg Duration: {data['summary']['overall_metrics']['avg_duration_seconds']:.2f}s",
            "",
            "=== Top Performers ==="
        ]
        
        for performer in data['summary']['top_performers']:
            lines.append(
                f"- {performer['task_type']}: "
                f"{performer['success_rate']:.1%} success, "
                f"{performer['avg_duration']:.2f}s avg"
            )
        
        if data['bottlenecks']:
            lines.extend([
                "",
                "=== Bottlenecks Detected ==="
            ])
            
            for bottleneck in data['bottlenecks']:
                lines.append(
                    f"- [{bottleneck['severity'].upper()}] "
                    f"{bottleneck['task_type']}: {bottleneck['type']}"
                )
        
        return "\n".join(lines)
    
    def _format_markdown_report(self, data: Dict[str, Any]) -> str:
        """Format report as markdown"""
        lines = [
            "# Progress Analytics Report",
            f"*Generated: {data['generated_at']}*",
            "",
            "## Real-Time Metrics",
            f"- **Active Tasks:** {data['real_time']['active_tasks']}",
            f"- **Total Tasks:** {data['real_time']['total_tasks']}",
            f"- **Tasks/Minute:** {data['real_time']['tasks_per_minute']:.2f}",
            f"- **Average Progress:** {data['real_time']['average_active_progress']:.1f}%",
            "",
            "## Overall Performance",
            f"- **Success Rate:** {data['summary']['overall_metrics']['overall_success_rate']:.1%}",
            f"- **Average Duration:** {data['summary']['overall_metrics']['avg_duration_seconds']:.2f}s",
            "",
            "## Top Performers"
        ]
        
        for performer in data['summary']['top_performers']:
            lines.append(
                f"1. **{performer['task_type']}**"
                f" - {performer['success_rate']:.1%} success rate"
                f" - {performer['avg_duration']:.2f}s average duration"
            )
        
        if data['bottlenecks']:
            lines.extend([
                "",
                "## ⚠️ Bottlenecks Detected"
            ])
            
            for bottleneck in data['bottlenecks']:
                severity_emoji = {
                    "critical": "🔴",
                    "high": "🟠",
                    "medium": "🟡",
                    "low": "🟢"
                }.get(bottleneck['severity'], "⚪")
                
                lines.append(
                    f"- {severity_emoji} **{bottleneck['task_type']}**: "
                    f"{bottleneck['type'].replace('_', ' ').title()}"
                )
        
        return "\n".join(lines)