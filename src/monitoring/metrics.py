"""Metrics collection for monitoring agent performance"""

from typing import Dict, Any, List, Optional
from datetime import datetime
import time
import json
from pathlib import Path
from collections import defaultdict
import statistics


class MetricsCollector:
    """Collects and aggregates metrics for agent performance monitoring"""
    
    def __init__(self, metrics_dir: str = "./metrics"):
        """Initialize metrics collector
        
        Args:
            metrics_dir: Directory to store metrics data
        """
        self.metrics_dir = Path(metrics_dir)
        self.metrics_dir.mkdir(exist_ok=True)
        
        self.current_metrics = defaultdict(list)
        self.timers = {}
    
    def start_timer(self, metric_name: str):
        """Start a timer for a metric
        
        Args:
            metric_name: Name of the metric
        """
        self.timers[metric_name] = time.time()
    
    def end_timer(self, metric_name: str) -> float:
        """End a timer and record the duration
        
        Args:
            metric_name: Name of the metric
            
        Returns:
            Duration in seconds
        """
        if metric_name not in self.timers:
            return 0.0
        
        duration = time.time() - self.timers[metric_name]
        del self.timers[metric_name]
        
        self.record_metric(f"{metric_name}_duration", duration)
        return duration
    
    def record_metric(self, metric_name: str, value: Any):
        """Record a metric value
        
        Args:
            metric_name: Name of the metric
            value: Value to record
        """
        self.current_metrics[metric_name].append({
            "value": value,
            "timestamp": datetime.utcnow().isoformat()
        })
    
    def increment_counter(self, counter_name: str, amount: int = 1):
        """Increment a counter metric
        
        Args:
            counter_name: Name of the counter
            amount: Amount to increment by
        """
        current = self.get_current_value(counter_name, 0)
        self.record_metric(counter_name, current + amount)
    
    def get_current_value(self, metric_name: str, default: Any = None) -> Any:
        """Get the current value of a metric
        
        Args:
            metric_name: Name of the metric
            default: Default value if metric doesn't exist
            
        Returns:
            Current metric value
        """
        if metric_name in self.current_metrics and self.current_metrics[metric_name]:
            return self.current_metrics[metric_name][-1]["value"]
        return default
    
    def get_statistics(self, metric_name: str) -> Dict[str, float]:
        """Get statistics for a metric
        
        Args:
            metric_name: Name of the metric
            
        Returns:
            Dictionary of statistics
        """
        if metric_name not in self.current_metrics:
            return {}
        
        values = [entry["value"] for entry in self.current_metrics[metric_name]]
        
        if not values:
            return {}
        
        # Filter to numeric values only
        numeric_values = [v for v in values if isinstance(v, (int, float))]
        
        if not numeric_values:
            return {"count": len(values)}
        
        return {
            "count": len(numeric_values),
            "min": min(numeric_values),
            "max": max(numeric_values),
            "mean": statistics.mean(numeric_values),
            "median": statistics.median(numeric_values),
            "stdev": statistics.stdev(numeric_values) if len(numeric_values) > 1 else 0
        }
    
    def save_metrics(self, session_id: str):
        """Save current metrics to file
        
        Args:
            session_id: Session identifier
        """
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        filename = self.metrics_dir / f"metrics_{session_id}_{timestamp}.json"
        
        metrics_data = {
            "session_id": session_id,
            "timestamp": datetime.utcnow().isoformat(),
            "metrics": dict(self.current_metrics),
            "statistics": {
                metric: self.get_statistics(metric)
                for metric in self.current_metrics
            }
        }
        
        with open(filename, 'w') as f:
            json.dump(metrics_data, f, indent=2)
    
    def load_metrics(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Load metrics for a session
        
        Args:
            session_id: Session identifier
            
        Returns:
            Metrics data if found
        """
        # Find the most recent metrics file for the session
        pattern = f"metrics_{session_id}_*.json"
        files = list(self.metrics_dir.glob(pattern))
        
        if not files:
            return None
        
        # Get the most recent file
        latest_file = max(files, key=lambda f: f.stat().st_mtime)
        
        with open(latest_file, 'r') as f:
            return json.load(f)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of all current metrics
        
        Returns:
            Summary dictionary
        """
        summary = {
            "metrics": {},
            "counters": {},
            "timers": {}
        }
        
        for metric_name, entries in self.current_metrics.items():
            if metric_name.endswith("_duration"):
                # Timer metric
                stats = self.get_statistics(metric_name)
                summary["timers"][metric_name] = stats
            elif metric_name.endswith("_count"):
                # Counter metric
                summary["counters"][metric_name] = self.get_current_value(metric_name, 0)
            else:
                # General metric
                summary["metrics"][metric_name] = {
                    "current": self.get_current_value(metric_name),
                    "stats": self.get_statistics(metric_name)
                }
        
        return summary
    
    def reset(self):
        """Reset all metrics"""
        self.current_metrics.clear()
        self.timers.clear()


class PerformanceMonitor:
    """Monitors agent performance and system resources"""
    
    def __init__(self):
        self.metrics_collector = MetricsCollector()
        self.performance_thresholds = {
            "response_time": 5.0,  # seconds
            "memory_usage": 1024,  # MB
            "token_usage": 4000,   # tokens
            "error_rate": 0.1      # 10%
        }
    
    def track_agent_execution(
        self,
        agent_name: str,
        execution_time: float,
        token_usage: int,
        success: bool,
        error: Optional[str] = None
    ):
        """Track agent execution metrics
        
        Args:
            agent_name: Name of the agent
            execution_time: Execution time in seconds
            token_usage: Number of tokens used
            success: Whether execution was successful
            error: Error message if failed
        """
        # Record metrics
        self.metrics_collector.record_metric(f"{agent_name}_execution_time", execution_time)
        self.metrics_collector.record_metric(f"{agent_name}_token_usage", token_usage)
        self.metrics_collector.increment_counter(f"{agent_name}_total_executions")
        
        if success:
            self.metrics_collector.increment_counter(f"{agent_name}_successful_executions")
        else:
            self.metrics_collector.increment_counter(f"{agent_name}_failed_executions")
            if error:
                self.metrics_collector.record_metric(f"{agent_name}_errors", error)
        
        # Check thresholds
        self._check_performance_thresholds(agent_name, execution_time, token_usage)
    
    def _check_performance_thresholds(
        self,
        agent_name: str,
        execution_time: float,
        token_usage: int
    ):
        """Check if performance metrics exceed thresholds
        
        Args:
            agent_name: Name of the agent
            execution_time: Execution time in seconds
            token_usage: Number of tokens used
        """
        alerts = []
        
        if execution_time > self.performance_thresholds["response_time"]:
            alerts.append(f"High response time: {execution_time:.2f}s")
        
        if token_usage > self.performance_thresholds["token_usage"]:
            alerts.append(f"High token usage: {token_usage}")
        
        # Calculate error rate
        total = self.metrics_collector.get_current_value(f"{agent_name}_total_executions", 1)
        failed = self.metrics_collector.get_current_value(f"{agent_name}_failed_executions", 0)
        error_rate = failed / total if total > 0 else 0
        
        if error_rate > self.performance_thresholds["error_rate"]:
            alerts.append(f"High error rate: {error_rate:.2%}")
        
        # Record alerts
        if alerts:
            self.metrics_collector.record_metric(f"{agent_name}_alerts", alerts)
    
    def get_agent_performance_summary(self, agent_name: str) -> Dict[str, Any]:
        """Get performance summary for an agent
        
        Args:
            agent_name: Name of the agent
            
        Returns:
            Performance summary
        """
        total = self.metrics_collector.get_current_value(f"{agent_name}_total_executions", 0)
        successful = self.metrics_collector.get_current_value(f"{agent_name}_successful_executions", 0)
        failed = self.metrics_collector.get_current_value(f"{agent_name}_failed_executions", 0)
        
        return {
            "total_executions": total,
            "successful_executions": successful,
            "failed_executions": failed,
            "success_rate": successful / total if total > 0 else 0,
            "execution_time_stats": self.metrics_collector.get_statistics(f"{agent_name}_execution_time"),
            "token_usage_stats": self.metrics_collector.get_statistics(f"{agent_name}_token_usage"),
            "recent_alerts": self.metrics_collector.get_current_value(f"{agent_name}_alerts", [])
        }
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get overall system health metrics
        
        Returns:
            System health summary
        """
        all_metrics = self.metrics_collector.get_summary()
        
        # Calculate overall health score
        health_score = 1.0
        issues = []
        
        # Check error rates
        for counter_name, value in all_metrics["counters"].items():
            if counter_name.endswith("_failed_executions"):
                agent_name = counter_name.replace("_failed_executions", "")
                total = all_metrics["counters"].get(f"{agent_name}_total_executions", 0)
                if total > 0:
                    error_rate = value / total
                    if error_rate > self.performance_thresholds["error_rate"]:
                        health_score -= 0.2
                        issues.append(f"{agent_name} has high error rate: {error_rate:.2%}")
        
        # Check response times
        for timer_name, stats in all_metrics["timers"].items():
            if stats.get("mean", 0) > self.performance_thresholds["response_time"]:
                health_score -= 0.1
                issues.append(f"{timer_name} has high average time: {stats['mean']:.2f}s")
        
        health_score = max(0, health_score)
        
        return {
            "health_score": health_score,
            "status": "healthy" if health_score > 0.7 else "degraded" if health_score > 0.4 else "unhealthy",
            "issues": issues,
            "metrics_summary": all_metrics
        }