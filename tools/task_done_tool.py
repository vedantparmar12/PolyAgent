from .base_tool import BaseTool

class TaskDoneTool(BaseTool):
    def __init__(self, config: dict):
        self.config = config
    
    @property
    def name(self) -> str:
        return "mark_task_complete"
    
    @property
    def description(self) -> str:
        return "REQUIRED: Call this tool when the user's original request has been fully satisfied and you have provided a complete answer. This signals task completion and exits the agent loop."
    
    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "task_summary": {
                    "type": "string",
                    "description": "Brief summary of what was accomplished"
                },
                "completion_message": {
                    "type": "string",
                    "description": "Message to show the user indicating the task is complete"
                }
            },
            "required": ["task_summary", "completion_message"]
        }
    
    def execute(self, task_summary: str, completion_message: str) -> dict:
        """Mark a task as complete"""
        return {
            "status": "completed",
            "task_summary": task_summary,
            "completion_message": completion_message,
            "timestamp": self._get_timestamp()
        }
    
    def _get_timestamp(self) -> str:
        """Get current timestamp"""
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")