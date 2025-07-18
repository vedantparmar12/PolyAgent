"""PRP executor that implements features from PRPs with validation loops."""

from typing import Dict, List, Optional, Any
from pathlib import Path
import re
from agent import OpenRouterAgent
from .models import ValidationResult

class PRPExecutor:
    """Executes PRPs to implement features with validation."""
    
    def __init__(self, agent: Optional[OpenRouterAgent] = None):
        """Initialize PRP executor.
        
        Args:
            agent: OpenRouterAgent instance for implementation
        """
        self.agent = agent or OpenRouterAgent(silent=False)
        self.validation_results: List[ValidationResult] = []
        
    def execute_prp(self, prp_path: str) -> Dict[str, Any]:
        """Execute a PRP to implement the feature.
        
        Args:
            prp_path: Path to the PRP file
            
        Returns:
            Dictionary with execution results
        """
        # Load PRP content
        prp_content = self._load_prp(prp_path)
        
        # Extract tasks from PRP
        tasks = self._extract_tasks(prp_content)
        
        # Create implementation plan
        plan_prompt = f"""
Based on this PRP, create a detailed implementation plan:

{prp_content}

Use the mark_task_complete tool to track progress.
For each task:
1. Understand the requirements
2. Implement the code
3. Run validation
4. Fix any issues
5. Move to next task

Start implementing now.
"""
        
        # Execute implementation
        implementation_result = self.agent.run(plan_prompt)
        
        # Run final validation
        validation_result = self._run_final_validation(prp_content)
        
        return {
            "status": "success" if validation_result.success else "failed",
            "implementation_result": implementation_result,
            "validation_result": validation_result,
            "tasks_completed": len(tasks)
        }
    
    def _load_prp(self, prp_path: str) -> str:
        """Load PRP content from file."""
        path = Path(prp_path)
        if not path.exists():
            raise FileNotFoundError(f"PRP not found: {prp_path}")
        
        return path.read_text(encoding='utf-8')
    
    def _extract_tasks(self, prp_content: str) -> List[Dict[str, str]]:
        """Extract task list from PRP content."""
        tasks = []
        
        # Find the tasks section
        tasks_section = re.search(
            r'### List of Tasks\s*```yaml\s*(.*?)\s*```',
            prp_content,
            re.DOTALL | re.IGNORECASE
        )
        
        if tasks_section:
            tasks_yaml = tasks_section.group(1)
            # Parse tasks (simple extraction)
            task_pattern = r'Task \d+:(.*?)(?=Task \d+:|$)'
            matches = re.findall(task_pattern, tasks_yaml, re.DOTALL)
            
            for i, match in enumerate(matches):
                tasks.append({
                    "id": f"task_{i+1}",
                    "description": match.strip()
                })
        
        return tasks
    
    def _run_final_validation(self, prp_content: str) -> ValidationResult:
        """Run final validation based on PRP checklist."""
        # Extract validation checklist
        checklist = self._extract_validation_checklist(prp_content)
        
        # Run validation using validation tool
        validation_prompt = f"""
Run the final validation checklist from the PRP:

{checklist}

Use the run_validation tool to execute each validation step.
Report results for each item.
"""
        
        validation_output = self.agent.run(validation_prompt)
        
        # Parse results
        success = "failed" not in validation_output.lower()
        errors = []
        warnings: List[str] = []
        
        if not success:
            # Extract errors from output
            error_lines = [
                line for line in validation_output.split('\n')
                if 'error' in line.lower() or 'failed' in line.lower()
            ]
            errors = error_lines[:5]  # Limit to 5 errors
        
        return ValidationResult(
            success=success,
            errors=errors,
            warnings=warnings
        )
    
    def _extract_validation_checklist(self, prp_content: str) -> str:
        """Extract validation checklist from PRP."""
        # Find the Final Validation Checklist section
        checklist_match = re.search(
            r'## Final Validation Checklist\s*(.*?)(?=##|\Z)',
            prp_content,
            re.DOTALL
        )
        
        if checklist_match:
            return checklist_match.group(1).strip()
        
        # Default checklist if not found
        return """
- [ ] All tests pass: `pytest tests/ -v`
- [ ] No linting errors: `ruff check .`
- [ ] Feature works as expected
"""
    
    def execute_task(self, task: Dict[str, str]) -> bool:
        """Execute a single task with validation.
        
        Args:
            task: Task dictionary with description
            
        Returns:
            True if task completed successfully
        """
        task_prompt = f"""
Execute this task:
{task['description']}

After implementation, run validation to ensure correctness.
Fix any issues found during validation.
"""
        
        result = self.agent.run(task_prompt)
        
        # Check if task was marked complete
        return "mark_task_complete" in result or "completed" in result.lower()