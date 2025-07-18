"""Tool for running validation and tests on code."""

import subprocess
from typing import Dict, Any
from pathlib import Path
from .base_tool import BaseTool

class ValidationTool(BaseTool):
    """Tool that runs linting, type checking, and tests."""
    
    def __init__(self, config: dict):
        self.config = config
        self.validation_config = config.get('validation', {})
    
    @property
    def name(self) -> str:
        return "run_validation"
    
    @property
    def description(self) -> str:
        return "Run code validation including linting (ruff), type checking (mypy), and tests (pytest)"
    
    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "validation_type": {
                    "type": "string",
                    "description": "Type of validation to run",
                    "enum": ["lint", "typecheck", "test", "all"],
                    "default": "all"
                },
                "target": {
                    "type": "string",
                    "description": "File or directory to validate",
                    "default": "."
                },
                "fix": {
                    "type": "boolean",
                    "description": "Attempt to auto-fix issues (only for linting)",
                    "default": True
                }
            },
            "required": ["validation_type"]
        }
    
    def execute(self, validation_type: str = "all", target: str = ".", fix: bool = True) -> Dict[str, Any]:
        """Run validation on code.
        
        Args:
            validation_type: Type of validation to run
            target: File or directory to validate
            fix: Whether to auto-fix linting issues
            
        Returns:
            Dictionary containing validation results
        """
        results = {
            "status": "success",
            "validations": {}
        }
        
        try:
            if validation_type in ["lint", "all"]:
                results["validations"]["lint"] = self._run_linting(target, fix)
                
            if validation_type in ["typecheck", "all"]:
                results["validations"]["typecheck"] = self._run_typecheck(target)
                
            if validation_type in ["test", "all"]:
                results["validations"]["test"] = self._run_tests(target)
            
            # Determine overall status
            all_passed = all(
                v.get("success", False) 
                for v in results["validations"].values()
            )
            results["status"] = "success" if all_passed else "failed"
            
            return results
            
        except Exception as e:
            return {
                "status": "error",
                "error": f"Validation failed: {str(e)}"
            }
    
    def _run_linting(self, target: str, fix: bool) -> Dict[str, Any]:
        """Run ruff linting."""
        try:
            cmd = ["ruff", "check", target]
            if fix:
                cmd.append("--fix")
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True
            )
            
            return {
                "success": result.returncode == 0,
                "output": result.stdout,
                "errors": result.stderr,
                "fixed": fix and "fixed" in result.stdout.lower()
            }
        except FileNotFoundError:
            return {
                "success": False,
                "error": "ruff not found. Install with: pip install ruff"
            }
    
    def _run_typecheck(self, target: str) -> Dict[str, Any]:
        """Run mypy type checking."""
        try:
            result = subprocess.run(
                ["mypy", target, "--ignore-missing-imports"],
                capture_output=True,
                text=True
            )
            
            return {
                "success": result.returncode == 0,
                "output": result.stdout,
                "errors": result.stderr
            }
        except FileNotFoundError:
            return {
                "success": False,
                "error": "mypy not found. Install with: pip install mypy"
            }
    
    def _run_tests(self, target: str) -> Dict[str, Any]:
        """Run pytest tests."""
        try:
            # Determine test path
            if target == ".":
                test_path = "tests/"
            elif target.startswith("tests/"):
                test_path = target
            else:
                # Find corresponding test file
                path = Path(target)
                test_file = f"test_{path.name}"
                test_path = f"tests/{test_file}"
            
            result = subprocess.run(
                ["pytest", test_path, "-v", "--tb=short"],
                capture_output=True,
                text=True
            )
            
            # Parse test results
            output_lines = result.stdout.split('\n')
            passed = failed = 0
            
            for line in output_lines:
                if " passed" in line and " failed" in line:
                    parts = line.split()
                    for i, part in enumerate(parts):
                        if part == "passed":
                            passed = int(parts[i-1])
                        elif part == "failed":
                            failed = int(parts[i-1])
            
            return {
                "success": result.returncode == 0,
                "output": result.stdout,
                "errors": result.stderr,
                "passed": passed,
                "failed": failed
            }
        except FileNotFoundError:
            return {
                "success": False,
                "error": "pytest not found. Install with: pip install pytest"
            }