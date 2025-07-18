"""Tool for generating and executing PRPs within agent conversations."""

from typing import Dict, Any, Optional
from pathlib import Path
from .base_tool import BaseTool
from context import PRPGenerator, PRPExecutor, PRPRequest

class PRPTool(BaseTool):
    """Tool that allows agents to generate and execute PRPs."""
    
    def __init__(self, config: dict):
        self.config = config
        self.context_config = config.get('context_engineering', {})
    
    @property
    def name(self) -> str:
        return "manage_prp"
    
    @property
    def description(self) -> str:
        return "Generate or execute a PRP (Product Requirements Prompt) for implementing complex features"
    
    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "description": "Action to perform: 'generate' or 'execute'",
                    "enum": ["generate", "execute"]
                },
                "feature_description": {
                    "type": "string",
                    "description": "Description of the feature to implement (for generate)"
                },
                "prp_name": {
                    "type": "string",
                    "description": "Name for the PRP file (for generate) or path to PRP (for execute)"
                },
                "examples": {
                    "type": "array",
                    "description": "List of example files to reference (for generate)",
                    "items": {"type": "string"},
                    "default": []
                },
                "documentation_urls": {
                    "type": "array",
                    "description": "Documentation URLs to include (for generate)",
                    "items": {"type": "string"},
                    "default": []
                },
                "considerations": {
                    "type": "string",
                    "description": "Special considerations or requirements (for generate)"
                }
            },
            "required": ["action"]
        }
    
    def execute(
        self,
        action: str,
        feature_description: Optional[str] = None,
        prp_name: Optional[str] = None,
        examples: list = None,
        documentation_urls: list = None,
        considerations: Optional[str] = None
    ) -> Dict[str, Any]:
        """Generate or execute a PRP."""
        try:
            if action == "generate":
                return self._generate_prp(
                    feature_description,
                    prp_name,
                    examples or [],
                    documentation_urls or [],
                    considerations
                )
            elif action == "execute":
                return self._execute_prp(prp_name)
            else:
                return {
                    "status": "error",
                    "error": f"Unknown action: {action}"
                }
                
        except Exception as e:
            return {
                "status": "error",
                "error": f"PRP operation failed: {str(e)}"
            }
    
    def _generate_prp(
        self,
        feature_description: str,
        prp_name: Optional[str],
        examples: list,
        documentation_urls: list,
        considerations: Optional[str]
    ) -> Dict[str, Any]:
        """Generate a new PRP."""
        if not feature_description:
            return {
                "status": "error",
                "error": "Feature description is required for generation"
            }
        
        # Create PRP request
        request = PRPRequest(
            feature_description=feature_description,
            examples=examples,
            documentation_urls=documentation_urls,
            considerations=considerations
        )
        
        # Generate PRP
        generator = PRPGenerator()
        prp_content = generator.generate_prp(request)
        
        # Determine filename
        if not prp_name:
            # Generate name from feature description
            prp_name = feature_description.lower()
            prp_name = prp_name.replace(' ', '_')[:50]
            prp_name = ''.join(c for c in prp_name if c.isalnum() or c == '_')
        
        # Save PRP
        saved_path = generator.save_prp(prp_content, prp_name)
        
        return {
            "status": "success",
            "action": "generated",
            "prp_path": str(saved_path),
            "prp_name": prp_name,
            "preview": prp_content[:500] + "...",
            "message": f"PRP generated and saved to {saved_path}"
        }
    
    def _execute_prp(self, prp_path: Optional[str]) -> Dict[str, Any]:
        """Execute an existing PRP."""
        if not prp_path:
            return {
                "status": "error",
                "error": "PRP path is required for execution"
            }
        
        # Check if path exists
        path = Path(prp_path)
        if not path.exists():
            # Try in PRPs directory
            path = Path("PRPs") / prp_path
            if not path.exists() and not prp_path.endswith('.md'):
                path = Path("PRPs") / f"{prp_path}.md"
        
        if not path.exists():
            return {
                "status": "error",
                "error": f"PRP not found: {prp_path}"
            }
        
        # Execute PRP
        executor = PRPExecutor()
        result = executor.execute_prp(str(path))
        
        return {
            "status": result["status"],
            "action": "executed",
            "prp_path": str(path),
            "tasks_completed": result["tasks_completed"],
            "validation_passed": result["validation_result"].success,
            "message": f"PRP execution {'completed successfully' if result['status'] == 'success' else 'failed'}",
            "details": result.get("validation_result", {})
        }