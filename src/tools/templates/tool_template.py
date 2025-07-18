"""Tool template system for creating new tools"""

from typing import Dict, Any, List, Optional
from pathlib import Path
import yaml
import json
from jinja2 import Template
from ..base_tool import BaseTool, ToolParameter


class ToolTemplate:
    """Template for creating new tools"""
    
    def __init__(self, template_data: Dict[str, Any]):
        """Initialize tool template
        
        Args:
            template_data: Template configuration
        """
        self.name = template_data.get("name", "unnamed_template")
        self.description = template_data.get("description", "")
        self.category = template_data.get("category", "custom")
        self.version = template_data.get("version", "1.0.0")
        self.parameters = template_data.get("parameters", [])
        self.code_template = template_data.get("code_template", "")
        self.examples = template_data.get("examples", [])
        self.metadata = template_data.get("metadata", {})
    
    def generate_tool_code(
        self,
        tool_name: str,
        customization: Dict[str, Any]
    ) -> str:
        """Generate tool code from template
        
        Args:
            tool_name: Name for the new tool
            customization: Customization parameters
            
        Returns:
            Generated Python code
        """
        # Default values
        context = {
            "tool_name": tool_name,
            "tool_class_name": self._to_class_name(tool_name),
            "description": customization.get("description", self.description),
            "category": customization.get("category", self.category),
            "parameters": self._process_parameters(customization.get("parameters", self.parameters)),
            "imports": customization.get("imports", []),
            "methods": customization.get("methods", {}),
            **customization
        }
        
        # Use Jinja2 template if code_template contains template syntax
        if "{{" in self.code_template:
            template = Template(self.code_template)
            return template.render(**context)
        else:
            # Use simple string replacement
            code = self.code_template
            for key, value in context.items():
                code = code.replace(f"{{{key}}}", str(value))
            return code
    
    def _to_class_name(self, tool_name: str) -> str:
        """Convert tool name to class name"""
        parts = tool_name.replace("-", "_").split("_")
        return "".join(part.capitalize() for part in parts) + "Tool"
    
    def _process_parameters(self, parameters: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process parameter definitions"""
        processed = []
        
        for param in parameters:
            # Ensure all required fields
            processed_param = {
                "name": param.get("name", "param"),
                "type": param.get("type", "string"),
                "description": param.get("description", "Parameter description"),
                "required": param.get("required", True),
                "default": param.get("default"),
                "enum": param.get("enum"),
                "min_value": param.get("min_value"),
                "max_value": param.get("max_value"),
                "pattern": param.get("pattern")
            }
            processed.append(processed_param)
        
        return processed
    
    def validate(self) -> Dict[str, Any]:
        """Validate template configuration
        
        Returns:
            Validation results
        """
        validation = {
            "valid": True,
            "errors": [],
            "warnings": []
        }
        
        # Check required fields
        if not self.name:
            validation["errors"].append("Template name is required")
            validation["valid"] = False
        
        if not self.code_template:
            validation["errors"].append("Code template is required")
            validation["valid"] = False
        
        # Check parameters
        for i, param in enumerate(self.parameters):
            if "name" not in param:
                validation["errors"].append(f"Parameter {i} missing name")
                validation["valid"] = False
            
            if "type" not in param:
                validation["warnings"].append(f"Parameter {param.get('name', i)} missing type, defaulting to 'string'")
        
        return validation
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert template to dictionary
        
        Returns:
            Template data
        """
        return {
            "name": self.name,
            "description": self.description,
            "category": self.category,
            "version": self.version,
            "parameters": self.parameters,
            "code_template": self.code_template,
            "examples": self.examples,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_file(cls, file_path: Path) -> "ToolTemplate":
        """Load template from file
        
        Args:
            file_path: Path to template file
            
        Returns:
            Tool template
        """
        with open(file_path, 'r') as f:
            if file_path.suffix == ".yaml" or file_path.suffix == ".yml":
                data = yaml.safe_load(f)
            elif file_path.suffix == ".json":
                data = json.load(f)
            else:
                raise ValueError(f"Unsupported file format: {file_path.suffix}")
        
        return cls(data)
    
    def save_to_file(self, file_path: Path) -> None:
        """Save template to file
        
        Args:
            file_path: Path to save template
        """
        data = self.to_dict()
        
        with open(file_path, 'w') as f:
            if file_path.suffix == ".yaml" or file_path.suffix == ".yml":
                yaml.dump(data, f, default_flow_style=False)
            elif file_path.suffix == ".json":
                json.dump(data, f, indent=2)
            else:
                raise ValueError(f"Unsupported file format: {file_path.suffix}")


class ToolTemplateLibrary:
    """Library of tool templates"""
    
    def __init__(self, templates_dir: Optional[Path] = None):
        """Initialize template library
        
        Args:
            templates_dir: Directory containing templates
        """
        self.templates_dir = templates_dir or Path(__file__).parent / "library"
        self._templates: Dict[str, ToolTemplate] = {}
        self._load_templates()
    
    def _load_templates(self) -> None:
        """Load all templates from directory"""
        if not self.templates_dir.exists():
            self.templates_dir.mkdir(parents=True, exist_ok=True)
            self._create_default_templates()
        
        for template_file in self.templates_dir.glob("*.yaml"):
            try:
                template = ToolTemplate.from_file(template_file)
                self._templates[template.name] = template
            except Exception as e:
                print(f"Failed to load template {template_file}: {e}")
        
        for template_file in self.templates_dir.glob("*.json"):
            try:
                template = ToolTemplate.from_file(template_file)
                self._templates[template.name] = template
            except Exception as e:
                print(f"Failed to load template {template_file}: {e}")
    
    def _create_default_templates(self) -> None:
        """Create default tool templates"""
        # API Client Template
        api_client_template = ToolTemplate({
            "name": "api_client",
            "description": "Template for creating API client tools",
            "category": "api",
            "parameters": [
                {
                    "name": "endpoint",
                    "type": "string",
                    "description": "API endpoint URL",
                    "required": True
                },
                {
                    "name": "method",
                    "type": "string",
                    "description": "HTTP method",
                    "required": False,
                    "default": "GET",
                    "enum": ["GET", "POST", "PUT", "DELETE"]
                },
                {
                    "name": "data",
                    "type": "object",
                    "description": "Request data",
                    "required": False
                }
            ],
            "code_template": '''"""{{ description }}"""

from typing import List, Dict, Any, Optional
import httpx
from ..base_tool import BaseTool, ToolParameter, ToolResult


class {{ tool_class_name }}(BaseTool):
    """{{ description }}"""
    
    @property
    def name(self) -> str:
        return "{{ tool_name }}"
    
    @property
    def description(self) -> str:
        return "{{ description }}"
    
    @property
    def category(self) -> str:
        return "{{ category }}"
    
    @property
    def parameters(self) -> List[ToolParameter]:
        return [
            {% for param in parameters %}
            ToolParameter(
                name="{{ param.name }}",
                type="{{ param.type }}",
                description="{{ param.description }}",
                required={{ param.required }},
                {% if param.default is not none %}default={{ param.default | tojson }},{% endif %}
                {% if param.enum %}enum={{ param.enum | tojson }},{% endif %}
            ),
            {% endfor %}
        ]
    
    async def _execute(self, params: Dict[str, Any]) -> ToolResult:
        """Execute API request"""
        endpoint = params["endpoint"]
        method = params.get("method", "GET")
        data = params.get("data")
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.request(
                    method=method,
                    url=endpoint,
                    json=data if method in ["POST", "PUT"] else None,
                    params=data if method == "GET" else None
                )
                response.raise_for_status()
                
                return ToolResult(
                    success=True,
                    data=response.json() if response.headers.get("content-type", "").startswith("application/json") else response.text
                )
                
        except httpx.HTTPError as e:
            return ToolResult(
                success=False,
                error=f"HTTP error: {str(e)}"
            )
        except Exception as e:
            return ToolResult(
                success=False,
                error=f"Request failed: {str(e)}"
            )
'''
        })
        api_client_template.save_to_file(self.templates_dir / "api_client.yaml")
        
        # Data Processor Template
        data_processor_template = ToolTemplate({
            "name": "data_processor",
            "description": "Template for creating data processing tools",
            "category": "data",
            "parameters": [
                {
                    "name": "input_data",
                    "type": "object",
                    "description": "Input data to process",
                    "required": True
                },
                {
                    "name": "operation",
                    "type": "string",
                    "description": "Processing operation",
                    "required": True
                }
            ],
            "code_template": '''"""{{ description }}"""

from typing import List, Dict, Any, Optional
from ..base_tool import BaseTool, ToolParameter, ToolResult


class {{ tool_class_name }}(BaseTool):
    """{{ description }}"""
    
    @property
    def name(self) -> str:
        return "{{ tool_name }}"
    
    @property
    def description(self) -> str:
        return "{{ description }}"
    
    @property
    def category(self) -> str:
        return "{{ category }}"
    
    @property
    def parameters(self) -> List[ToolParameter]:
        return [
            {% for param in parameters %}
            ToolParameter(
                name="{{ param.name }}",
                type="{{ param.type }}",
                description="{{ param.description }}",
                required={{ param.required }}
            ),
            {% endfor %}
        ]
    
    async def _execute(self, params: Dict[str, Any]) -> ToolResult:
        """Process data"""
        input_data = params["input_data"]
        operation = params["operation"]
        
        try:
            # Implement processing logic here
            result = self._process_data(input_data, operation)
            
            return ToolResult(
                success=True,
                data=result
            )
            
        except Exception as e:
            return ToolResult(
                success=False,
                error=f"Processing failed: {str(e)}"
            )
    
    def _process_data(self, data: Any, operation: str) -> Any:
        """Process data based on operation"""
        # Implement processing logic
        return data
'''
        })
        data_processor_template.save_to_file(self.templates_dir / "data_processor.yaml")
    
    def get_template(self, name: str) -> Optional[ToolTemplate]:
        """Get a template by name
        
        Args:
            name: Template name
            
        Returns:
            Tool template or None
        """
        return self._templates.get(name)
    
    def list_templates(self) -> List[str]:
        """List available templates
        
        Returns:
            List of template names
        """
        return list(self._templates.keys())
    
    def add_template(self, template: ToolTemplate) -> None:
        """Add a template to the library
        
        Args:
            template: Tool template
        """
        self._templates[template.name] = template
        template.save_to_file(self.templates_dir / f"{template.name}.yaml")
    
    def create_tool_from_template(
        self,
        template_name: str,
        tool_name: str,
        customization: Dict[str, Any]
    ) -> Optional[str]:
        """Create tool code from template
        
        Args:
            template_name: Name of template
            tool_name: Name for new tool
            customization: Customization parameters
            
        Returns:
            Generated tool code or None
        """
        template = self.get_template(template_name)
        if not template:
            return None
        
        return template.generate_tool_code(tool_name, customization)