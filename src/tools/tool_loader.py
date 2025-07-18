"""Tool loader for dynamic tool loading and management"""

import json
import yaml
from pathlib import Path
from typing import Dict, Any, List, Optional, Type
import importlib.util
import inspect
from .base_tool import BaseTool
from .tool_registry import tool_registry
import logfire


class ToolLoader:
    """Loader for dynamically loading tools from various sources"""
    
    def __init__(self, base_path: Optional[Path] = None):
        """Initialize tool loader
        
        Args:
            base_path: Base path for tool discovery
        """
        self.base_path = base_path or Path(__file__).parent
        self._logger = logfire.span("tool_loader")
        self._loaded_modules: Dict[str, Any] = {}
    
    def load_from_directory(self, directory: Path) -> int:
        """Load all tools from a directory
        
        Args:
            directory: Directory containing tool implementations
            
        Returns:
            Number of tools loaded
        """
        if not directory.exists():
            self._logger.warning(f"Directory not found: {directory}")
            return 0
        
        loaded = 0
        
        # Load Python files
        for py_file in directory.glob("*.py"):
            if py_file.name.startswith("_") or py_file.name in ["base_tool.py", "__init__.py"]:
                continue
            
            try:
                loaded += self._load_python_file(py_file)
            except Exception as e:
                self._logger.error(f"Failed to load {py_file}: {e}")
        
        # Load tool definitions
        for config_file in directory.glob("*.yaml"):
            try:
                loaded += self._load_yaml_config(config_file)
            except Exception as e:
                self._logger.error(f"Failed to load {config_file}: {e}")
        
        for config_file in directory.glob("*.json"):
            try:
                loaded += self._load_json_config(config_file)
            except Exception as e:
                self._logger.error(f"Failed to load {config_file}: {e}")
        
        self._logger.info(f"Loaded {loaded} tools from {directory}")
        return loaded
    
    def load_from_config(self, config_path: Path) -> int:
        """Load tools from a configuration file
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            Number of tools loaded
        """
        if not config_path.exists():
            self._logger.warning(f"Config file not found: {config_path}")
            return 0
        
        if config_path.suffix == ".yaml" or config_path.suffix == ".yml":
            return self._load_yaml_config(config_path)
        elif config_path.suffix == ".json":
            return self._load_json_config(config_path)
        else:
            self._logger.warning(f"Unsupported config format: {config_path.suffix}")
            return 0
    
    def load_from_module(self, module_path: str) -> int:
        """Load tools from a Python module
        
        Args:
            module_path: Python module path (e.g., 'mytools.custom')
            
        Returns:
            Number of tools loaded
        """
        try:
            module = importlib.import_module(module_path)
            return self._load_tools_from_module(module, module_path)
        except ImportError as e:
            self._logger.error(f"Failed to import module {module_path}: {e}")
            return 0
    
    def create_tool_from_template(
        self,
        template_name: str,
        tool_name: str,
        customization: Dict[str, Any]
    ) -> Optional[Type[BaseTool]]:
        """Create a tool from a template
        
        Args:
            template_name: Name of the template
            tool_name: Name for the new tool
            customization: Customization parameters
            
        Returns:
            Tool class or None if failed
        """
        template = self._load_template(template_name)
        if not template:
            return None
        
        # Generate tool code from template
        tool_code = self._generate_tool_code(template, tool_name, customization)
        
        # Create tool class dynamically
        return self._create_tool_class(tool_name, tool_code)
    
    def _load_python_file(self, file_path: Path) -> int:
        """Load tools from a Python file
        
        Args:
            file_path: Path to Python file
            
        Returns:
            Number of tools loaded
        """
        # Import module dynamically
        module_name = file_path.stem
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        
        if not spec or not spec.loader:
            return 0
        
        module = importlib.util.module_from_spec(spec)
        self._loaded_modules[module_name] = module
        spec.loader.exec_module(module)
        
        return self._load_tools_from_module(module, str(file_path))
    
    def _load_tools_from_module(self, module: Any, source: str) -> int:
        """Load tool classes from a module
        
        Args:
            module: Python module
            source: Source identifier
            
        Returns:
            Number of tools loaded
        """
        loaded = 0
        
        for name, obj in inspect.getmembers(module):
            if (inspect.isclass(obj) and 
                issubclass(obj, BaseTool) and 
                obj != BaseTool and
                not inspect.isabstract(obj)):
                
                try:
                    tool_registry.register(obj)
                    loaded += 1
                    self._logger.info(f"Loaded tool {obj.__name__} from {source}")
                except Exception as e:
                    self._logger.error(f"Failed to register tool {obj.__name__}: {e}")
        
        return loaded
    
    def _load_yaml_config(self, config_path: Path) -> int:
        """Load tools from YAML configuration
        
        Args:
            config_path: Path to YAML file
            
        Returns:
            Number of tools loaded
        """
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        return self._load_from_config_data(config, str(config_path))
    
    def _load_json_config(self, config_path: Path) -> int:
        """Load tools from JSON configuration
        
        Args:
            config_path: Path to JSON file
            
        Returns:
            Number of tools loaded
        """
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        return self._load_from_config_data(config, str(config_path))
    
    def _load_from_config_data(self, config: Dict[str, Any], source: str) -> int:
        """Load tools from configuration data
        
        Args:
            config: Configuration data
            source: Source identifier
            
        Returns:
            Number of tools loaded
        """
        loaded = 0
        
        # Check for tool definitions
        tools = config.get("tools", [])
        for tool_def in tools:
            if self._create_tool_from_definition(tool_def):
                loaded += 1
        
        # Check for tool references
        imports = config.get("imports", [])
        for import_path in imports:
            loaded += self.load_from_module(import_path)
        
        # Check for directories
        directories = config.get("directories", [])
        for directory in directories:
            dir_path = Path(directory)
            if not dir_path.is_absolute():
                dir_path = Path(source).parent / dir_path
            loaded += self.load_from_directory(dir_path)
        
        return loaded
    
    def _create_tool_from_definition(self, definition: Dict[str, Any]) -> bool:
        """Create a tool from a configuration definition
        
        Args:
            definition: Tool definition
            
        Returns:
            True if successful
        """
        try:
            # Extract tool information
            name = definition.get("name")
            description = definition.get("description", "")
            category = definition.get("category", "general")
            parameters = definition.get("parameters", [])
            implementation = definition.get("implementation", {})
            
            if not name:
                return False
            
            # Create tool class dynamically
            tool_class = self._create_configured_tool_class(
                name, description, category, parameters, implementation
            )
            
            if tool_class:
                tool_registry.register(tool_class)
                return True
                
        except Exception as e:
            self._logger.error(f"Failed to create tool from definition: {e}")
        
        return False
    
    def _create_configured_tool_class(
        self,
        name: str,
        description: str,
        category: str,
        parameters: List[Dict[str, Any]],
        implementation: Dict[str, Any]
    ) -> Optional[Type[BaseTool]]:
        """Create a configured tool class
        
        Args:
            name: Tool name
            description: Tool description
            category: Tool category
            parameters: Parameter definitions
            implementation: Implementation configuration
            
        Returns:
            Tool class or None
        """
        from .base_tool import ToolParameter
        
        # Create parameter objects
        param_objects = []
        for param_def in parameters:
            param_objects.append(ToolParameter(**param_def))
        
        # Create tool class
        class ConfiguredTool(BaseTool):
            @property
            def name(self) -> str:
                return name
            
            @property
            def description(self) -> str:
                return description
            
            @property
            def category(self) -> str:
                return category
            
            @property
            def parameters(self) -> List[ToolParameter]:
                return param_objects
            
            async def _execute(self, params: Dict[str, Any]) -> Any:
                # Implementation based on configuration
                impl_type = implementation.get("type", "noop")
                
                if impl_type == "noop":
                    return {"message": "No operation"}
                elif impl_type == "http":
                    return await self._execute_http(params, implementation)
                elif impl_type == "command":
                    return await self._execute_command(params, implementation)
                elif impl_type == "script":
                    return await self._execute_script(params, implementation)
                else:
                    raise NotImplementedError(f"Implementation type '{impl_type}' not supported")
            
            async def _execute_http(self, params: Dict[str, Any], config: Dict[str, Any]) -> Any:
                """Execute HTTP request"""
                import httpx
                
                url = config.get("url", "")
                method = config.get("method", "GET")
                headers = config.get("headers", {})
                
                # Substitute parameters in URL
                for key, value in params.items():
                    url = url.replace(f"{{{key}}}", str(value))
                
                async with httpx.AsyncClient() as client:
                    response = await client.request(
                        method=method,
                        url=url,
                        headers=headers,
                        json=params if method in ["POST", "PUT"] else None,
                        params=params if method == "GET" else None
                    )
                    response.raise_for_status()
                    return response.json()
            
            async def _execute_command(self, params: Dict[str, Any], config: Dict[str, Any]) -> Any:
                """Execute system command"""
                import asyncio
                import shlex
                
                command = config.get("command", "")
                
                # Substitute parameters
                for key, value in params.items():
                    command = command.replace(f"{{{key}}}", shlex.quote(str(value)))
                
                proc = await asyncio.create_subprocess_shell(
                    command,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                
                stdout, stderr = await proc.communicate()
                
                if proc.returncode != 0:
                    raise RuntimeError(f"Command failed: {stderr.decode()}")
                
                return {"output": stdout.decode(), "return_code": proc.returncode}
            
            async def _execute_script(self, params: Dict[str, Any], config: Dict[str, Any]) -> Any:
                """Execute script"""
                script_type = config.get("script_type", "python")
                script = config.get("script", "")
                
                if script_type == "python":
                    # Execute Python script in isolated namespace
                    namespace = {"params": params, "result": None}
                    exec(script, namespace)
                    return namespace.get("result", {})
                else:
                    raise NotImplementedError(f"Script type '{script_type}' not supported")
        
        # Set class name
        ConfiguredTool.__name__ = f"{name.title().replace(' ', '')}Tool"
        
        return ConfiguredTool
    
    def _load_template(self, template_name: str) -> Optional[Dict[str, Any]]:
        """Load a tool template
        
        Args:
            template_name: Name of the template
            
        Returns:
            Template data or None
        """
        template_path = self.base_path / "templates" / f"{template_name}.yaml"
        
        if not template_path.exists():
            template_path = template_path.with_suffix(".json")
        
        if not template_path.exists():
            self._logger.warning(f"Template not found: {template_name}")
            return None
        
        with open(template_path, 'r') as f:
            if template_path.suffix == ".yaml":
                return yaml.safe_load(f)
            else:
                return json.load(f)
    
    def _generate_tool_code(
        self,
        template: Dict[str, Any],
        tool_name: str,
        customization: Dict[str, Any]
    ) -> str:
        """Generate tool code from template
        
        Args:
            template: Template data
            tool_name: Tool name
            customization: Customization parameters
            
        Returns:
            Generated Python code
        """
        # This would use a template engine to generate code
        # For now, return a simple template
        code_template = template.get("code_template", "")
        
        # Replace placeholders
        code = code_template.replace("{{tool_name}}", tool_name)
        
        for key, value in customization.items():
            code = code.replace(f"{{{{{key}}}}}", str(value))
        
        return code
    
    def _create_tool_class(self, tool_name: str, tool_code: str) -> Optional[Type[BaseTool]]:
        """Create a tool class from code
        
        Args:
            tool_name: Tool name
            tool_code: Python code
            
        Returns:
            Tool class or None
        """
        try:
            # Create namespace
            namespace = {
                "BaseTool": BaseTool,
                "ToolParameter": ToolParameter,
                "ToolResult": ToolResult,
                "Dict": Dict,
                "Any": Any,
                "List": List,
                "Optional": Optional
            }
            
            # Execute code
            exec(tool_code, namespace)
            
            # Find tool class
            for name, obj in namespace.items():
                if (inspect.isclass(obj) and 
                    issubclass(obj, BaseTool) and 
                    obj != BaseTool):
                    return obj
            
        except Exception as e:
            self._logger.error(f"Failed to create tool class: {e}")
        
        return None


# Global loader instance
tool_loader = ToolLoader()