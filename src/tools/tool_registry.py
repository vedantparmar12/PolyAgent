"""Tool registry for managing available tools"""

from typing import Dict, List, Optional, Type, Any
from .base_tool import BaseTool
import logfire
from pathlib import Path
import importlib
import inspect


class ToolRegistry:
    """Registry for managing and discovering tools"""
    
    def __init__(self):
        """Initialize tool registry"""
        self._tools: Dict[str, Type[BaseTool]] = {}
        self._instances: Dict[str, BaseTool] = {}
        self._categories: Dict[str, List[str]] = {}
        self._logger = logfire.span("tool_registry")
    
    def register(self, tool_class: Type[BaseTool]) -> None:
        """Register a tool class
        
        Args:
            tool_class: Tool class to register
        """
        # Create temporary instance to get metadata
        temp_instance = tool_class()
        tool_name = temp_instance.name
        category = temp_instance.category
        
        # Register tool
        self._tools[tool_name] = tool_class
        
        # Update category index
        if category not in self._categories:
            self._categories[category] = []
        if tool_name not in self._categories[category]:
            self._categories[category].append(tool_name)
        
        self._logger.info(
            "tool_registered",
            tool_name=tool_name,
            category=category
        )
    
    def unregister(self, tool_name: str) -> None:
        """Unregister a tool
        
        Args:
            tool_name: Name of tool to unregister
        """
        if tool_name in self._tools:
            # Remove from registry
            tool_class = self._tools.pop(tool_name)
            
            # Remove instance if exists
            if tool_name in self._instances:
                self._instances.pop(tool_name)
            
            # Remove from categories
            temp_instance = tool_class()
            category = temp_instance.category
            if category in self._categories:
                self._categories[category].remove(tool_name)
                if not self._categories[category]:
                    self._categories.pop(category)
            
            self._logger.info("tool_unregistered", tool_name=tool_name)
    
    def get_tool(self, tool_name: str, config: Optional[Dict[str, Any]] = None) -> Optional[BaseTool]:
        """Get a tool instance
        
        Args:
            tool_name: Name of the tool
            config: Optional configuration for the tool
            
        Returns:
            Tool instance or None if not found
        """
        if tool_name not in self._tools:
            return None
        
        # Create new instance with config if provided
        if config:
            return self._tools[tool_name](config)
        
        # Return cached instance
        if tool_name not in self._instances:
            self._instances[tool_name] = self._tools[tool_name]()
        
        return self._instances[tool_name]
    
    def list_tools(self, category: Optional[str] = None) -> List[str]:
        """List available tools
        
        Args:
            category: Optional category filter
            
        Returns:
            List of tool names
        """
        if category:
            return self._categories.get(category, [])
        return list(self._tools.keys())
    
    def list_categories(self) -> List[str]:
        """List available categories
        
        Returns:
            List of category names
        """
        return list(self._categories.keys())
    
    def get_tool_info(self, tool_name: str) -> Optional[Dict[str, Any]]:
        """Get information about a tool
        
        Args:
            tool_name: Name of the tool
            
        Returns:
            Tool information or None if not found
        """
        tool = self.get_tool(tool_name)
        if not tool:
            return None
        
        return {
            "name": tool.name,
            "description": tool.description,
            "category": tool.category,
            "version": tool.version,
            "parameters": [param.dict() for param in tool.parameters],
            "requires_confirmation": tool.requires_confirmation,
            "cost_estimate": tool.cost_estimate,
            "rate_limit": tool.rate_limit,
            "schema": tool.get_schema(),
            "example": tool.get_usage_example()
        }
    
    def search_tools(self, query: str) -> List[Dict[str, Any]]:
        """Search for tools by name or description
        
        Args:
            query: Search query
            
        Returns:
            List of matching tools
        """
        query_lower = query.lower()
        results = []
        
        for tool_name in self._tools:
            tool = self.get_tool(tool_name)
            if tool:
                # Search in name and description
                if (query_lower in tool.name.lower() or 
                    query_lower in tool.description.lower()):
                    results.append({
                        "name": tool.name,
                        "description": tool.description,
                        "category": tool.category,
                        "relevance": self._calculate_relevance(query_lower, tool)
                    })
        
        # Sort by relevance
        results.sort(key=lambda x: x["relevance"], reverse=True)
        
        return results
    
    def _calculate_relevance(self, query: str, tool: BaseTool) -> float:
        """Calculate search relevance score
        
        Args:
            query: Search query
            tool: Tool to check
            
        Returns:
            Relevance score
        """
        score = 0.0
        
        # Exact name match
        if query == tool.name.lower():
            score += 1.0
        # Name contains query
        elif query in tool.name.lower():
            score += 0.5
        
        # Description contains query
        if query in tool.description.lower():
            score += 0.3
        
        # Check parameters
        for param in tool.parameters:
            if query in param.name.lower() or query in param.description.lower():
                score += 0.1
        
        return score
    
    def validate_tool_compatibility(self, tool_names: List[str]) -> Dict[str, Any]:
        """Validate if tools can work together
        
        Args:
            tool_names: List of tool names to check
            
        Returns:
            Compatibility analysis
        """
        compatibility = {
            "compatible": True,
            "warnings": [],
            "suggestions": []
        }
        
        tools = []
        for name in tool_names:
            tool = self.get_tool(name)
            if tool:
                tools.append(tool)
            else:
                compatibility["warnings"].append(f"Tool '{name}' not found")
        
        # Check for conflicting tools
        categories = [tool.category for tool in tools]
        if len(set(categories)) == 1 and len(tools) > 3:
            compatibility["warnings"].append(
                f"All tools are from the same category '{categories[0]}'. Consider diversifying."
            )
        
        # Check for complementary tools
        if "data_input" in categories and "data_output" not in categories:
            compatibility["suggestions"].append(
                "Consider adding a data output tool to complete the pipeline"
            )
        
        return compatibility
    
    def export_registry(self) -> Dict[str, Any]:
        """Export registry data
        
        Returns:
            Registry data
        """
        return {
            "tools": {
                name: self.get_tool_info(name)
                for name in self._tools
            },
            "categories": self._categories,
            "total_tools": len(self._tools),
            "total_categories": len(self._categories)
        }
    
    def import_registry(self, data: Dict[str, Any]) -> None:
        """Import registry data
        
        Args:
            data: Registry data to import
        """
        # This would need to handle dynamic tool loading
        # For now, just log
        self._logger.info(
            "registry_import_requested",
            tool_count=len(data.get("tools", {}))
        )
    
    def auto_discover(self, directory: Path) -> int:
        """Auto-discover tools in a directory
        
        Args:
            directory: Directory to search
            
        Returns:
            Number of tools discovered
        """
        discovered = 0
        
        if not directory.exists():
            return discovered
        
        # Search for Python files
        for file_path in directory.glob("*.py"):
            if file_path.name.startswith("_") or file_path.name == "base_tool.py":
                continue
            
            try:
                # Import module
                module_name = file_path.stem
                spec = importlib.util.spec_from_file_location(module_name, file_path)
                if spec and spec.loader:
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    
                    # Find tool classes
                    for name, obj in inspect.getmembers(module):
                        if (inspect.isclass(obj) and 
                            issubclass(obj, BaseTool) and 
                            obj != BaseTool):
                            self.register(obj)
                            discovered += 1
                            
            except Exception as e:
                self._logger.error(
                    "tool_discovery_failed",
                    file=str(file_path),
                    error=str(e)
                )
        
        self._logger.info("auto_discovery_complete", discovered=discovered)
        return discovered


# Global registry instance
tool_registry = ToolRegistry()