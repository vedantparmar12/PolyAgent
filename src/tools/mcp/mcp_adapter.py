"""MCP adapter for converting between MCP and internal tool formats"""

from typing import Dict, Any, List, Optional, Type
from ..base_tool import BaseTool, ToolParameter, ToolResult
from ..tool_registry import tool_registry
import json


class MCPToolAdapter:
    """Adapter for converting between MCP protocol and internal tool format"""
    
    @staticmethod
    def tool_to_mcp_schema(tool: BaseTool) -> Dict[str, Any]:
        """Convert tool to MCP schema format
        
        Args:
            tool: Tool instance
            
        Returns:
            MCP-compatible schema
        """
        # Get base schema
        schema = tool.get_schema()
        
        # Convert to MCP format
        mcp_schema = {
            "name": schema["name"],
            "description": schema["description"],
            "version": schema["version"],
            "inputSchema": {
                "type": "object",
                "properties": {},
                "required": []
            },
            "outputSchema": {
                "type": "object",
                "properties": {
                    "success": {"type": "boolean"},
                    "data": {"type": "object"},
                    "error": {"type": "string"}
                },
                "required": ["success"]
            },
            "metadata": {
                "category": schema["category"],
                "requiresConfirmation": schema["requires_confirmation"],
                "costEstimate": schema["cost_estimate"],
                "rateLimit": schema["rate_limit"]
            }
        }
        
        # Convert parameters
        for param_name, param_schema in schema["parameters"]["properties"].items():
            mcp_param = {
                "type": MCPToolAdapter._convert_type_to_mcp(param_schema["type"]),
                "description": param_schema["description"]
            }
            
            # Add constraints
            if "enum" in param_schema:
                mcp_param["enum"] = param_schema["enum"]
            if "minimum" in param_schema:
                mcp_param["minimum"] = param_schema["minimum"]
            if "maximum" in param_schema:
                mcp_param["maximum"] = param_schema["maximum"]
            if "pattern" in param_schema:
                mcp_param["pattern"] = param_schema["pattern"]
            if "default" in param_schema:
                mcp_param["default"] = param_schema["default"]
            
            mcp_schema["inputSchema"]["properties"][param_name] = mcp_param
        
        # Set required parameters
        mcp_schema["inputSchema"]["required"] = schema["parameters"].get("required", [])
        
        return mcp_schema
    
    @staticmethod
    def mcp_to_tool_params(mcp_params: Dict[str, Any], tool: BaseTool) -> Dict[str, Any]:
        """Convert MCP parameters to tool parameters
        
        Args:
            mcp_params: MCP format parameters
            tool: Target tool
            
        Returns:
            Tool-compatible parameters
        """
        # Validate against tool parameters
        tool_params = {}
        
        for param in tool.parameters:
            if param.name in mcp_params:
                value = mcp_params[param.name]
                
                # Type conversion if needed
                if param.type == "integer" and isinstance(value, str):
                    try:
                        value = int(value)
                    except ValueError:
                        pass
                elif param.type == "number" and isinstance(value, str):
                    try:
                        value = float(value)
                    except ValueError:
                        pass
                elif param.type == "boolean" and isinstance(value, str):
                    value = value.lower() in ["true", "1", "yes", "on"]
                
                tool_params[param.name] = value
        
        return tool_params
    
    @staticmethod
    def tool_result_to_mcp(result: ToolResult) -> Dict[str, Any]:
        """Convert tool result to MCP format
        
        Args:
            result: Tool execution result
            
        Returns:
            MCP-compatible result
        """
        mcp_result = {
            "success": result.success,
            "timestamp": result.execution_time_ms,
            "metadata": result.metadata
        }
        
        if result.success:
            mcp_result["data"] = result.data
        else:
            mcp_result["error"] = {
                "message": result.error,
                "type": "execution_error"
            }
        
        return mcp_result
    
    @staticmethod
    def create_tool_from_mcp(mcp_definition: Dict[str, Any]) -> Type[BaseTool]:
        """Create a tool class from MCP definition
        
        Args:
            mcp_definition: MCP tool definition
            
        Returns:
            Tool class
        """
        name = mcp_definition["name"]
        description = mcp_definition.get("description", "")
        input_schema = mcp_definition.get("inputSchema", {})
        metadata = mcp_definition.get("metadata", {})
        
        # Extract parameters
        parameters = []
        properties = input_schema.get("properties", {})
        required = input_schema.get("required", [])
        
        for param_name, param_schema in properties.items():
            param = ToolParameter(
                name=param_name,
                type=MCPToolAdapter._convert_type_from_mcp(param_schema.get("type", "string")),
                description=param_schema.get("description", ""),
                required=param_name in required,
                default=param_schema.get("default"),
                enum=param_schema.get("enum"),
                min_value=param_schema.get("minimum"),
                max_value=param_schema.get("maximum"),
                pattern=param_schema.get("pattern")
            )
            parameters.append(param)
        
        # Create tool class
        class MCPTool(BaseTool):
            @property
            def name(self) -> str:
                return name
            
            @property
            def description(self) -> str:
                return description
            
            @property
            def category(self) -> str:
                return metadata.get("category", "mcp")
            
            @property
            def parameters(self) -> List[ToolParameter]:
                return parameters
            
            @property
            def requires_confirmation(self) -> bool:
                return metadata.get("requiresConfirmation", False)
            
            @property
            def cost_estimate(self) -> Optional[float]:
                return metadata.get("costEstimate")
            
            @property
            def rate_limit(self) -> Optional[Dict[str, int]]:
                return metadata.get("rateLimit")
            
            async def _execute(self, params: Dict[str, Any]) -> Any:
                # This would need to be implemented based on MCP execution endpoint
                raise NotImplementedError("MCP tool execution not implemented")
        
        # Set class name
        MCPTool.__name__ = f"{name.title().replace(' ', '').replace('-', '')}MCPTool"
        
        return MCPTool
    
    @staticmethod
    def export_registry_to_mcp(file_path: str) -> None:
        """Export tool registry to MCP format
        
        Args:
            file_path: Path to save MCP definitions
        """
        mcp_tools = []
        
        for tool_name in tool_registry.list_tools():
            tool = tool_registry.get_tool(tool_name)
            if tool:
                mcp_schema = MCPToolAdapter.tool_to_mcp_schema(tool)
                mcp_tools.append(mcp_schema)
        
        mcp_manifest = {
            "version": "1.0.0",
            "tools": mcp_tools,
            "metadata": {
                "exported_at": datetime.utcnow().isoformat(),
                "total_tools": len(mcp_tools)
            }
        }
        
        with open(file_path, 'w') as f:
            json.dump(mcp_manifest, f, indent=2)
    
    @staticmethod
    def import_mcp_tools(file_path: str) -> int:
        """Import tools from MCP format
        
        Args:
            file_path: Path to MCP definitions file
            
        Returns:
            Number of tools imported
        """
        with open(file_path, 'r') as f:
            mcp_manifest = json.load(f)
        
        imported = 0
        tools = mcp_manifest.get("tools", [])
        
        for mcp_tool in tools:
            try:
                tool_class = MCPToolAdapter.create_tool_from_mcp(mcp_tool)
                tool_registry.register(tool_class)
                imported += 1
            except Exception as e:
                print(f"Failed to import tool {mcp_tool.get('name')}: {e}")
        
        return imported
    
    @staticmethod
    def _convert_type_to_mcp(tool_type: str) -> str:
        """Convert tool type to MCP type"""
        type_mapping = {
            "string": "string",
            "number": "number",
            "integer": "integer",
            "boolean": "boolean",
            "object": "object",
            "array": "array"
        }
        return type_mapping.get(tool_type, "string")
    
    @staticmethod
    def _convert_type_from_mcp(mcp_type: str) -> str:
        """Convert MCP type to tool type"""
        type_mapping = {
            "string": "string",
            "number": "number",
            "integer": "integer",
            "boolean": "boolean",
            "object": "object",
            "array": "array"
        }
        return type_mapping.get(mcp_type, "string")


from datetime import datetime