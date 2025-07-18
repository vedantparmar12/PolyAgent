"""Base tool class for all agent tools"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union
from pydantic import BaseModel, Field
from datetime import datetime
import logfire


class ToolParameter(BaseModel):
    """Tool parameter definition"""
    name: str = Field(description="Parameter name")
    type: str = Field(description="Parameter type (string, number, boolean, etc)")
    description: str = Field(description="Parameter description")
    required: bool = Field(default=True, description="Whether parameter is required")
    default: Optional[Any] = Field(default=None, description="Default value if not required")
    enum: Optional[List[Any]] = Field(default=None, description="Allowed values")
    min_value: Optional[Union[int, float]] = Field(default=None, description="Minimum value for numbers")
    max_value: Optional[Union[int, float]] = Field(default=None, description="Maximum value for numbers")
    pattern: Optional[str] = Field(default=None, description="Regex pattern for strings")


class ToolResult(BaseModel):
    """Standard tool execution result"""
    success: bool = Field(description="Whether execution was successful")
    data: Optional[Any] = Field(default=None, description="Result data")
    error: Optional[str] = Field(default=None, description="Error message if failed")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    execution_time_ms: Optional[float] = Field(default=None, description="Execution time in milliseconds")
    
    class Config:
        arbitrary_types_allowed = True


class BaseTool(ABC):
    """Base class for all tools"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize tool with optional configuration
        
        Args:
            config: Tool-specific configuration
        """
        self.config = config or {}
        self._logger = logfire.span(f"tool.{self.name}")
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Tool name"""
        pass
    
    @property
    @abstractmethod
    def description(self) -> str:
        """Tool description"""
        pass
    
    @property
    @abstractmethod
    def parameters(self) -> List[ToolParameter]:
        """Tool parameters"""
        pass
    
    @property
    def category(self) -> str:
        """Tool category for organization"""
        return "general"
    
    @property
    def version(self) -> str:
        """Tool version"""
        return "1.0.0"
    
    @property
    def requires_confirmation(self) -> bool:
        """Whether tool requires user confirmation before execution"""
        return False
    
    @property
    def cost_estimate(self) -> Optional[float]:
        """Estimated cost per execution (if applicable)"""
        return None
    
    @property
    def rate_limit(self) -> Optional[Dict[str, int]]:
        """Rate limiting configuration"""
        return None
    
    def validate_parameters(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and transform parameters
        
        Args:
            params: Raw parameters
            
        Returns:
            Validated parameters
            
        Raises:
            ValueError: If validation fails
        """
        validated = {}
        
        # Check required parameters
        for param in self.parameters:
            if param.required and param.name not in params:
                raise ValueError(f"Required parameter '{param.name}' is missing")
            
            if param.name in params:
                value = params[param.name]
                
                # Type validation
                if param.type == "string" and not isinstance(value, str):
                    value = str(value)
                elif param.type == "number" and not isinstance(value, (int, float)):
                    try:
                        value = float(value)
                    except ValueError:
                        raise ValueError(f"Parameter '{param.name}' must be a number")
                elif param.type == "integer" and not isinstance(value, int):
                    try:
                        value = int(value)
                    except ValueError:
                        raise ValueError(f"Parameter '{param.name}' must be an integer")
                elif param.type == "boolean" and not isinstance(value, bool):
                    value = str(value).lower() in ["true", "1", "yes", "on"]
                
                # Enum validation
                if param.enum and value not in param.enum:
                    raise ValueError(f"Parameter '{param.name}' must be one of {param.enum}")
                
                # Range validation
                if param.type in ["number", "integer"]:
                    if param.min_value is not None and value < param.min_value:
                        raise ValueError(f"Parameter '{param.name}' must be >= {param.min_value}")
                    if param.max_value is not None and value > param.max_value:
                        raise ValueError(f"Parameter '{param.name}' must be <= {param.max_value}")
                
                # Pattern validation
                if param.type == "string" and param.pattern:
                    import re
                    if not re.match(param.pattern, value):
                        raise ValueError(f"Parameter '{param.name}' does not match pattern {param.pattern}")
                
                validated[param.name] = value
            elif param.default is not None:
                validated[param.name] = param.default
        
        return validated
    
    async def execute(self, **params) -> ToolResult:
        """Execute the tool
        
        Args:
            **params: Tool parameters
            
        Returns:
            Tool execution result
        """
        start_time = datetime.utcnow()
        
        try:
            # Validate parameters
            validated_params = self.validate_parameters(params)
            
            # Log execution start
            self._logger.info(
                "executing_tool",
                tool_name=self.name,
                params=validated_params
            )
            
            # Execute tool logic
            result = await self._execute(validated_params)
            
            # Calculate execution time
            execution_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            # Ensure result is a ToolResult
            if not isinstance(result, ToolResult):
                result = ToolResult(
                    success=True,
                    data=result,
                    execution_time_ms=execution_time
                )
            else:
                result.execution_time_ms = execution_time
            
            # Log success
            self._logger.info(
                "tool_execution_success",
                tool_name=self.name,
                execution_time_ms=execution_time
            )
            
            return result
            
        except Exception as e:
            # Calculate execution time
            execution_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            # Log error
            self._logger.error(
                "tool_execution_failed",
                tool_name=self.name,
                error=str(e),
                execution_time_ms=execution_time
            )
            
            return ToolResult(
                success=False,
                error=str(e),
                execution_time_ms=execution_time
            )
    
    @abstractmethod
    async def _execute(self, params: Dict[str, Any]) -> Union[ToolResult, Any]:
        """Execute tool logic
        
        Args:
            params: Validated parameters
            
        Returns:
            Execution result
        """
        pass
    
    def get_schema(self) -> Dict[str, Any]:
        """Get tool schema for MCP or other protocols
        
        Returns:
            Tool schema
        """
        return {
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "category": self.category,
            "parameters": {
                "type": "object",
                "properties": {
                    param.name: {
                        "type": param.type,
                        "description": param.description,
                        "enum": param.enum,
                        "minimum": param.min_value,
                        "maximum": param.max_value,
                        "pattern": param.pattern,
                        "default": param.default
                    }
                    for param in self.parameters
                },
                "required": [param.name for param in self.parameters if param.required]
            },
            "requires_confirmation": self.requires_confirmation,
            "cost_estimate": self.cost_estimate,
            "rate_limit": self.rate_limit
        }
    
    def get_usage_example(self) -> Dict[str, Any]:
        """Get usage example for the tool
        
        Returns:
            Usage example
        """
        example_params = {}
        for param in self.parameters:
            if param.enum:
                example_params[param.name] = param.enum[0]
            elif param.type == "string":
                example_params[param.name] = f"example_{param.name}"
            elif param.type == "number":
                example_params[param.name] = 1.0
            elif param.type == "integer":
                example_params[param.name] = 1
            elif param.type == "boolean":
                example_params[param.name] = True
            elif param.default is not None:
                example_params[param.name] = param.default
        
        return {
            "tool": self.name,
            "params": example_params,
            "expected_result": {
                "success": True,
                "data": "Example result data"
            }
        }