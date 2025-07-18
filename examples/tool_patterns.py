"""Tool implementation patterns for Make It Heavy."""

from typing import Dict, Any, Optional
from tools.base_tool import BaseTool
import requests
from pathlib import Path

# Pattern 1: Basic tool implementation
class ExampleTool(BaseTool):
    """Example of a simple tool implementation."""
    
    def __init__(self, config: dict):
        self.config = config
    
    @property
    def name(self) -> str:
        return "example_tool"
    
    @property
    def description(self) -> str:
        return "An example tool that demonstrates basic patterns"
    
    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "input_text": {
                    "type": "string",
                    "description": "Text to process"
                },
                "options": {
                    "type": "object",
                    "description": "Optional configuration",
                    "properties": {
                        "uppercase": {"type": "boolean", "default": False},
                        "reverse": {"type": "boolean", "default": False}
                    }
                }
            },
            "required": ["input_text"]
        }
    
    def execute(self, input_text: str, options: Optional[Dict] = None) -> Dict[str, Any]:
        """Execute the tool with given parameters."""
        try:
            result = input_text
            
            if options:
                if options.get('uppercase'):
                    result = result.upper()
                if options.get('reverse'):
                    result = result[::-1]
            
            return {
                "status": "success",
                "original": input_text,
                "processed": result,
                "options_applied": options or {}
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": f"Processing failed: {str(e)}"
            }

# Pattern 2: Tool with external API calls
class APITool(BaseTool):
    """Tool that makes external API calls with proper error handling."""
    
    def __init__(self, config: dict):
        self.config = config
        self.timeout = config.get('api_timeout', 30)
        self.max_retries = config.get('max_retries', 3)
    
    @property
    def name(self) -> str:
        return "api_tool"
    
    @property
    def description(self) -> str:
        return "Makes API calls with retry logic and error handling"
    
    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "endpoint": {
                    "type": "string",
                    "description": "API endpoint URL"
                },
                "method": {
                    "type": "string",
                    "enum": ["GET", "POST"],
                    "default": "GET"
                },
                "data": {
                    "type": "object",
                    "description": "Data to send (for POST)"
                }
            },
            "required": ["endpoint"]
        }
    
    def execute(self, endpoint: str, method: str = "GET", data: Optional[Dict] = None) -> Dict[str, Any]:
        """Execute API call with retry logic."""
        import time
        
        for attempt in range(self.max_retries):
            try:
                # Validate URL
                if not endpoint.startswith(('http://', 'https://')):
                    return {"status": "error", "error": "Invalid URL format"}
                
                # Make request
                if method == "GET":
                    response = requests.get(endpoint, timeout=self.timeout)
                else:
                    response = requests.post(endpoint, json=data, timeout=self.timeout)
                
                response.raise_for_status()
                
                return {
                    "status": "success",
                    "status_code": response.status_code,
                    "data": response.json() if response.headers.get('content-type', '').startswith('application/json') else response.text
                }
                
            except requests.exceptions.Timeout:
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                    continue
                return {"status": "error", "error": "Request timed out"}
                
            except requests.exceptions.RequestException as e:
                return {"status": "error", "error": f"Request failed: {str(e)}"}
            
            except Exception as e:
                return {"status": "error", "error": f"Unexpected error: {str(e)}"}

# Pattern 3: File operation tool with validation
class FileOperationTool(BaseTool):
    """Tool for file operations with safety checks."""
    
    def __init__(self, config: dict):
        self.config = config
        self.allowed_extensions = config.get('allowed_extensions', ['.txt', '.json', '.md', '.py'])
        self.max_file_size = config.get('max_file_size', 10 * 1024 * 1024)  # 10MB
    
    @property
    def name(self) -> str:
        return "file_operation"
    
    @property
    def description(self) -> str:
        return "Safely read or write files with validation"
    
    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": ["read", "write", "append"],
                    "description": "Operation to perform"
                },
                "path": {
                    "type": "string",
                    "description": "File path (relative to project root)"
                },
                "content": {
                    "type": "string",
                    "description": "Content to write (for write/append operations)"
                }
            },
            "required": ["operation", "path"]
        }
    
    def execute(self, operation: str, path: str, content: Optional[str] = None) -> Dict[str, Any]:
        """Execute file operation with safety checks."""
        try:
            # Validate path
            file_path = Path(path)
            
            # Security check - prevent directory traversal
            if ".." in str(file_path):
                return {"status": "error", "error": "Invalid path - directory traversal not allowed"}
            
            # Check extension
            if file_path.suffix not in self.allowed_extensions:
                return {"status": "error", "error": f"File type {file_path.suffix} not allowed"}
            
            if operation == "read":
                if not file_path.exists():
                    return {"status": "error", "error": "File not found"}
                
                # Check file size
                if file_path.stat().st_size > self.max_file_size:
                    return {"status": "error", "error": "File too large"}
                
                content = file_path.read_text(encoding='utf-8')
                return {
                    "status": "success",
                    "content": content,
                    "size": len(content),
                    "path": str(file_path)
                }
                
            elif operation in ["write", "append"]:
                if content is None:
                    return {"status": "error", "error": "Content required for write/append"}
                
                # Create parent directories if needed
                file_path.parent.mkdir(parents=True, exist_ok=True)
                
                if operation == "write":
                    file_path.write_text(content, encoding='utf-8')
                else:
                    with open(file_path, 'a', encoding='utf-8') as f:
                        f.write(content)
                
                return {
                    "status": "success",
                    "operation": operation,
                    "path": str(file_path),
                    "size": len(content)
                }
                
        except Exception as e:
            return {"status": "error", "error": f"File operation failed: {str(e)}"}

# Pattern 4: Async tool (for future enhancement)
class AsyncTool(BaseTool):
    """Example of how to structure an async tool.
    
    Note: Current system doesn't support async tools,
    but this shows the pattern for future enhancement.
    """
    
    @property
    def name(self) -> str:
        return "async_tool"
    
    @property
    def description(self) -> str:
        return "Demonstrates async tool pattern"
    
    @property
    def parameters(self) -> dict:
        return {"type": "object", "properties": {}}
    
    def execute(self, **kwargs) -> Dict[str, Any]:
        """Sync wrapper for async operation."""
        import asyncio
        
        try:
            # Run async operation in sync context
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(self._async_execute(**kwargs))
            loop.close()
            return result
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    async def _async_execute(self, **kwargs) -> Dict[str, Any]:
        """Async implementation."""
        import asyncio
        await asyncio.sleep(1)  # Simulate async operation
        return {"status": "success", "message": "Async operation completed"}

# Anti-patterns to avoid in tools
def tool_anti_patterns():
    """Examples of what NOT to do in tools."""
    
    # ❌ Don't forget error handling
    # def execute(self, param):
    #     return external_api.call(param)  # What if it fails?
    
    # ❌ Don't return inconsistent formats
    # def execute(self, param):
    #     if success:
    #         return "Success!"  # Should be dict
    #     else:
    #         return {"error": "Failed"}  # Inconsistent
    
    # ❌ Don't perform side effects in properties
    # @property
    # def name(self):
    #     self.counter += 1  # Side effect!
    #     return f"tool_{self.counter}"
    
    # ❌ Don't ignore the config
    # def __init__(self, config):
    #     pass  # Should store and use config
    
    pass

# Helper function for tool testing
def test_tool_implementation(tool_class: type) -> bool:
    """Validate that a tool implements the required interface."""
    required_attrs = ['name', 'description', 'parameters', 'execute']
    
    tool = tool_class({})
    
    for attr in required_attrs:
        if not hasattr(tool, attr):
            print(f"❌ Missing required attribute: {attr}")
            return False
    
    # Check property types
    if not isinstance(tool.name, str):
        print("❌ name must return a string")
        return False
        
    if not isinstance(tool.description, str):
        print("❌ description must return a string")
        return False
        
    if not isinstance(tool.parameters, dict):
        print("❌ parameters must return a dict")
        return False
    
    print("✅ Tool implementation is valid")
    return True

if __name__ == "__main__":
    # Test the example tool
    print("Testing ExampleTool implementation...")
    test_tool_implementation(ExampleTool)
    
    # Demonstrate usage
    tool = ExampleTool({})
    result = tool.execute("Hello World", {"uppercase": True})
    print(f"\nExample execution result: {result}")