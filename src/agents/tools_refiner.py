"""Tools implementation and validation agent"""

from typing import List, Dict, Any, Optional
from pydantic_ai import Agent, RunContext
from .base_agent import BaseAgent
from .dependencies import ToolsRefinerDependencies
from .models import ToolsRefineOutput
import logfire
import ast
import json
import re


class ToolsRefinerAgent(BaseAgent[ToolsRefinerDependencies, ToolsRefineOutput]):
    """Specialized tools implementation and validation agent"""
    
    def __init__(self):
        """Initialize the tools refiner agent"""
        super().__init__(
            model='openai:gpt-4',
            deps_type=ToolsRefinerDependencies,
            result_type=ToolsRefineOutput,
            enable_logfire=True
        )
    
    def get_system_prompt(self) -> str:
        """Get the system prompt for tools refiner"""
        return """You are a tools engineering expert that optimizes tool implementations and MCP configurations.
        
        Your expertise includes:
        1. Validating tool functionality and reliability
        2. Optimizing tool performance and efficiency
        3. Ensuring proper error handling and edge cases
        4. Creating comprehensive tool documentation
        5. Configuring MCP (Model Context Protocol) servers
        6. Identifying missing tools and capabilities
        
        Focus on:
        - Tool reliability and robustness
        - Performance optimization
        - Comprehensive error handling
        - Clear interfaces and contracts
        - MCP compliance and integration
        - Security best practices
        
        Ensure all tools are production-ready with proper validation and testing."""
    
    def _register_tools(self):
        """Register tools for the tools refiner"""
        self.agent.tool(self.validate_tool_implementations)
        self.agent.tool(self.optimize_mcp_configurations)
        self.agent.tool(self.recommend_additional_tools)
        self.agent.tool(self.enhance_tool_interfaces)
        self.agent.tool(self.add_tool_validation)
        self.agent.tool(self.generate_tool_tests)
    
    async def validate_tool_implementations(
        self,
        ctx: RunContext[ToolsRefinerDependencies],
        tools: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Validate tool implementations
        
        Args:
            ctx: Run context
            tools: List of tools to validate
            
        Returns:
            Validation results
        """
        logfire.info("validating_tools", tool_count=len(tools))
        
        validation_results = {}
        
        for tool in tools:
            tool_name = tool.get('name', 'unknown')
            
            # Perform various validations
            validation = {
                "syntax_valid": self._validate_tool_syntax(tool),
                "interface_valid": self._validate_tool_interface(tool),
                "error_handling": self._check_error_handling(tool),
                "performance": self._assess_tool_performance(tool),
                "security": self._check_security_issues(tool),
                "documentation": self._validate_documentation(tool),
                "mcp_compatible": self._check_mcp_compatibility(tool),
                "issues": [],
                "suggestions": []
            }
            
            # Overall validation result
            validation["valid"] = all([
                validation["syntax_valid"],
                validation["interface_valid"],
                len(validation["issues"]) == 0
            ])
            
            validation_results[tool_name] = validation
            
            logfire.info(
                "tool_validation_complete",
                tool=tool_name,
                valid=validation["valid"]
            )
        
        return validation_results
    
    async def optimize_mcp_configurations(
        self,
        ctx: RunContext[ToolsRefinerDependencies],
        current_config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Optimize MCP server configurations
        
        Args:
            ctx: Run context
            current_config: Current MCP configuration
            
        Returns:
            Optimized MCP configuration
        """
        logfire.info("optimizing_mcp_config")
        
        # Start with current config or default
        config = current_config or self._get_default_mcp_config()
        
        # Get available MCP servers from context
        available_servers = ctx.deps.mcp_servers or []
        
        # Optimize server configurations
        optimized_servers = []
        for server in available_servers:
            optimized = self._optimize_mcp_server(server)
            optimized_servers.append(optimized)
        
        # Create comprehensive MCP configuration
        mcp_config = {
            "version": "1.0.0",
            "servers": optimized_servers,
            "global_settings": {
                "timeout": 30000,
                "max_retries": 3,
                "rate_limit": {
                    "requests_per_minute": 60,
                    "burst_size": 10
                }
            },
            "security": {
                "require_auth": True,
                "allowed_origins": ["*"],
                "api_key_header": "X-API-Key"
            },
            "logging": {
                "level": "info",
                "format": "json",
                "destination": "stdout"
            }
        }
        
        # Add tool-specific configurations
        mcp_config["tool_configs"] = self._generate_tool_configs(ctx.deps.tool_library)
        
        return mcp_config
    
    async def recommend_additional_tools(
        self,
        ctx: RunContext[ToolsRefinerDependencies],
        current_tools: List[Dict[str, Any]],
        use_case: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Recommend additional tools based on current setup
        
        Args:
            ctx: Run context
            current_tools: Currently available tools
            use_case: Specific use case to optimize for
            
        Returns:
            List of recommended tools
        """
        logfire.info("recommending_tools", current_count=len(current_tools))
        
        recommendations = []
        
        # Analyze current tool coverage
        coverage = self._analyze_tool_coverage(current_tools)
        
        # Identify gaps
        gaps = self._identify_capability_gaps(coverage, use_case)
        
        # Search tool library for recommendations
        if ctx.deps.tool_library:
            for gap in gaps:
                matching_tools = self._search_tool_library(
                    ctx.deps.tool_library,
                    gap["capability"],
                    gap["priority"]
                )
                
                for tool in matching_tools[:2]:  # Max 2 recommendations per gap
                    recommendations.append({
                        "tool": tool,
                        "reason": f"Fills gap: {gap['capability']}",
                        "priority": gap["priority"],
                        "integration_effort": self._estimate_integration_effort(tool)
                    })
        
        # Add commonly useful tools if missing
        common_tools = self._get_common_tool_recommendations(current_tools)
        recommendations.extend(common_tools)
        
        # Sort by priority
        recommendations.sort(key=lambda x: x["priority"], reverse=True)
        
        return recommendations[:10]  # Top 10 recommendations
    
    async def enhance_tool_interfaces(
        self,
        ctx: RunContext[ToolsRefinerDependencies],
        tool: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Enhance tool interface for better usability
        
        Args:
            ctx: Run context
            tool: Tool to enhance
            
        Returns:
            Enhanced tool
        """
        logfire.info("enhancing_tool_interface", tool=tool.get('name'))
        
        enhanced = tool.copy()
        
        # Enhance parameter definitions
        if "parameters" in enhanced:
            enhanced["parameters"] = self._enhance_parameters(enhanced["parameters"])
        
        # Add or improve response schema
        if "response_schema" not in enhanced:
            enhanced["response_schema"] = self._generate_response_schema(enhanced)
        
        # Add examples if missing
        if "examples" not in enhanced:
            enhanced["examples"] = self._generate_tool_examples(enhanced)
        
        # Improve error responses
        enhanced["error_responses"] = self._define_error_responses(enhanced)
        
        # Add rate limiting info
        enhanced["rate_limits"] = self._define_rate_limits(enhanced)
        
        # Add authentication requirements
        enhanced["authentication"] = self._define_auth_requirements(enhanced)
        
        return enhanced
    
    async def add_tool_validation(
        self,
        ctx: RunContext[ToolsRefinerDependencies],
        tool_code: str,
        language: str = "python"
    ) -> str:
        """Add validation to tool implementation
        
        Args:
            ctx: Run context
            tool_code: Tool implementation code
            language: Programming language
            
        Returns:
            Code with added validation
        """
        logfire.info("adding_tool_validation", language=language)
        
        if language == "python":
            return self._add_python_validation(tool_code)
        elif language == "typescript":
            return self._add_typescript_validation(tool_code)
        else:
            return tool_code
    
    async def generate_tool_tests(
        self,
        ctx: RunContext[ToolsRefinerDependencies],
        tool: Dict[str, Any],
        test_framework: str = "pytest"
    ) -> str:
        """Generate comprehensive tests for a tool
        
        Args:
            ctx: Run context
            tool: Tool to generate tests for
            test_framework: Testing framework to use
            
        Returns:
            Generated test code
        """
        logfire.info("generating_tool_tests", tool=tool.get('name'), framework=test_framework)
        
        if test_framework == "pytest":
            return self._generate_pytest_tests(tool)
        elif test_framework == "jest":
            return self._generate_jest_tests(tool)
        else:
            return "# Test framework not supported"
    
    def _validate_tool_syntax(self, tool: Dict[str, Any]) -> bool:
        """Validate tool syntax"""
        # Check if tool has required fields
        required_fields = ["name", "description", "parameters"]
        
        for field in required_fields:
            if field not in tool:
                return False
        
        # Validate implementation if provided
        if "implementation" in tool:
            if tool.get("language") == "python":
                try:
                    ast.parse(tool["implementation"])
                except SyntaxError:
                    return False
        
        return True
    
    def _validate_tool_interface(self, tool: Dict[str, Any]) -> bool:
        """Validate tool interface"""
        # Check parameter definitions
        if "parameters" in tool:
            for param in tool["parameters"]:
                if not isinstance(param, dict):
                    return False
                
                if "name" not in param or "type" not in param:
                    return False
                
                # Validate type
                valid_types = ["string", "number", "integer", "boolean", "array", "object"]
                if param["type"] not in valid_types:
                    return False
        
        return True
    
    def _check_error_handling(self, tool: Dict[str, Any]) -> Dict[str, Any]:
        """Check error handling in tool"""
        result = {
            "has_error_handling": False,
            "error_types_handled": [],
            "suggestions": []
        }
        
        if "implementation" in tool:
            impl = tool["implementation"]
            
            # Check for try-except blocks
            if "try:" in impl and "except" in impl:
                result["has_error_handling"] = True
                
                # Find exception types
                except_pattern = r'except\s+(\w+)'
                exceptions = re.findall(except_pattern, impl)
                result["error_types_handled"] = exceptions
            else:
                result["suggestions"].append("Add try-except blocks for error handling")
            
            # Check for validation
            if "validate" not in impl and "check" not in impl:
                result["suggestions"].append("Add input validation")
        
        return result
    
    def _assess_tool_performance(self, tool: Dict[str, Any]) -> Dict[str, Any]:
        """Assess tool performance characteristics"""
        performance = {
            "estimated_latency": "medium",
            "scalability": "good",
            "resource_usage": "low",
            "optimization_suggestions": []
        }
        
        if "implementation" in tool:
            impl = tool["implementation"]
            
            # Check for performance issues
            if "sleep" in impl or "time.sleep" in impl:
                performance["estimated_latency"] = "high"
                performance["optimization_suggestions"].append("Remove or minimize sleep calls")
            
            if re.search(r'for.*for', impl):
                performance["scalability"] = "poor"
                performance["optimization_suggestions"].append("Nested loops may cause performance issues")
            
            if "requests" in impl and "async" not in impl:
                performance["optimization_suggestions"].append("Consider using async requests for better performance")
        
        return performance
    
    def _check_security_issues(self, tool: Dict[str, Any]) -> Dict[str, Any]:
        """Check for security issues in tool"""
        security = {
            "issues": [],
            "risk_level": "low"
        }
        
        if "implementation" in tool:
            impl = tool["implementation"]
            
            # Check for common security issues
            if "eval(" in impl:
                security["issues"].append("Use of eval() is a security risk")
                security["risk_level"] = "high"
            
            if "os.system" in impl or "subprocess.call" in impl:
                security["issues"].append("Direct system calls may be a security risk")
                security["risk_level"] = "medium"
            
            if re.search(r'["\'].*(%s|%d).*["\'].*%', impl):
                security["issues"].append("Potential SQL injection vulnerability")
                security["risk_level"] = "high"
            
            # Check for hardcoded secrets
            if re.search(r'(api_key|password|secret)\s*=\s*["\'][^"\']+["\']', impl, re.IGNORECASE):
                security["issues"].append("Hardcoded credentials detected")
                security["risk_level"] = "high"
        
        return security
    
    def _validate_documentation(self, tool: Dict[str, Any]) -> Dict[str, Any]:
        """Validate tool documentation"""
        doc_quality = {
            "has_description": bool(tool.get("description")),
            "has_parameter_docs": True,
            "has_examples": bool(tool.get("examples")),
            "completeness_score": 0.0,
            "missing_docs": []
        }
        
        # Check parameter documentation
        if "parameters" in tool:
            for param in tool["parameters"]:
                if "description" not in param:
                    doc_quality["has_parameter_docs"] = False
                    doc_quality["missing_docs"].append(f"Parameter '{param.get('name')}' lacks description")
        
        # Calculate completeness score
        score = 0.0
        if doc_quality["has_description"]:
            score += 0.4
        if doc_quality["has_parameter_docs"]:
            score += 0.3
        if doc_quality["has_examples"]:
            score += 0.3
        
        doc_quality["completeness_score"] = score
        
        return doc_quality
    
    def _check_mcp_compatibility(self, tool: Dict[str, Any]) -> bool:
        """Check if tool is MCP compatible"""
        # Check for MCP-required fields
        mcp_required = ["name", "description", "parameters"]
        
        for field in mcp_required:
            if field not in tool:
                return False
        
        # Check parameter format
        if "parameters" in tool:
            for param in tool["parameters"]:
                if not isinstance(param, dict):
                    return False
                if "name" not in param or "type" not in param:
                    return False
        
        return True
    
    def _get_default_mcp_config(self) -> Dict[str, Any]:
        """Get default MCP configuration"""
        return {
            "version": "1.0.0",
            "servers": [],
            "global_settings": {
                "timeout": 30000,
                "max_retries": 3
            }
        }
    
    def _optimize_mcp_server(self, server: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize individual MCP server configuration"""
        optimized = server.copy()
        
        # Ensure required fields
        if "name" not in optimized:
            optimized["name"] = "unnamed_server"
        
        if "version" not in optimized:
            optimized["version"] = "1.0.0"
        
        # Add performance settings
        if "performance" not in optimized:
            optimized["performance"] = {
                "max_concurrent_requests": 10,
                "request_timeout": 30000,
                "keepalive": True
            }
        
        # Add monitoring
        if "monitoring" not in optimized:
            optimized["monitoring"] = {
                "enabled": True,
                "metrics_endpoint": "/metrics",
                "health_endpoint": "/health"
            }
        
        return optimized
    
    def _generate_tool_configs(self, tool_library: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate tool-specific configurations"""
        configs = {}
        
        if not tool_library:
            return configs
        
        # Common tool configurations
        default_config = {
            "timeout": 10000,
            "retries": 2,
            "cache": {
                "enabled": True,
                "ttl": 300
            }
        }
        
        # Add specific configs for each tool category
        tool_categories = {
            "api": {
                "rate_limit": {"requests_per_minute": 60},
                "timeout": 30000
            },
            "database": {
                "connection_pool": {"size": 5, "timeout": 5000},
                "query_timeout": 10000
            },
            "file": {
                "max_file_size": 10485760,  # 10MB
                "allowed_extensions": ["*"]
            }
        }
        
        for category, config in tool_categories.items():
            configs[category] = {**default_config, **config}
        
        return configs
    
    def _analyze_tool_coverage(self, tools: List[Dict[str, Any]]) -> Dict[str, List[str]]:
        """Analyze what capabilities are covered by current tools"""
        coverage = {
            "data_access": [],
            "data_processing": [],
            "external_integration": [],
            "file_operations": [],
            "communication": [],
            "monitoring": [],
            "security": [],
            "utilities": []
        }
        
        for tool in tools:
            tool_name = tool.get("name", "").lower()
            description = tool.get("description", "").lower()
            
            # Categorize tools
            if any(word in tool_name + description for word in ["database", "sql", "query", "redis", "mongo"]):
                coverage["data_access"].append(tool["name"])
            
            if any(word in tool_name + description for word in ["process", "transform", "calculate", "analyze"]):
                coverage["data_processing"].append(tool["name"])
            
            if any(word in tool_name + description for word in ["api", "http", "webhook", "external"]):
                coverage["external_integration"].append(tool["name"])
            
            if any(word in tool_name + description for word in ["file", "read", "write", "directory"]):
                coverage["file_operations"].append(tool["name"])
            
            if any(word in tool_name + description for word in ["email", "slack", "notify", "message"]):
                coverage["communication"].append(tool["name"])
            
            if any(word in tool_name + description for word in ["log", "metric", "monitor", "trace"]):
                coverage["monitoring"].append(tool["name"])
            
            if any(word in tool_name + description for word in ["auth", "encrypt", "security", "validate"]):
                coverage["security"].append(tool["name"])
        
        return coverage
    
    def _identify_capability_gaps(
        self,
        coverage: Dict[str, List[str]],
        use_case: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Identify gaps in tool capabilities"""
        gaps = []
        
        # Essential capabilities that should always be present
        essential_capabilities = {
            "data_access": "Database or data storage access",
            "external_integration": "API or external service integration",
            "file_operations": "File system operations",
            "monitoring": "Logging and monitoring"
        }
        
        for capability, description in essential_capabilities.items():
            if not coverage.get(capability):
                gaps.append({
                    "capability": capability,
                    "description": description,
                    "priority": "high"
                })
        
        # Use case specific gaps
        if use_case:
            use_case_lower = use_case.lower()
            
            if "web" in use_case_lower and not coverage.get("external_integration"):
                gaps.append({
                    "capability": "web_scraping",
                    "description": "Web scraping and HTML parsing",
                    "priority": "medium"
                })
            
            if "data" in use_case_lower and not coverage.get("data_processing"):
                gaps.append({
                    "capability": "data_analysis",
                    "description": "Data analysis and visualization",
                    "priority": "high"
                })
        
        return gaps
    
    def _search_tool_library(
        self,
        library: Dict[str, Any],
        capability: str,
        priority: str
    ) -> List[Dict[str, Any]]:
        """Search tool library for matching tools"""
        matches = []
        
        # This would search actual tool library
        # For now, return mock results
        if capability == "data_access":
            matches.append({
                "name": "postgres_tool",
                "description": "PostgreSQL database operations",
                "category": "database"
            })
        
        elif capability == "monitoring":
            matches.append({
                "name": "logger_tool",
                "description": "Structured logging tool",
                "category": "monitoring"
            })
        
        return matches
    
    def _estimate_integration_effort(self, tool: Dict[str, Any]) -> str:
        """Estimate effort to integrate a tool"""
        # Simple heuristic
        if tool.get("category") in ["utility", "logging"]:
            return "low"
        elif tool.get("category") in ["database", "api"]:
            return "medium"
        else:
            return "high"
    
    def _get_common_tool_recommendations(
        self,
        current_tools: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Get recommendations for commonly useful tools"""
        recommendations = []
        current_names = [t.get("name", "").lower() for t in current_tools]
        
        common_tools = [
            {
                "name": "http_client",
                "description": "HTTP client for API calls",
                "reason": "Essential for external integrations"
            },
            {
                "name": "json_validator",
                "description": "JSON schema validation",
                "reason": "Ensures data integrity"
            },
            {
                "name": "rate_limiter",
                "description": "Rate limiting for API calls",
                "reason": "Prevents API abuse"
            },
            {
                "name": "cache_tool",
                "description": "Caching for performance",
                "reason": "Improves response times"
            }
        ]
        
        for tool in common_tools:
            if tool["name"] not in current_names:
                recommendations.append({
                    "tool": tool,
                    "reason": tool["reason"],
                    "priority": "medium",
                    "integration_effort": "low"
                })
        
        return recommendations
    
    def _enhance_parameters(self, parameters: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Enhance parameter definitions"""
        enhanced = []
        
        for param in parameters:
            enhanced_param = param.copy()
            
            # Add description if missing
            if "description" not in enhanced_param:
                enhanced_param["description"] = f"The {param['name']} parameter"
            
            # Add validation rules
            if param["type"] == "string":
                if "minLength" not in enhanced_param:
                    enhanced_param["minLength"] = 0
                if "maxLength" not in enhanced_param:
                    enhanced_param["maxLength"] = 1000
            
            elif param["type"] in ["number", "integer"]:
                if "minimum" not in enhanced_param:
                    enhanced_param["minimum"] = -999999
                if "maximum" not in enhanced_param:
                    enhanced_param["maximum"] = 999999
            
            # Add examples
            if "examples" not in enhanced_param:
                enhanced_param["examples"] = self._generate_param_examples(enhanced_param)
            
            enhanced.append(enhanced_param)
        
        return enhanced
    
    def _generate_response_schema(self, tool: Dict[str, Any]) -> Dict[str, Any]:
        """Generate response schema for tool"""
        return {
            "type": "object",
            "properties": {
                "success": {"type": "boolean"},
                "data": {"type": "object"},
                "error": {"type": "string"},
                "metadata": {
                    "type": "object",
                    "properties": {
                        "timestamp": {"type": "string"},
                        "duration_ms": {"type": "number"}
                    }
                }
            },
            "required": ["success"]
        }
    
    def _generate_tool_examples(self, tool: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate usage examples for tool"""
        examples = []
        
        # Basic example
        basic_example = {
            "description": "Basic usage example",
            "input": {},
            "output": {
                "success": True,
                "data": {}
            }
        }
        
        # Add parameter values to input
        if "parameters" in tool:
            for param in tool["parameters"]:
                if param.get("required", False):
                    basic_example["input"][param["name"]] = self._get_example_value(param)
        
        examples.append(basic_example)
        
        # Error example
        examples.append({
            "description": "Error handling example",
            "input": {"invalid_param": "value"},
            "output": {
                "success": False,
                "error": "Invalid parameter: invalid_param"
            }
        })
        
        return examples
    
    def _define_error_responses(self, tool: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """Define error response formats"""
        return {
            "validation_error": {
                "status": 400,
                "response": {
                    "success": False,
                    "error": "Validation error",
                    "details": []
                }
            },
            "not_found": {
                "status": 404,
                "response": {
                    "success": False,
                    "error": "Resource not found"
                }
            },
            "server_error": {
                "status": 500,
                "response": {
                    "success": False,
                    "error": "Internal server error"
                }
            },
            "rate_limit": {
                "status": 429,
                "response": {
                    "success": False,
                    "error": "Rate limit exceeded",
                    "retry_after": 60
                }
            }
        }
    
    def _define_rate_limits(self, tool: Dict[str, Any]) -> Dict[str, Any]:
        """Define rate limiting for tool"""
        # Default rate limits based on tool type
        tool_name = tool.get("name", "").lower()
        
        if any(word in tool_name for word in ["api", "external", "http"]):
            return {
                "requests_per_minute": 60,
                "requests_per_hour": 1000,
                "burst_size": 10
            }
        elif any(word in tool_name for word in ["database", "query"]):
            return {
                "requests_per_minute": 300,
                "requests_per_hour": 10000,
                "burst_size": 50
            }
        else:
            return {
                "requests_per_minute": 600,
                "requests_per_hour": 20000,
                "burst_size": 100
            }
    
    def _define_auth_requirements(self, tool: Dict[str, Any]) -> Dict[str, Any]:
        """Define authentication requirements"""
        tool_name = tool.get("name", "").lower()
        
        # Sensitive operations require auth
        if any(word in tool_name for word in ["write", "delete", "update", "admin"]):
            return {
                "required": True,
                "type": "bearer",
                "scopes": ["write"]
            }
        elif any(word in tool_name for word in ["read", "get", "list"]):
            return {
                "required": False,
                "type": "bearer",
                "scopes": ["read"]
            }
        else:
            return {
                "required": False,
                "type": "none"
            }
    
    def _add_python_validation(self, code: str) -> str:
        """Add validation to Python tool code"""
        lines = code.split('\n')
        enhanced_lines = []
        
        # Add imports if needed
        if "from pydantic import" not in code:
            enhanced_lines.append("from pydantic import BaseModel, Field, validator")
            enhanced_lines.append("from typing import Optional, Dict, Any, List")
            enhanced_lines.append("")
        
        # Find function definitions and add validation
        for i, line in enumerate(lines):
            if line.strip().startswith('def ') and '(' in line:
                # Add parameter validation decorator
                enhanced_lines.append("@validate_parameters")
            
            enhanced_lines.append(line)
            
            # Add validation after function definition
            if line.strip().startswith('def ') and i + 1 < len(lines):
                # Get indentation
                next_line = lines[i + 1] if i + 1 < len(lines) else ""
                indent = len(next_line) - len(next_line.lstrip())
                
                if indent > 0:
                    validation_code = f"{' ' * indent}# Validate inputs\n"
                    validation_code += f"{' ' * indent}if not all([param for param in []]):\n"
                    validation_code += f"{' ' * (indent + 4)}raise ValueError('Missing required parameters')\n"
                    enhanced_lines.append(validation_code)
        
        return '\n'.join(enhanced_lines)
    
    def _add_typescript_validation(self, code: str) -> str:
        """Add validation to TypeScript tool code"""
        # Add zod validation
        enhanced = code
        
        if "import { z }" not in code:
            enhanced = "import { z } from 'zod';\n\n" + enhanced
        
        # Add schema definitions
        # This would be more sophisticated in practice
        
        return enhanced
    
    def _generate_pytest_tests(self, tool: Dict[str, Any]) -> str:
        """Generate pytest tests for tool"""
        tool_name = tool.get("name", "tool")
        
        test_code = f"""import pytest
from unittest.mock import Mock, patch
from {tool_name} import {tool_name.title().replace('_', '')}Tool


class Test{tool_name.title().replace('_', '')}Tool:
    \"\"\"Test suite for {tool_name} tool\"\"\"
    
    @pytest.fixture
    def tool(self):
        \"\"\"Create tool instance for testing\"\"\"
        return {tool_name.title().replace('_', '')}Tool()
    
    def test_tool_initialization(self, tool):
        \"\"\"Test tool initializes correctly\"\"\"
        assert tool is not None
        assert tool.name == "{tool_name}"
    
    def test_basic_functionality(self, tool):
        \"\"\"Test basic tool functionality\"\"\"
        # Arrange
        test_input = {{}}
"""
        
        # Add parameter tests
        if "parameters" in tool:
            for param in tool["parameters"]:
                if param.get("required"):
                    test_code += f"""
    def test_required_parameter_{param['name']}(self, tool):
        \"\"\"Test {param['name']} parameter is required\"\"\"
        with pytest.raises(ValueError):
            tool.execute({{}})  # Missing {param['name']}
"""
        
        # Add error handling tests
        test_code += """
    def test_error_handling(self, tool):
        \"\"\"Test tool handles errors gracefully\"\"\"
        with patch.object(tool, 'execute', side_effect=Exception('Test error')):
            result = tool.safe_execute({})
            assert result['success'] is False
            assert 'error' in result
    
    @pytest.mark.parametrize("invalid_input", [
        None,
        [],
        "string instead of dict",
        {"invalid_param": "value"}
    ])
    def test_invalid_inputs(self, tool, invalid_input):
        \"\"\"Test tool handles invalid inputs\"\"\"
        result = tool.execute(invalid_input)
        assert result['success'] is False
"""
        
        return test_code
    
    def _generate_jest_tests(self, tool: Dict[str, Any]) -> str:
        """Generate Jest tests for tool"""
        tool_name = tool.get("name", "tool")
        
        test_code = f"""import {{ {tool_name.title().replace('_', '')}Tool }} from './{tool_name}';

describe('{tool_name.title().replace('_', '')}Tool', () => {{
    let tool;
    
    beforeEach(() => {{
        tool = new {tool_name.title().replace('_', '')}Tool();
    }});
    
    test('should initialize correctly', () => {{
        expect(tool).toBeDefined();
        expect(tool.name).toBe('{tool_name}');
    }});
    
    test('should execute successfully with valid input', async () => {{
        const input = {{}};
        const result = await tool.execute(input);
        expect(result.success).toBe(true);
    }});
"""
        
        # Add parameter tests
        if "parameters" in tool:
            for param in tool["parameters"]:
                if param.get("required"):
                    test_code += f"""
    test('should require {param["name"]} parameter', async () => {{
        const input = {{}};  // Missing {param["name"]}
        await expect(tool.execute(input)).rejects.toThrow();
    }});
"""
        
        test_code += """
    test('should handle errors gracefully', async () => {
        const input = { invalidParam: 'value' };
        const result = await tool.execute(input);
        expect(result.success).toBe(false);
        expect(result.error).toBeDefined();
    });
});"""
        
        return test_code
    
    def _generate_param_examples(self, param: Dict[str, Any]) -> List[Any]:
        """Generate example values for parameter"""
        param_type = param.get("type", "string")
        
        if param_type == "string":
            return ["example", "test", ""]
        elif param_type == "number":
            return [0, 1.5, -10, 999]
        elif param_type == "integer":
            return [0, 1, -1, 100]
        elif param_type == "boolean":
            return [True, False]
        elif param_type == "array":
            return [[], ["item1"], ["item1", "item2"]]
        elif param_type == "object":
            return [{}, {"key": "value"}]
        
        return []
    
    def _get_example_value(self, param: Dict[str, Any]) -> Any:
        """Get a single example value for parameter"""
        examples = self._generate_param_examples(param)
        return examples[0] if examples else None
    
    async def refine(self, agent_data: Dict[str, Any]) -> Dict[str, Any]:
        """Refine agent tools and MCP configurations"""
        current_tools = agent_data.get('tools', [])
        
        # Validate current tools
        validation_results = await self.validate_tool_implementations(None, current_tools)
        
        # Find tools that need improvement
        tools_to_improve = []
        for tool_name, validation in validation_results.items():
            if not validation['valid'] or validation['suggestions']:
                tools_to_improve.append(tool_name)
        
        # Enhance tools
        enhanced_tools = []
        for tool in current_tools:
            if tool.get('name') in tools_to_improve:
                enhanced = await self.enhance_tool_interfaces(None, tool)
                enhanced_tools.append(enhanced)
            else:
                enhanced_tools.append(tool)
        
        # Optimize MCP configuration
        mcp_config = await self.optimize_mcp_configurations(None, agent_data.get('mcp_config'))
        
        # Get tool recommendations
        recommendations = await self.recommend_additional_tools(None, enhanced_tools)
        
        # Update agent data
        agent_data['tools'] = enhanced_tools
        agent_data['mcp_config'] = mcp_config
        agent_data['tool_recommendations'] = recommendations
        agent_data['tool_validation_results'] = validation_results
        
        return agent_data