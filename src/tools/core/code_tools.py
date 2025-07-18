"""Code operation tools"""

import ast
import subprocess
import asyncio
from typing import List, Dict, Any, Optional
from pathlib import Path
import tempfile
import os
from ..base_tool import BaseTool, ToolParameter, ToolResult


class CodeExecutorTool(BaseTool):
    """Tool for executing code snippets"""
    
    @property
    def name(self) -> str:
        return "execute_code"
    
    @property
    def description(self) -> str:
        return "Execute code snippets in various languages"
    
    @property
    def category(self) -> str:
        return "code_operations"
    
    @property
    def requires_confirmation(self) -> bool:
        return True
    
    @property
    def parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="code",
                type="string",
                description="Code to execute",
                required=True
            ),
            ToolParameter(
                name="language",
                type="string",
                description="Programming language",
                required=True,
                enum=["python", "javascript", "bash", "shell"]
            ),
            ToolParameter(
                name="timeout",
                type="integer",
                description="Execution timeout in seconds",
                required=False,
                default=30,
                min_value=1,
                max_value=300
            ),
            ToolParameter(
                name="working_directory",
                type="string",
                description="Working directory for execution",
                required=False
            )
        ]
    
    async def _execute(self, params: Dict[str, Any]) -> ToolResult:
        """Execute code snippet"""
        code = params["code"]
        language = params["language"]
        timeout = params.get("timeout", 30)
        working_dir = params.get("working_directory")
        
        if working_dir and not Path(working_dir).exists():
            return ToolResult(
                success=False,
                error=f"Working directory not found: {working_dir}"
            )
        
        try:
            if language == "python":
                result = await self._execute_python(code, timeout, working_dir)
            elif language == "javascript":
                result = await self._execute_javascript(code, timeout, working_dir)
            elif language in ["bash", "shell"]:
                result = await self._execute_shell(code, timeout, working_dir)
            else:
                return ToolResult(
                    success=False,
                    error=f"Unsupported language: {language}"
                )
            
            return result
            
        except asyncio.TimeoutError:
            return ToolResult(
                success=False,
                error=f"Code execution timed out after {timeout} seconds"
            )
        except Exception as e:
            return ToolResult(
                success=False,
                error=f"Code execution failed: {str(e)}"
            )
    
    async def _execute_python(self, code: str, timeout: int, working_dir: Optional[str]) -> ToolResult:
        """Execute Python code"""
        # Create temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code)
            temp_file = f.name
        
        try:
            proc = await asyncio.create_subprocess_exec(
                'python', temp_file,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=working_dir
            )
            
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(),
                timeout=timeout
            )
            
            return ToolResult(
                success=proc.returncode == 0,
                data={
                    "stdout": stdout.decode(),
                    "stderr": stderr.decode(),
                    "return_code": proc.returncode,
                    "language": "python"
                },
                error=stderr.decode() if proc.returncode != 0 else None
            )
            
        finally:
            # Clean up temporary file
            os.unlink(temp_file)
    
    async def _execute_javascript(self, code: str, timeout: int, working_dir: Optional[str]) -> ToolResult:
        """Execute JavaScript code"""
        # Create temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.js', delete=False) as f:
            f.write(code)
            temp_file = f.name
        
        try:
            proc = await asyncio.create_subprocess_exec(
                'node', temp_file,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=working_dir
            )
            
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(),
                timeout=timeout
            )
            
            return ToolResult(
                success=proc.returncode == 0,
                data={
                    "stdout": stdout.decode(),
                    "stderr": stderr.decode(),
                    "return_code": proc.returncode,
                    "language": "javascript"
                },
                error=stderr.decode() if proc.returncode != 0 else None
            )
            
        finally:
            # Clean up temporary file
            os.unlink(temp_file)
    
    async def _execute_shell(self, code: str, timeout: int, working_dir: Optional[str]) -> ToolResult:
        """Execute shell code"""
        proc = await asyncio.create_subprocess_shell(
            code,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=working_dir
        )
        
        stdout, stderr = await asyncio.wait_for(
            proc.communicate(),
            timeout=timeout
        )
        
        return ToolResult(
            success=proc.returncode == 0,
            data={
                "stdout": stdout.decode(),
                "stderr": stderr.decode(),
                "return_code": proc.returncode,
                "language": "shell"
            },
            error=stderr.decode() if proc.returncode != 0 else None
        )


class CodeAnalyzerTool(BaseTool):
    """Tool for analyzing code structure and quality"""
    
    @property
    def name(self) -> str:
        return "analyze_code"
    
    @property
    def description(self) -> str:
        return "Analyze code structure, complexity, and quality"
    
    @property
    def category(self) -> str:
        return "code_operations"
    
    @property
    def parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="code",
                type="string",
                description="Code to analyze",
                required=True
            ),
            ToolParameter(
                name="language",
                type="string",
                description="Programming language",
                required=True,
                enum=["python", "javascript"]
            ),
            ToolParameter(
                name="analysis_type",
                type="string",
                description="Type of analysis to perform",
                required=False,
                default="all",
                enum=["all", "structure", "complexity", "quality", "dependencies"]
            )
        ]
    
    async def _execute(self, params: Dict[str, Any]) -> ToolResult:
        """Analyze code"""
        code = params["code"]
        language = params["language"]
        analysis_type = params.get("analysis_type", "all")
        
        try:
            if language == "python":
                analysis = await self._analyze_python(code, analysis_type)
            elif language == "javascript":
                analysis = await self._analyze_javascript(code, analysis_type)
            else:
                return ToolResult(
                    success=False,
                    error=f"Unsupported language: {language}"
                )
            
            return ToolResult(
                success=True,
                data=analysis
            )
            
        except Exception as e:
            return ToolResult(
                success=False,
                error=f"Code analysis failed: {str(e)}"
            )
    
    async def _analyze_python(self, code: str, analysis_type: str) -> Dict[str, Any]:
        """Analyze Python code"""
        analysis = {
            "language": "python",
            "lines_of_code": len(code.split('\n')),
            "characters": len(code)
        }
        
        try:
            tree = ast.parse(code)
            
            if analysis_type in ["all", "structure"]:
                analysis["structure"] = self._analyze_python_structure(tree)
            
            if analysis_type in ["all", "complexity"]:
                analysis["complexity"] = self._analyze_python_complexity(tree)
            
            if analysis_type in ["all", "quality"]:
                analysis["quality"] = self._analyze_python_quality(code, tree)
            
            if analysis_type in ["all", "dependencies"]:
                analysis["dependencies"] = self._analyze_python_dependencies(tree)
                
        except SyntaxError as e:
            analysis["syntax_error"] = str(e)
        
        return analysis
    
    def _analyze_python_structure(self, tree: ast.AST) -> Dict[str, Any]:
        """Analyze Python code structure"""
        structure = {
            "classes": [],
            "functions": [],
            "imports": [],
            "global_variables": []
        }
        
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                class_info = {
                    "name": node.name,
                    "methods": [],
                    "line": node.lineno
                }
                for item in node.body:
                    if isinstance(item, ast.FunctionDef):
                        class_info["methods"].append(item.name)
                structure["classes"].append(class_info)
                
            elif isinstance(node, ast.FunctionDef) and not any(
                isinstance(parent, ast.ClassDef) for parent in ast.walk(tree)
                if hasattr(parent, 'body') and node in parent.body
            ):
                func_info = {
                    "name": node.name,
                    "args": [arg.arg for arg in node.args.args],
                    "line": node.lineno
                }
                structure["functions"].append(func_info)
                
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    structure["imports"].append(alias.name)
                    
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                for alias in node.names:
                    structure["imports"].append(f"{module}.{alias.name}")
        
        return structure
    
    def _analyze_python_complexity(self, tree: ast.AST) -> Dict[str, Any]:
        """Analyze Python code complexity"""
        complexity = {
            "cyclomatic_complexity": 1,  # Base complexity
            "nesting_depth": 0,
            "function_complexities": {}
        }
        
        # Calculate cyclomatic complexity
        for node in ast.walk(tree):
            if isinstance(node, (ast.If, ast.While, ast.For, ast.ExceptHandler)):
                complexity["cyclomatic_complexity"] += 1
            elif isinstance(node, ast.BoolOp):
                complexity["cyclomatic_complexity"] += len(node.values) - 1
        
        # Calculate function complexities
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                func_complexity = 1
                for child in ast.walk(node):
                    if isinstance(child, (ast.If, ast.While, ast.For, ast.ExceptHandler)):
                        func_complexity += 1
                complexity["function_complexities"][node.name] = func_complexity
        
        return complexity
    
    def _analyze_python_quality(self, code: str, tree: ast.AST) -> Dict[str, Any]:
        """Analyze Python code quality"""
        quality = {
            "has_docstrings": False,
            "has_type_hints": False,
            "follows_pep8": True,  # Simplified
            "issues": []
        }
        
        # Check for module docstring
        if ast.get_docstring(tree):
            quality["has_docstrings"] = True
        
        # Check for type hints
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                if node.returns or any(arg.annotation for arg in node.args.args):
                    quality["has_type_hints"] = True
                    break
        
        # Simple quality checks
        lines = code.split('\n')
        for i, line in enumerate(lines):
            if len(line) > 79:
                quality["issues"].append(f"Line {i+1}: exceeds 79 characters")
                quality["follows_pep8"] = False
        
        return quality
    
    def _analyze_python_dependencies(self, tree: ast.AST) -> List[str]:
        """Extract Python dependencies"""
        dependencies = set()
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    dependencies.add(alias.name.split('.')[0])
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    dependencies.add(node.module.split('.')[0])
        
        return sorted(list(dependencies))
    
    async def _analyze_javascript(self, code: str, analysis_type: str) -> Dict[str, Any]:
        """Analyze JavaScript code"""
        # Simplified JavaScript analysis
        analysis = {
            "language": "javascript",
            "lines_of_code": len(code.split('\n')),
            "characters": len(code)
        }
        
        if analysis_type in ["all", "structure"]:
            analysis["structure"] = {
                "functions": len([line for line in code.split('\n') if 'function' in line]),
                "classes": len([line for line in code.split('\n') if 'class' in line])
            }
        
        return analysis


class CodeFormatterTool(BaseTool):
    """Tool for formatting code"""
    
    @property
    def name(self) -> str:
        return "format_code"
    
    @property
    def description(self) -> str:
        return "Format code according to language standards"
    
    @property
    def category(self) -> str:
        return "code_operations"
    
    @property
    def parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="code",
                type="string",
                description="Code to format",
                required=True
            ),
            ToolParameter(
                name="language",
                type="string",
                description="Programming language",
                required=True,
                enum=["python", "javascript", "json"]
            ),
            ToolParameter(
                name="style",
                type="string",
                description="Formatting style",
                required=False,
                default="default"
            )
        ]
    
    async def _execute(self, params: Dict[str, Any]) -> ToolResult:
        """Format code"""
        code = params["code"]
        language = params["language"]
        style = params.get("style", "default")
        
        try:
            if language == "python":
                formatted = await self._format_python(code, style)
            elif language == "javascript":
                formatted = await self._format_javascript(code, style)
            elif language == "json":
                formatted = await self._format_json(code, style)
            else:
                return ToolResult(
                    success=False,
                    error=f"Unsupported language: {language}"
                )
            
            return ToolResult(
                success=True,
                data={
                    "formatted_code": formatted,
                    "language": language,
                    "style": style
                }
            )
            
        except Exception as e:
            return ToolResult(
                success=False,
                error=f"Code formatting failed: {str(e)}"
            )
    
    async def _format_python(self, code: str, style: str) -> str:
        """Format Python code"""
        try:
            # Try to use black if available
            import black
            return black.format_str(code, mode=black.Mode())
        except ImportError:
            # Fallback to basic formatting
            try:
                tree = ast.parse(code)
                return ast.unparse(tree)
            except:
                return code
    
    async def _format_javascript(self, code: str, style: str) -> str:
        """Format JavaScript code"""
        # Basic formatting - proper implementation would use prettier
        lines = code.split('\n')
        formatted_lines = []
        indent_level = 0
        
        for line in lines:
            stripped = line.strip()
            if stripped:
                if stripped.endswith('{'):
                    formatted_lines.append('  ' * indent_level + stripped)
                    indent_level += 1
                elif stripped.startswith('}'):
                    indent_level = max(0, indent_level - 1)
                    formatted_lines.append('  ' * indent_level + stripped)
                else:
                    formatted_lines.append('  ' * indent_level + stripped)
            else:
                formatted_lines.append('')
        
        return '\n'.join(formatted_lines)
    
    async def _format_json(self, code: str, style: str) -> str:
        """Format JSON code"""
        import json
        
        parsed = json.loads(code)
        return json.dumps(parsed, indent=2, sort_keys=True)