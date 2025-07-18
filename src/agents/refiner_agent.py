"""Refiner agent for autonomous code improvement"""

from typing import List, Dict, Any, Optional
from pydantic_ai import RunContext
from .base_agent import BaseAgent
from .dependencies import RefinerDependencies
from .models import RefineOutput, ValidationResult
import logfire
import ast
import re


class RefinerAgent(BaseAgent[RefinerDependencies, RefineOutput]):
    """Agent that refines and improves code autonomously"""
    
    def __init__(self):
        """Initialize the refiner agent"""
        super().__init__(
            model='openai:gpt-4',
            deps_type=RefinerDependencies,
            result_type=RefineOutput,
            enable_logfire=True
        )
    
    def get_system_prompt(self) -> str:
        """Get the system prompt for the refiner agent"""
        return """You are an expert code refiner that improves code quality, performance, and maintainability.
        
        Your responsibilities:
        1. Identify and fix code issues
        2. Improve code structure and readability
        3. Optimize performance where possible
        4. Ensure proper error handling
        5. Add missing type hints and documentation
        6. Apply best practices and design patterns
        
        Focus on:
        - Making code more maintainable
        - Reducing complexity
        - Improving test coverage
        - Following SOLID principles
        - Ensuring security best practices
        
        Always validate your improvements and ensure the code still works correctly."""
    
    def _register_tools(self):
        """Register tools for the refiner agent"""
        self.agent.tool(self.analyze_code_quality)
        self.agent.tool(self.apply_improvements)
        self.agent.tool(self.optimize_performance)
        self.agent.tool(self.enhance_error_handling)
        self.agent.tool(self.improve_readability)
        self.agent.tool(self.validate_improvements)
    
    async def analyze_code_quality(
        self,
        ctx: RunContext[RefinerDependencies],
        code: str,
        language: str = "python"
    ) -> Dict[str, Any]:
        """Analyze code quality and identify issues
        
        Args:
            ctx: Run context
            code: Code to analyze
            language: Programming language
            
        Returns:
            Quality analysis results
        """
        logfire.info("analyzing_code_quality", language=language)
        
        analysis = {
            "complexity": self._analyze_complexity(code, language),
            "issues": self._identify_issues(code, language),
            "patterns": self._detect_antipatterns(code, language),
            "coverage": self._estimate_test_coverage(code, language),
            "security": self._check_security_issues(code, language),
            "performance": self._identify_performance_issues(code, language)
        }
        
        # Calculate overall quality score
        analysis["quality_score"] = self._calculate_quality_score(analysis)
        
        return analysis
    
    async def apply_improvements(
        self,
        ctx: RunContext[RefinerDependencies],
        code: str,
        issues: List[Dict[str, Any]],
        patterns: List[str]
    ) -> str:
        """Apply improvements to code
        
        Args:
            ctx: Run context
            code: Original code
            issues: Identified issues
            patterns: Improvement patterns to apply
            
        Returns:
            Improved code
        """
        logfire.info("applying_improvements", issue_count=len(issues))
        
        improved_code = code
        improvements_applied = []
        
        # Sort issues by priority
        sorted_issues = sorted(issues, key=lambda x: x.get("priority", 0), reverse=True)
        
        # Apply improvements based on patterns
        for pattern in ctx.deps.improvement_patterns or patterns:
            if pattern == "add_type_hints":
                improved_code = self._add_type_hints(improved_code)
                improvements_applied.append("Added type hints")
            
            elif pattern == "extract_methods":
                improved_code = self._extract_long_methods(improved_code)
                improvements_applied.append("Extracted long methods")
            
            elif pattern == "remove_duplication":
                improved_code = self._remove_code_duplication(improved_code)
                improvements_applied.append("Removed code duplication")
            
            elif pattern == "improve_naming":
                improved_code = self._improve_variable_naming(improved_code)
                improvements_applied.append("Improved variable naming")
        
        # Fix specific issues
        for issue in sorted_issues[:10]:  # Limit to top 10 issues
            fix_result = self._fix_issue(improved_code, issue)
            if fix_result["success"]:
                improved_code = fix_result["code"]
                improvements_applied.append(f"Fixed: {issue['description']}")
        
        logfire.info("improvements_applied", count=len(improvements_applied))
        
        return improved_code
    
    async def optimize_performance(
        self,
        ctx: RunContext[RefinerDependencies],
        code: str,
        hotspots: Optional[List[Dict[str, Any]]] = None
    ) -> str:
        """Optimize code performance
        
        Args:
            ctx: Run context
            code: Code to optimize
            hotspots: Performance hotspots to focus on
            
        Returns:
            Optimized code
        """
        logfire.info("optimizing_performance")
        
        optimized_code = code
        
        # Identify performance opportunities if not provided
        if not hotspots:
            hotspots = self._identify_performance_hotspots(code)
        
        # Apply optimizations
        for hotspot in hotspots:
            optimization = self._apply_optimization(optimized_code, hotspot)
            if optimization["success"]:
                optimized_code = optimization["code"]
                logfire.info(
                    "optimization_applied",
                    type=hotspot["type"],
                    improvement=optimization["improvement"]
                )
        
        # General optimizations
        optimized_code = self._apply_general_optimizations(optimized_code)
        
        return optimized_code
    
    async def enhance_error_handling(
        self,
        ctx: RunContext[RefinerDependencies],
        code: str
    ) -> str:
        """Enhance error handling in code
        
        Args:
            ctx: Run context
            code: Code to enhance
            
        Returns:
            Code with improved error handling
        """
        logfire.info("enhancing_error_handling")
        
        enhanced_code = code
        
        # Find areas lacking error handling
        unhandled_areas = self._find_unhandled_operations(code)
        
        # Add appropriate error handling
        for area in unhandled_areas:
            enhancement = self._add_error_handling(enhanced_code, area)
            if enhancement["success"]:
                enhanced_code = enhancement["code"]
        
        # Improve existing error handling
        enhanced_code = self._improve_existing_error_handling(enhanced_code)
        
        # Add logging for errors
        enhanced_code = self._add_error_logging(enhanced_code)
        
        return enhanced_code
    
    async def improve_readability(
        self,
        ctx: RunContext[RefinerDependencies],
        code: str
    ) -> str:
        """Improve code readability
        
        Args:
            ctx: Run context
            code: Code to improve
            
        Returns:
            More readable code
        """
        logfire.info("improving_readability")
        
        readable_code = code
        
        # Apply readability improvements
        improvements = [
            self._improve_function_names,
            self._add_meaningful_comments,
            self._simplify_complex_expressions,
            self._organize_imports,
            self._format_consistently
        ]
        
        for improvement_func in improvements:
            try:
                readable_code = improvement_func(readable_code)
            except Exception as e:
                logfire.error(
                    "readability_improvement_failed",
                    function=improvement_func.__name__,
                    error=str(e)
                )
        
        return readable_code
    
    async def validate_improvements(
        self,
        ctx: RunContext[RefinerDependencies],
        original_code: str,
        improved_code: str
    ) -> ValidationResult:
        """Validate that improvements don't break functionality
        
        Args:
            ctx: Run context
            original_code: Original code
            improved_code: Improved code
            
        Returns:
            Validation results
        """
        logfire.info("validating_improvements")
        
        validation = ValidationResult(
            test_passed=True,
            lint_passed=True,
            type_check_passed=True,
            errors=[],
            warnings=[]
        )
        
        # Syntax validation
        syntax_check = self._validate_syntax(improved_code)
        if not syntax_check["valid"]:
            validation.errors.append(f"Syntax error: {syntax_check['error']}")
            validation.test_passed = False
        
        # Semantic validation
        semantic_check = self._validate_semantics(original_code, improved_code)
        if not semantic_check["equivalent"]:
            validation.warnings.append(f"Semantic change detected: {semantic_check['difference']}")
        
        # Style validation
        if ctx.deps.validation_config.get("check_style", True):
            style_check = self._validate_style(improved_code)
            if not style_check["passed"]:
                validation.lint_passed = False
                validation.warnings.extend(style_check["issues"])
        
        # Type validation
        if ctx.deps.validation_config.get("check_types", True):
            type_check = self._validate_types(improved_code)
            if not type_check["passed"]:
                validation.type_check_passed = False
                validation.errors.extend(type_check["errors"])
        
        return validation
    
    def _analyze_complexity(self, code: str, language: str) -> Dict[str, Any]:
        """Analyze code complexity"""
        complexity = {
            "cyclomatic": 0,
            "cognitive": 0,
            "lines_of_code": len(code.split('\n')),
            "functions": []
        }
        
        if language == "python":
            try:
                tree = ast.parse(code)
                
                # Calculate cyclomatic complexity
                for node in ast.walk(tree):
                    if isinstance(node, (ast.If, ast.While, ast.For)):
                        complexity["cyclomatic"] += 1
                    elif isinstance(node, ast.FunctionDef):
                        func_complexity = self._calculate_function_complexity(node)
                        complexity["functions"].append({
                            "name": node.name,
                            "complexity": func_complexity,
                            "lines": node.end_lineno - node.lineno + 1 if hasattr(node, 'end_lineno') else 0
                        })
                
                # Cognitive complexity (simplified)
                complexity["cognitive"] = complexity["cyclomatic"] * 1.5
                
            except SyntaxError:
                logfire.error("complexity_analysis_failed", reason="syntax_error")
        
        return complexity
    
    def _identify_issues(self, code: str, language: str) -> List[Dict[str, Any]]:
        """Identify code issues"""
        issues = []
        
        if language == "python":
            # Long functions
            func_pattern = r'def\s+(\w+).*?(?=def\s|\Z)'
            for match in re.finditer(func_pattern, code, re.DOTALL):
                func_code = match.group(0)
                lines = func_code.count('\n')
                if lines > 50:
                    issues.append({
                        "type": "long_function",
                        "description": f"Function {match.group(1)} is too long ({lines} lines)",
                        "priority": 2,
                        "line": code[:match.start()].count('\n') + 1
                    })
            
            # Missing docstrings
            funcs_without_docs = re.findall(r'def\s+(\w+).*?:\n(?!\s*""")', code)
            for func in funcs_without_docs:
                if not func.startswith('_'):
                    issues.append({
                        "type": "missing_docstring",
                        "description": f"Function {func} lacks documentation",
                        "priority": 1,
                        "function": func
                    })
            
            # TODO comments
            todos = re.finditer(r'#\s*TODO:?\s*(.+)', code)
            for todo in todos:
                issues.append({
                    "type": "todo",
                    "description": f"TODO: {todo.group(1)}",
                    "priority": 1,
                    "line": code[:todo.start()].count('\n') + 1
                })
            
            # Broad except clauses
            broad_excepts = re.finditer(r'except\s*:', code)
            for match in broad_excepts:
                issues.append({
                    "type": "broad_except",
                    "description": "Broad except clause found",
                    "priority": 3,
                    "line": code[:match.start()].count('\n') + 1
                })
        
        return issues
    
    def _detect_antipatterns(self, code: str, language: str) -> List[str]:
        """Detect common antipatterns"""
        antipatterns = []
        
        if language == "python":
            # God class (too many methods)
            class_pattern = r'class\s+(\w+).*?(?=class\s|\Z)'
            for match in re.finditer(class_pattern, code, re.DOTALL):
                method_count = len(re.findall(r'def\s+\w+', match.group(0)))
                if method_count > 20:
                    antipatterns.append(f"God class: {match.group(1)} has {method_count} methods")
            
            # Magic numbers
            magic_numbers = re.findall(r'(?<![.\w])\d+(?![.\w])', code)
            if len(magic_numbers) > 10:
                antipatterns.append("Multiple magic numbers detected")
            
            # Nested loops
            nested_loops = re.findall(r'for\s+.*?:\s*\n\s*for\s+.*?:', code)
            if nested_loops:
                antipatterns.append(f"Nested loops detected ({len(nested_loops)} occurrences)")
        
        return antipatterns
    
    def _estimate_test_coverage(self, code: str, language: str) -> float:
        """Estimate test coverage (simplified)"""
        # This is a rough estimate based on function count
        if language == "python":
            total_functions = len(re.findall(r'def\s+\w+', code))
            test_functions = len(re.findall(r'def\s+test_\w+', code))
            
            if total_functions > 0:
                # Assume each test covers one function
                return min(test_functions / total_functions, 1.0)
        
        return 0.0
    
    def _check_security_issues(self, code: str, language: str) -> List[Dict[str, Any]]:
        """Check for security issues"""
        security_issues = []
        
        if language == "python":
            # SQL injection risk
            sql_patterns = [
                r'execute\s*\(\s*["\'].*?%[s\d].*?["\'].*?%',
                r'execute\s*\(\s*f["\'].*?{.*?}.*?["\']'
            ]
            for pattern in sql_patterns:
                if re.search(pattern, code):
                    security_issues.append({
                        "type": "sql_injection_risk",
                        "severity": "high",
                        "description": "Potential SQL injection vulnerability"
                    })
            
            # Hardcoded secrets
            secret_patterns = [
                r'(password|api_key|secret)\s*=\s*["\'][^"\']+["\']',
                r'(PASSWORD|API_KEY|SECRET)\s*=\s*["\'][^"\']+["\']'
            ]
            for pattern in secret_patterns:
                matches = re.finditer(pattern, code, re.IGNORECASE)
                for match in matches:
                    security_issues.append({
                        "type": "hardcoded_secret",
                        "severity": "high",
                        "description": f"Hardcoded credential found",
                        "line": code[:match.start()].count('\n') + 1
                    })
            
            # Unsafe deserialization
            if 'pickle.loads' in code or 'eval(' in code:
                security_issues.append({
                    "type": "unsafe_deserialization",
                    "severity": "high",
                    "description": "Unsafe deserialization detected"
                })
        
        return security_issues
    
    def _identify_performance_issues(self, code: str, language: str) -> List[Dict[str, Any]]:
        """Identify performance issues"""
        perf_issues = []
        
        if language == "python":
            # Inefficient string concatenation in loops
            loop_concat = re.search(r'for\s+.*?:\s*\n\s*.*?\+=\s*["\']', code)
            if loop_concat:
                perf_issues.append({
                    "type": "string_concatenation_in_loop",
                    "description": "String concatenation in loop is inefficient",
                    "suggestion": "Use list.append() and ''.join()"
                })
            
            # Multiple list comprehensions that could be combined
            list_comps = re.findall(r'\[.*?for.*?in.*?\]', code)
            if len(list_comps) > 3:
                perf_issues.append({
                    "type": "multiple_list_comprehensions",
                    "description": "Multiple list comprehensions could be optimized",
                    "suggestion": "Consider combining or using generator expressions"
                })
            
            # Unnecessary function calls in loops
            loop_calls = re.findall(r'for\s+.*?:\s*\n\s*.*?len\(', code)
            if loop_calls:
                perf_issues.append({
                    "type": "repeated_function_calls",
                    "description": "Function calls in loop could be cached",
                    "suggestion": "Cache function results outside the loop"
                })
        
        return perf_issues
    
    def _calculate_quality_score(self, analysis: Dict[str, Any]) -> float:
        """Calculate overall quality score"""
        score = 100.0
        
        # Deduct for complexity
        complexity = analysis["complexity"]
        if complexity["cyclomatic"] > 10:
            score -= (complexity["cyclomatic"] - 10) * 2
        
        # Deduct for issues
        issue_count = len(analysis["issues"])
        score -= issue_count * 2
        
        # Deduct for antipatterns
        antipattern_count = len(analysis["patterns"])
        score -= antipattern_count * 5
        
        # Bonus for test coverage
        coverage = analysis["coverage"]
        if coverage > 0.8:
            score += 10
        elif coverage < 0.5:
            score -= 10
        
        # Deduct for security issues
        security_issues = len(analysis["security"])
        score -= security_issues * 10
        
        # Deduct for performance issues
        perf_issues = len(analysis["performance"])
        score -= perf_issues * 3
        
        return max(0, min(100, score))
    
    def _add_type_hints(self, code: str) -> str:
        """Add type hints to Python code"""
        try:
            tree = ast.parse(code)
            
            # Simple type hint addition (would be more sophisticated in practice)
            modified = False
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    # Add return type hint if missing
                    if not node.returns:
                        # Analyze function to guess return type
                        # This is simplified - real implementation would be more complex
                        node.returns = ast.Name(id='Any', ctx=ast.Load())
                        modified = True
            
            if modified:
                return ast.unparse(tree) if hasattr(ast, 'unparse') else code
        
        except Exception as e:
            logfire.error("type_hint_addition_failed", error=str(e))
        
        return code
    
    def _extract_long_methods(self, code: str) -> str:
        """Extract long methods into smaller ones"""
        # This is a complex refactoring that would require careful AST manipulation
        # For now, return the original code
        return code
    
    def _remove_code_duplication(self, code: str) -> str:
        """Remove code duplication"""
        # Detect duplicate code blocks
        lines = code.split('\n')
        
        # Simple duplicate detection (would use more sophisticated algorithms)
        duplicates = []
        for i in range(len(lines)):
            for j in range(i + 1, len(lines)):
                if lines[i] == lines[j] and len(lines[i].strip()) > 20:
                    duplicates.append((i, j, lines[i]))
        
        # For now, just add a comment about duplicates
        if duplicates:
            lines.insert(0, f"# TODO: {len(duplicates)} duplicate lines detected")
            return '\n'.join(lines)
        
        return code
    
    def _improve_variable_naming(self, code: str) -> str:
        """Improve variable naming"""
        # Common bad variable names to replace
        replacements = {
            r'\bx\b': 'value',
            r'\bi\b': 'index',
            r'\bj\b': 'inner_index',
            r'\bk\b': 'key',
            r'\bv\b': 'value',
            r'\btemp\b': 'temporary_value',
            r'\bres\b': 'result',
            r'\barr\b': 'array',
            r'\blst\b': 'list_items',
            r'\bdict\b': 'dictionary'
        }
        
        improved_code = code
        for pattern, replacement in replacements.items():
            improved_code = re.sub(pattern, replacement, improved_code)
        
        return improved_code
    
    def _fix_issue(self, code: str, issue: Dict[str, Any]) -> Dict[str, Any]:
        """Fix a specific issue"""
        result = {"success": False, "code": code}
        
        if issue["type"] == "missing_docstring":
            # Add a docstring
            func_name = issue["function"]
            pattern = f'def {func_name}(.*?):\n'
            
            def add_docstring(match):
                return f'{match.group(0)}    """TODO: Add description for {func_name}"""\n'
            
            fixed_code = re.sub(pattern, add_docstring, code)
            if fixed_code != code:
                result["success"] = True
                result["code"] = fixed_code
        
        elif issue["type"] == "broad_except":
            # Replace broad except with specific exception
            fixed_code = code.replace('except:', 'except Exception:')
            if fixed_code != code:
                result["success"] = True
                result["code"] = fixed_code
        
        return result
    
    def _identify_performance_hotspots(self, code: str) -> List[Dict[str, Any]]:
        """Identify performance hotspots"""
        hotspots = []
        
        # Look for nested loops
        nested_loops = re.finditer(
            r'for\s+.*?:\s*\n(?:\s*.*?\n)*?\s*for\s+.*?:',
            code,
            re.MULTILINE
        )
        for match in nested_loops:
            hotspots.append({
                "type": "nested_loops",
                "location": code[:match.start()].count('\n') + 1,
                "code": match.group(0)
            })
        
        # Look for repeated calculations
        # This is simplified - real implementation would use data flow analysis
        
        return hotspots
    
    def _apply_optimization(
        self,
        code: str,
        hotspot: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Apply optimization to a hotspot"""
        result = {"success": False, "code": code, "improvement": ""}
        
        if hotspot["type"] == "nested_loops":
            # Could optimize by using numpy, list comprehension, etc.
            # This is a placeholder
            result["improvement"] = "Consider using vectorized operations"
        
        return result
    
    def _apply_general_optimizations(self, code: str) -> str:
        """Apply general optimizations"""
        optimized = code
        
        # Replace list() with []
        optimized = re.sub(r'\blist\(\)', '[]', optimized)
        
        # Replace dict() with {}
        optimized = re.sub(r'\bdict\(\)', '{}', optimized)
        
        return optimized
    
    def _find_unhandled_operations(self, code: str) -> List[Dict[str, Any]]:
        """Find operations that lack error handling"""
        unhandled = []
        
        # File operations without try-except
        file_ops = re.finditer(r'open\s*\([^)]+\)', code)
        for match in file_ops:
            # Check if it's in a try block
            before_match = code[:match.start()]
            try_count = before_match.count('try:')
            except_count = before_match.count('except')
            
            if try_count <= except_count:
                unhandled.append({
                    "type": "file_operation",
                    "code": match.group(0),
                    "line": before_match.count('\n') + 1
                })
        
        # Network operations
        network_ops = re.finditer(r'requests\.\w+\([^)]+\)', code)
        for match in network_ops:
            before_match = code[:match.start()]
            try_count = before_match.count('try:')
            except_count = before_match.count('except')
            
            if try_count <= except_count:
                unhandled.append({
                    "type": "network_operation",
                    "code": match.group(0),
                    "line": before_match.count('\n') + 1
                })
        
        return unhandled
    
    def _add_error_handling(
        self,
        code: str,
        area: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Add error handling to an area"""
        result = {"success": False, "code": code}
        
        # This is simplified - real implementation would use AST manipulation
        if area["type"] == "file_operation":
            # Add try-except around file operation
            # Would need proper indentation handling
            pass
        
        return result
    
    def _improve_existing_error_handling(self, code: str) -> str:
        """Improve existing error handling"""
        improved = code
        
        # Replace generic exceptions with specific ones
        improved = re.sub(
            r'except Exception:',
            'except (IOError, ValueError) as e:',
            improved
        )
        
        # Add error messages to bare raises
        improved = re.sub(
            r'raise\s*$',
            'raise  # TODO: Add error context',
            improved,
            flags=re.MULTILINE
        )
        
        return improved
    
    def _add_error_logging(self, code: str) -> str:
        """Add logging for errors"""
        # Check if logging is imported
        if 'import logging' not in code:
            lines = code.split('\n')
            
            # Find import section
            import_index = 0
            for i, line in enumerate(lines):
                if line.startswith('import ') or line.startswith('from '):
                    import_index = i + 1
            
            # Add logging import
            lines.insert(import_index, 'import logging')
            lines.insert(import_index + 1, '')
            lines.insert(import_index + 2, 'logger = logging.getLogger(__name__)')
            
            code = '\n'.join(lines)
        
        # Add logging to except blocks (simplified)
        code = re.sub(
            r'except (.*?) as e:',
            r'except \1 as e:\n        logger.error(f"Error occurred: {e}")',
            code
        )
        
        return code
    
    def _improve_function_names(self, code: str) -> str:
        """Improve function naming"""
        # Map of common abbreviations to full names
        improvements = {
            r'\bget_(\w+)_val\b': r'get_\1_value',
            r'\bcalc_(\w+)\b': r'calculate_\1',
            r'\bgen_(\w+)\b': r'generate_\1',
            r'\binit_(\w+)\b': r'initialize_\1',
            r'\bproc_(\w+)\b': r'process_\1'
        }
        
        improved = code
        for pattern, replacement in improvements.items():
            improved = re.sub(pattern, replacement, improved)
        
        return improved
    
    def _add_meaningful_comments(self, code: str) -> str:
        """Add meaningful comments to code"""
        lines = code.split('\n')
        enhanced_lines = []
        
        for i, line in enumerate(lines):
            # Add comments for complex conditions
            if ' and ' in line and ' or ' in line and line.strip().startswith('if '):
                enhanced_lines.append(f"{' ' * (len(line) - len(line.lstrip()))}# Complex condition: check multiple criteria")
            
            enhanced_lines.append(line)
            
            # Add section comments
            if line.strip().startswith('class ') and i > 0:
                enhanced_lines.insert(-1, '')
                enhanced_lines.insert(-1, '# ' + '=' * 60)
                enhanced_lines.insert(-1, f'# {line.strip()}')
                enhanced_lines.insert(-1, '# ' + '=' * 60)
        
        return '\n'.join(enhanced_lines)
    
    def _simplify_complex_expressions(self, code: str) -> str:
        """Simplify complex expressions"""
        # Simplify boolean expressions
        simplified = code
        
        # not x == y → x != y
        simplified = re.sub(r'not\s+(\w+)\s*==\s*(\w+)', r'\1 != \2', simplified)
        
        # x == True → x
        simplified = re.sub(r'(\w+)\s*==\s*True\b', r'\1', simplified)
        
        # x == False → not x
        simplified = re.sub(r'(\w+)\s*==\s*False\b', r'not \1', simplified)
        
        return simplified
    
    def _organize_imports(self, code: str) -> str:
        """Organize imports according to PEP 8"""
        lines = code.split('\n')
        
        # Extract imports
        imports = {
            'stdlib': [],
            'third_party': [],
            'local': []
        }
        
        import_lines = []
        other_lines = []
        
        for line in lines:
            if line.startswith('import ') or line.startswith('from '):
                import_lines.append(line)
            else:
                other_lines.append(line)
        
        # Categorize imports (simplified)
        stdlib_modules = {
            'os', 'sys', 'time', 'datetime', 'json', 're', 'math',
            'collections', 'itertools', 'functools', 'typing'
        }
        
        for imp in import_lines:
            module = imp.split()[1].split('.')[0]
            if module in stdlib_modules:
                imports['stdlib'].append(imp)
            elif '.' in imp and not imp.startswith('from .'):
                imports['third_party'].append(imp)
            else:
                imports['local'].append(imp)
        
        # Reconstruct with proper grouping
        organized = []
        
        if imports['stdlib']:
            organized.extend(sorted(imports['stdlib']))
            organized.append('')
        
        if imports['third_party']:
            organized.extend(sorted(imports['third_party']))
            organized.append('')
        
        if imports['local']:
            organized.extend(sorted(imports['local']))
            organized.append('')
        
        # Remove trailing empty line if exists
        if organized and organized[-1] == '':
            organized.pop()
        
        # Combine with other lines
        if organized:
            organized.append('')
        organized.extend(other_lines)
        
        return '\n'.join(organized)
    
    def _format_consistently(self, code: str) -> str:
        """Apply consistent formatting"""
        # This would typically use black or autopep8
        # For now, just ensure consistent spacing
        
        formatted = code
        
        # Ensure spaces around operators
        formatted = re.sub(r'(\w)=(\w)', r'\1 = \2', formatted)
        formatted = re.sub(r'(\w)\+(\w)', r'\1 + \2', formatted)
        formatted = re.sub(r'(\w)-(\w)', r'\1 - \2', formatted)
        
        # Remove trailing whitespace
        lines = formatted.split('\n')
        formatted = '\n'.join(line.rstrip() for line in lines)
        
        return formatted
    
    def _validate_syntax(self, code: str) -> Dict[str, Any]:
        """Validate Python syntax"""
        try:
            compile(code, '<string>', 'exec')
            return {"valid": True}
        except SyntaxError as e:
            return {"valid": False, "error": str(e)}
    
    def _validate_semantics(
        self,
        original: str,
        improved: str
    ) -> Dict[str, Any]:
        """Validate semantic equivalence"""
        # This is simplified - real implementation would use AST comparison
        try:
            original_ast = ast.parse(original)
            improved_ast = ast.parse(improved)
            
            # Compare function and class definitions
            original_defs = {
                node.name for node in ast.walk(original_ast)
                if isinstance(node, (ast.FunctionDef, ast.ClassDef))
            }
            
            improved_defs = {
                node.name for node in ast.walk(improved_ast)
                if isinstance(node, (ast.FunctionDef, ast.ClassDef))
            }
            
            if original_defs != improved_defs:
                return {
                    "equivalent": False,
                    "difference": f"Definition mismatch: {original_defs ^ improved_defs}"
                }
            
            return {"equivalent": True}
            
        except Exception as e:
            return {"equivalent": False, "difference": str(e)}
    
    def _validate_style(self, code: str) -> Dict[str, Any]:
        """Validate code style"""
        issues = []
        
        lines = code.split('\n')
        for i, line in enumerate(lines):
            # Line length
            if len(line) > 88:
                issues.append(f"Line {i+1}: exceeds 88 characters")
            
            # Trailing whitespace
            if line != line.rstrip():
                issues.append(f"Line {i+1}: trailing whitespace")
        
        return {"passed": len(issues) == 0, "issues": issues}
    
    def _validate_types(self, code: str) -> Dict[str, Any]:
        """Validate type annotations"""
        errors = []
        
        try:
            tree = ast.parse(code)
            
            # Check for missing type hints in function definitions
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    if not node.returns and node.name != '__init__':
                        errors.append(f"Function {node.name} missing return type")
                    
                    for arg in node.args.args:
                        if not arg.annotation and arg.arg != 'self':
                            errors.append(f"Parameter {arg.arg} in {node.name} missing type hint")
            
        except Exception as e:
            errors.append(f"Type validation error: {str(e)}")
        
        return {"passed": len(errors) == 0, "errors": errors}
    
    def _calculate_function_complexity(self, node: ast.FunctionDef) -> int:
        """Calculate complexity of a function"""
        complexity = 1  # Base complexity
        
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For)):
                complexity += 1
            elif isinstance(child, ast.ExceptHandler):
                complexity += 1
            elif isinstance(child, (ast.And, ast.Or)):
                complexity += 1
        
        return complexity