"""Validation gate system for quality control"""

from typing import Dict, Any, List, Optional, Callable, Union
from datetime import datetime
from pydantic import BaseModel, Field
from enum import Enum
import logfire
import re
import ast


class ValidationLevel(str, Enum):
    """Validation severity levels"""
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


class ValidationResult(BaseModel):
    """Result of a validation check"""
    passed: bool = Field(description="Whether validation passed")
    level: ValidationLevel = Field(description="Severity level")
    message: str = Field(description="Validation message")
    details: Optional[Dict[str, Any]] = Field(default=None, description="Additional details")
    rule_name: Optional[str] = Field(default=None, description="Name of the rule that generated this result")


class ValidationRule(BaseModel):
    """Validation rule definition"""
    name: str = Field(description="Rule name")
    description: str = Field(description="Rule description")
    level: ValidationLevel = Field(default=ValidationLevel.ERROR, description="Default severity level")
    enabled: bool = Field(default=True, description="Whether rule is enabled")
    tags: List[str] = Field(default_factory=list, description="Rule tags")
    config: Dict[str, Any] = Field(default_factory=dict, description="Rule configuration")


class ValidationGate:
    """Validation gate for checking various outputs"""
    
    def __init__(self):
        """Initialize validation gate"""
        self._logger = logfire.span("validation_gate")
        self._rules: Dict[str, ValidationRule] = {}
        self._validators: Dict[str, Callable] = {}
        self._initialize_default_rules()
    
    def _initialize_default_rules(self):
        """Initialize default validation rules"""
        # Code validation rules
        self.register_rule(
            ValidationRule(
                name="syntax_check",
                description="Check code syntax",
                level=ValidationLevel.ERROR,
                tags=["code", "syntax"]
            ),
            self._validate_syntax
        )
        
        self.register_rule(
            ValidationRule(
                name="import_check",
                description="Check for valid imports",
                level=ValidationLevel.ERROR,
                tags=["code", "imports"]
            ),
            self._validate_imports
        )
        
        self.register_rule(
            ValidationRule(
                name="security_check",
                description="Check for security issues",
                level=ValidationLevel.ERROR,
                tags=["code", "security"]
            ),
            self._validate_security
        )
        
        # Documentation validation rules
        self.register_rule(
            ValidationRule(
                name="docstring_check",
                description="Check for docstrings",
                level=ValidationLevel.WARNING,
                tags=["code", "documentation"]
            ),
            self._validate_docstrings
        )
        
        # Output validation rules
        self.register_rule(
            ValidationRule(
                name="format_check",
                description="Check output format",
                level=ValidationLevel.ERROR,
                tags=["output", "format"]
            ),
            self._validate_format
        )
        
        self.register_rule(
            ValidationRule(
                name="completeness_check",
                description="Check output completeness",
                level=ValidationLevel.WARNING,
                tags=["output", "quality"]
            ),
            self._validate_completeness
        )
        
        # Performance validation rules
        self.register_rule(
            ValidationRule(
                name="complexity_check",
                description="Check code complexity",
                level=ValidationLevel.WARNING,
                tags=["code", "performance"],
                config={"max_complexity": 10}
            ),
            self._validate_complexity
        )
        
        # Test validation rules
        self.register_rule(
            ValidationRule(
                name="test_coverage_check",
                description="Check test coverage",
                level=ValidationLevel.WARNING,
                tags=["test", "quality"],
                config={"min_coverage": 0.8}
            ),
            self._validate_test_coverage
        )
    
    def register_rule(self, rule: ValidationRule, validator: Callable) -> None:
        """Register a validation rule
        
        Args:
            rule: Validation rule
            validator: Validator function
        """
        self._rules[rule.name] = rule
        self._validators[rule.name] = validator
        
        self._logger.info(
            "Rule registered",
            rule_name=rule.name,
            tags=rule.tags
        )
    
    def unregister_rule(self, rule_name: str) -> None:
        """Unregister a validation rule
        
        Args:
            rule_name: Name of rule to unregister
        """
        if rule_name in self._rules:
            del self._rules[rule_name]
            del self._validators[rule_name]
            
            self._logger.info("Rule unregistered", rule_name=rule_name)
    
    async def validate(
        self,
        data: Dict[str, Any],
        tags: Optional[List[str]] = None,
        exclude_rules: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Validate data against rules
        
        Args:
            data: Data to validate
            tags: Only run rules with these tags
            exclude_rules: Rules to exclude
            
        Returns:
            Validation results
        """
        results = {
            "passed": True,
            "errors": [],
            "warnings": [],
            "info": [],
            "timestamp": datetime.utcnow().isoformat(),
            "rules_run": 0,
            "data_type": data.get("type", "unknown")
        }
        
        # Select rules to run
        rules_to_run = []
        for rule_name, rule in self._rules.items():
            # Skip disabled rules
            if not rule.enabled:
                continue
            
            # Skip excluded rules
            if exclude_rules and rule_name in exclude_rules:
                continue
            
            # Filter by tags
            if tags and not any(tag in rule.tags for tag in tags):
                continue
            
            rules_to_run.append(rule_name)
        
        # Run validations
        for rule_name in rules_to_run:
            rule = self._rules[rule_name]
            validator = self._validators[rule_name]
            
            try:
                result = await validator(data, rule)
                result.rule_name = rule_name
                
                # Categorize result
                if result.level == ValidationLevel.ERROR:
                    results["errors"].append(result.dict())
                    if not result.passed:
                        results["passed"] = False
                elif result.level == ValidationLevel.WARNING:
                    results["warnings"].append(result.dict())
                else:
                    results["info"].append(result.dict())
                
                results["rules_run"] += 1
                
            except Exception as e:
                self._logger.error(
                    "Validation rule failed",
                    rule_name=rule_name,
                    error=str(e)
                )
                
                # Add error result
                error_result = ValidationResult(
                    passed=False,
                    level=ValidationLevel.ERROR,
                    message=f"Rule '{rule_name}' failed: {str(e)}",
                    rule_name=rule_name
                )
                results["errors"].append(error_result.dict())
                results["passed"] = False
        
        # Add summary
        results["summary"] = {
            "total_rules": results["rules_run"],
            "errors": len(results["errors"]),
            "warnings": len(results["warnings"]),
            "info": len(results["info"])
        }
        
        self._logger.info(
            "Validation complete",
            passed=results["passed"],
            rules_run=results["rules_run"],
            errors=len(results["errors"]),
            warnings=len(results["warnings"])
        )
        
        return results
    
    async def _validate_syntax(self, data: Dict[str, Any], rule: ValidationRule) -> ValidationResult:
        """Validate code syntax"""
        if data.get("type") != "code":
            return ValidationResult(
                passed=True,
                level=ValidationLevel.INFO,
                message="Not applicable to non-code data"
            )
        
        code = data.get("content", "")
        language = data.get("language", "python")
        
        if language == "python":
            try:
                compile(code, '<string>', 'exec')
                return ValidationResult(
                    passed=True,
                    level=ValidationLevel.INFO,
                    message="Syntax is valid"
                )
            except SyntaxError as e:
                return ValidationResult(
                    passed=False,
                    level=ValidationLevel.ERROR,
                    message=f"Syntax error: {str(e)}",
                    details={
                        "line": e.lineno,
                        "offset": e.offset,
                        "text": e.text
                    }
                )
        
        return ValidationResult(
            passed=True,
            level=ValidationLevel.INFO,
            message=f"Syntax validation not implemented for {language}"
        )
    
    async def _validate_imports(self, data: Dict[str, Any], rule: ValidationRule) -> ValidationResult:
        """Validate imports"""
        if data.get("type") != "code":
            return ValidationResult(
                passed=True,
                level=ValidationLevel.INFO,
                message="Not applicable to non-code data"
            )
        
        code = data.get("content", "")
        language = data.get("language", "python")
        allowed_imports = data.get("allowed_imports", [])
        
        if language == "python":
            try:
                tree = ast.parse(code)
                imports = []
                
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            imports.append(alias.name)
                    elif isinstance(node, ast.ImportFrom):
                        module = node.module or ""
                        imports.append(module)
                
                # Check against allowed imports
                if allowed_imports:
                    disallowed = [imp for imp in imports if imp not in allowed_imports]
                    if disallowed:
                        return ValidationResult(
                            passed=False,
                            level=ValidationLevel.ERROR,
                            message=f"Disallowed imports: {', '.join(disallowed)}",
                            details={"disallowed_imports": disallowed}
                        )
                
                return ValidationResult(
                    passed=True,
                    level=ValidationLevel.INFO,
                    message=f"All imports are valid ({len(imports)} imports)"
                )
                
            except Exception as e:
                return ValidationResult(
                    passed=False,
                    level=ValidationLevel.ERROR,
                    message=f"Failed to analyze imports: {str(e)}"
                )
        
        return ValidationResult(
            passed=True,
            level=ValidationLevel.INFO,
            message="Import validation not implemented for this language"
        )
    
    async def _validate_security(self, data: Dict[str, Any], rule: ValidationRule) -> ValidationResult:
        """Validate security issues"""
        if data.get("type") != "code":
            return ValidationResult(
                passed=True,
                level=ValidationLevel.INFO,
                message="Not applicable to non-code data"
            )
        
        code = data.get("content", "")
        issues = []
        
        # Check for dangerous patterns
        dangerous_patterns = [
            (r'\beval\s*\(', "Use of eval() is dangerous"),
            (r'\bexec\s*\(', "Use of exec() is dangerous"),
            (r'__import__', "Dynamic imports can be dangerous"),
            (r'os\.system', "Direct system calls are dangerous"),
            (r'subprocess\.call\s*\([^,]+,\s*shell\s*=\s*True', "Shell=True is dangerous"),
            (r'pickle\.loads', "Unpickling untrusted data is dangerous"),
            (r'password\s*=\s*["\'][^"\']+["\']', "Hardcoded password detected"),
            (r'api_key\s*=\s*["\'][^"\']+["\']', "Hardcoded API key detected")
        ]
        
        for pattern, message in dangerous_patterns:
            if re.search(pattern, code, re.IGNORECASE):
                issues.append(message)
        
        if issues:
            return ValidationResult(
                passed=False,
                level=ValidationLevel.ERROR,
                message=f"Security issues found: {len(issues)}",
                details={"issues": issues}
            )
        
        return ValidationResult(
            passed=True,
            level=ValidationLevel.INFO,
            message="No security issues detected"
        )
    
    async def _validate_docstrings(self, data: Dict[str, Any], rule: ValidationRule) -> ValidationResult:
        """Validate docstrings"""
        if data.get("type") != "code" or data.get("language") != "python":
            return ValidationResult(
                passed=True,
                level=ValidationLevel.INFO,
                message="Not applicable"
            )
        
        code = data.get("content", "")
        
        try:
            tree = ast.parse(code)
            missing_docstrings = []
            
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                    if not ast.get_docstring(node):
                        missing_docstrings.append(node.name)
            
            if missing_docstrings:
                return ValidationResult(
                    passed=False,
                    level=ValidationLevel.WARNING,
                    message=f"Missing docstrings: {', '.join(missing_docstrings)}",
                    details={"missing": missing_docstrings}
                )
            
            return ValidationResult(
                passed=True,
                level=ValidationLevel.INFO,
                message="All functions and classes have docstrings"
            )
            
        except Exception as e:
            return ValidationResult(
                passed=True,
                level=ValidationLevel.INFO,
                message=f"Could not analyze docstrings: {str(e)}"
            )
    
    async def _validate_format(self, data: Dict[str, Any], rule: ValidationRule) -> ValidationResult:
        """Validate output format"""
        data_type = data.get("type", "unknown")
        
        # Check required fields based on type
        required_fields = {
            "code": ["content", "language"],
            "documentation": ["content", "format"],
            "test": ["test_cases", "framework"],
            "response": ["message", "status"]
        }
        
        if data_type in required_fields:
            missing = [field for field in required_fields[data_type] if field not in data]
            
            if missing:
                return ValidationResult(
                    passed=False,
                    level=ValidationLevel.ERROR,
                    message=f"Missing required fields: {', '.join(missing)}",
                    details={"missing_fields": missing}
                )
        
        return ValidationResult(
            passed=True,
            level=ValidationLevel.INFO,
            message="Format is valid"
        )
    
    async def _validate_completeness(self, data: Dict[str, Any], rule: ValidationRule) -> ValidationResult:
        """Validate output completeness"""
        data_type = data.get("type", "unknown")
        
        if data_type == "code":
            code = data.get("content", "")
            
            # Check for TODOs or incomplete sections
            todo_count = len(re.findall(r'TODO|FIXME|XXX', code, re.IGNORECASE))
            if todo_count > 0:
                return ValidationResult(
                    passed=False,
                    level=ValidationLevel.WARNING,
                    message=f"Found {todo_count} TODO/FIXME markers",
                    details={"todo_count": todo_count}
                )
            
            # Check for pass statements (potential incomplete implementations)
            pass_count = len(re.findall(r'^\s*pass\s*$', code, re.MULTILINE))
            if pass_count > 2:
                return ValidationResult(
                    passed=False,
                    level=ValidationLevel.WARNING,
                    message=f"Found {pass_count} pass statements (possibly incomplete)",
                    details={"pass_count": pass_count}
                )
        
        return ValidationResult(
            passed=True,
            level=ValidationLevel.INFO,
            message="Output appears complete"
        )
    
    async def _validate_complexity(self, data: Dict[str, Any], rule: ValidationRule) -> ValidationResult:
        """Validate code complexity"""
        if data.get("type") != "code" or data.get("language") != "python":
            return ValidationResult(
                passed=True,
                level=ValidationLevel.INFO,
                message="Not applicable"
            )
        
        code = data.get("content", "")
        max_complexity = rule.config.get("max_complexity", 10)
        
        try:
            tree = ast.parse(code)
            complex_functions = []
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    complexity = self._calculate_cyclomatic_complexity(node)
                    if complexity > max_complexity:
                        complex_functions.append({
                            "name": node.name,
                            "complexity": complexity
                        })
            
            if complex_functions:
                return ValidationResult(
                    passed=False,
                    level=ValidationLevel.WARNING,
                    message=f"Functions exceed complexity threshold ({max_complexity})",
                    details={"complex_functions": complex_functions}
                )
            
            return ValidationResult(
                passed=True,
                level=ValidationLevel.INFO,
                message="Code complexity is acceptable"
            )
            
        except Exception as e:
            return ValidationResult(
                passed=True,
                level=ValidationLevel.INFO,
                message=f"Could not analyze complexity: {str(e)}"
            )
    
    def _calculate_cyclomatic_complexity(self, node: ast.AST) -> int:
        """Calculate cyclomatic complexity of a function"""
        complexity = 1
        
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.ExceptHandler)):
                complexity += 1
            elif isinstance(child, ast.BoolOp):
                complexity += len(child.values) - 1
        
        return complexity
    
    async def _validate_test_coverage(self, data: Dict[str, Any], rule: ValidationRule) -> ValidationResult:
        """Validate test coverage"""
        if data.get("type") != "test":
            return ValidationResult(
                passed=True,
                level=ValidationLevel.INFO,
                message="Not applicable to non-test data"
            )
        
        coverage = data.get("coverage", 0.0)
        min_coverage = rule.config.get("min_coverage", 0.8)
        
        if coverage < min_coverage:
            return ValidationResult(
                passed=False,
                level=ValidationLevel.WARNING,
                message=f"Test coverage {coverage:.1%} is below threshold {min_coverage:.1%}",
                details={
                    "coverage": coverage,
                    "threshold": min_coverage
                }
            )
        
        return ValidationResult(
            passed=True,
            level=ValidationLevel.INFO,
            message=f"Test coverage {coverage:.1%} meets threshold"
        )
    
    def get_rules(self, tags: Optional[List[str]] = None) -> List[ValidationRule]:
        """Get validation rules
        
        Args:
            tags: Filter by tags
            
        Returns:
            List of validation rules
        """
        rules = []
        
        for rule in self._rules.values():
            if tags and not any(tag in rule.tags for tag in tags):
                continue
            rules.append(rule)
        
        return rules
    
    def enable_rule(self, rule_name: str) -> None:
        """Enable a validation rule
        
        Args:
            rule_name: Name of rule to enable
        """
        if rule_name in self._rules:
            self._rules[rule_name].enabled = True
            self._logger.info("Rule enabled", rule_name=rule_name)
    
    def disable_rule(self, rule_name: str) -> None:
        """Disable a validation rule
        
        Args:
            rule_name: Name of rule to disable
        """
        if rule_name in self._rules:
            self._rules[rule_name].enabled = False
            self._logger.info("Rule disabled", rule_name=rule_name)
    
    def configure_rule(self, rule_name: str, config: Dict[str, Any]) -> None:
        """Configure a validation rule
        
        Args:
            rule_name: Name of rule to configure
            config: Configuration to apply
        """
        if rule_name in self._rules:
            self._rules[rule_name].config.update(config)
            self._logger.info("Rule configured", rule_name=rule_name, config=config)