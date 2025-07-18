"""Scope reasoner agent for task analysis and planning"""

from typing import List, Dict, Any
from pydantic_ai import RunContext
from .base_agent import BaseAgent
from .dependencies import ScopeDependencies
from .models import ScopeOutput, TaskComplexity
import logfire


class ScopeReasonerAgent(BaseAgent[ScopeDependencies, ScopeOutput]):
    """Agent that analyzes task scope and creates implementation plans"""
    
    def __init__(self):
        """Initialize the scope reasoner agent"""
        super().__init__(
            model='openai:gpt-4',
            deps_type=ScopeDependencies,
            result_type=ScopeOutput,
            enable_logfire=True
        )
    
    def get_system_prompt(self) -> str:
        """Get the system prompt for the scope reasoner"""
        return """You are an expert at analyzing task requirements and creating detailed implementation plans.
        
        Your responsibilities:
        1. Break down complex tasks into manageable subtasks
        2. Assess task complexity accurately
        3. Estimate effort and time requirements
        4. Identify dependencies between tasks
        5. Identify potential risks and challenges
        
        Use structured reasoning to:
        - Analyze the full scope of work
        - Consider edge cases and error handling
        - Plan for testing and validation
        - Ensure all requirements are addressed
        
        Provide clear, actionable breakdowns that other agents can execute."""
    
    def _register_tools(self):
        """Register tools for the scope reasoner"""
        self.agent.tool(self.analyze_requirements)
        self.agent.tool(self.create_task_breakdown)
        self.agent.tool(self.estimate_complexity)
        self.agent.tool(self.identify_dependencies)
        self.agent.tool(self.assess_risks)
    
    async def analyze_requirements(
        self,
        ctx: RunContext[ScopeDependencies],
        task_description: str
    ) -> Dict[str, Any]:
        """Analyze task requirements in detail
        
        Args:
            ctx: Run context
            task_description: Task to analyze
            
        Returns:
            Detailed requirements analysis
        """
        logfire.info("analyzing_requirements", task=task_description[:50])
        
        # Extract key requirements
        requirements = {
            "functional": self._extract_functional_requirements(task_description),
            "non_functional": self._extract_non_functional_requirements(task_description),
            "constraints": self._identify_constraints(task_description),
            "success_criteria": self._define_success_criteria(task_description)
        }
        
        # Check against project context if available
        if ctx.deps.project_context:
            requirements["context_alignment"] = self._check_context_alignment(
                requirements,
                ctx.deps.project_context
            )
        
        return requirements
    
    async def create_task_breakdown(
        self,
        ctx: RunContext[ScopeDependencies],
        requirements: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Create detailed task breakdown
        
        Args:
            ctx: Run context
            requirements: Analyzed requirements
            
        Returns:
            List of subtasks with details
        """
        logfire.info("creating_task_breakdown")
        
        subtasks = []
        
        # Break down functional requirements
        for req in requirements.get("functional", []):
            tasks = self._decompose_requirement(req)
            subtasks.extend(tasks)
        
        # Add supporting tasks
        subtasks.extend(self._add_supporting_tasks(requirements))
        
        # Order tasks logically
        ordered_tasks = self._order_tasks(subtasks)
        
        # Add metadata to each task
        for i, task in enumerate(ordered_tasks):
            task.update({
                "id": f"task_{i+1}",
                "priority": self._calculate_priority(task),
                "estimated_effort": self._estimate_task_effort(task)
            })
        
        return ordered_tasks
    
    async def estimate_complexity(
        self,
        ctx: RunContext[ScopeDependencies],
        task_breakdown: List[Dict[str, Any]]
    ) -> TaskComplexity:
        """Estimate overall task complexity
        
        Args:
            ctx: Run context
            task_breakdown: List of subtasks
            
        Returns:
            Task complexity assessment
        """
        logfire.info("estimating_complexity", task_count=len(task_breakdown))
        
        # Calculate complexity factors
        factors = {
            "task_count": len(task_breakdown),
            "max_depth": self._calculate_task_depth(task_breakdown),
            "interdependencies": self._count_interdependencies(task_breakdown),
            "technical_complexity": self._assess_technical_complexity(task_breakdown),
            "uncertainty": self._assess_uncertainty(task_breakdown)
        }
        
        # Weight factors
        complexity_score = (
            factors["task_count"] * 0.2 +
            factors["max_depth"] * 0.2 +
            factors["interdependencies"] * 0.3 +
            factors["technical_complexity"] * 0.2 +
            factors["uncertainty"] * 0.1
        )
        
        # Map to complexity level
        if complexity_score < 3:
            return TaskComplexity.SIMPLE
        elif complexity_score < 6:
            return TaskComplexity.COMPLEX
        else:
            return TaskComplexity.RESEARCH
    
    async def identify_dependencies(
        self,
        ctx: RunContext[ScopeDependencies],
        task_breakdown: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Identify dependencies between tasks
        
        Args:
            ctx: Run context
            task_breakdown: List of subtasks
            
        Returns:
            List of dependencies
        """
        logfire.info("identifying_dependencies")
        
        dependencies = []
        
        for i, task in enumerate(task_breakdown):
            task_deps = []
            
            # Check for explicit dependencies
            if "requires" in task:
                task_deps.extend(task["requires"])
            
            # Infer dependencies based on task type
            inferred = self._infer_dependencies(task, task_breakdown[:i])
            task_deps.extend(inferred)
            
            if task_deps:
                dependencies.append({
                    "task_id": task["id"],
                    "depends_on": task_deps,
                    "type": self._classify_dependency_type(task_deps)
                })
        
        return dependencies
    
    async def assess_risks(
        self,
        ctx: RunContext[ScopeDependencies],
        task_breakdown: List[Dict[str, Any]],
        dependencies: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Assess risks in the implementation plan
        
        Args:
            ctx: Run context
            task_breakdown: List of subtasks
            dependencies: Task dependencies
            
        Returns:
            List of identified risks
        """
        logfire.info("assessing_risks")
        
        risks = []
        
        # Technical risks
        tech_risks = self._assess_technical_risks(task_breakdown)
        risks.extend(tech_risks)
        
        # Dependency risks
        dep_risks = self._assess_dependency_risks(dependencies)
        risks.extend(dep_risks)
        
        # Resource risks
        resource_risks = self._assess_resource_risks(task_breakdown)
        risks.extend(resource_risks)
        
        # Historical risks (from task history)
        if ctx.deps.task_history:
            historical_risks = self._analyze_historical_risks(ctx.deps.task_history)
            risks.extend(historical_risks)
        
        # Rank risks by impact
        for risk in risks:
            risk["impact_score"] = self._calculate_risk_impact(risk)
        
        risks.sort(key=lambda x: x["impact_score"], reverse=True)
        
        return risks
    
    def _extract_functional_requirements(self, task_description: str) -> List[str]:
        """Extract functional requirements from task description"""
        requirements = []
        
        # Common functional requirement patterns
        patterns = [
            "implement", "create", "add", "update", "remove",
            "integrate", "connect", "process", "calculate", "generate"
        ]
        
        sentences = task_description.split('.')
        for sentence in sentences:
            sentence_lower = sentence.lower().strip()
            if any(pattern in sentence_lower for pattern in patterns):
                requirements.append(sentence.strip())
        
        return requirements or [task_description]
    
    def _extract_non_functional_requirements(self, task_description: str) -> List[str]:
        """Extract non-functional requirements"""
        nfr = []
        task_lower = task_description.lower()
        
        # Performance requirements
        if any(word in task_lower for word in ["fast", "performance", "efficient", "optimize"]):
            nfr.append("Performance optimization required")
        
        # Security requirements
        if any(word in task_lower for word in ["secure", "auth", "encrypt", "permission"]):
            nfr.append("Security measures required")
        
        # Scalability requirements
        if any(word in task_lower for word in ["scale", "concurrent", "parallel", "distributed"]):
            nfr.append("Scalability considerations")
        
        # Reliability requirements
        if any(word in task_lower for word in ["reliable", "fault", "resilient", "recovery"]):
            nfr.append("Reliability and fault tolerance")
        
        return nfr
    
    def _identify_constraints(self, task_description: str) -> List[str]:
        """Identify constraints in the task"""
        constraints = []
        task_lower = task_description.lower()
        
        # Time constraints
        if any(word in task_lower for word in ["deadline", "by", "within", "urgent"]):
            constraints.append("Time constraint identified")
        
        # Technology constraints
        if any(word in task_lower for word in ["must use", "required", "only", "specific"]):
            constraints.append("Technology constraint identified")
        
        # Resource constraints
        if any(word in task_lower for word in ["limited", "budget", "resource", "constraint"]):
            constraints.append("Resource constraint identified")
        
        return constraints
    
    def _define_success_criteria(self, task_description: str) -> List[str]:
        """Define success criteria for the task"""
        criteria = []
        
        # Always include basic criteria
        criteria.append("Code compiles without errors")
        criteria.append("All tests pass")
        criteria.append("Code follows project conventions")
        
        # Add specific criteria based on task type
        task_lower = task_description.lower()
        
        if "api" in task_lower or "endpoint" in task_lower:
            criteria.append("API endpoints respond correctly")
            criteria.append("Proper error handling implemented")
        
        if "ui" in task_lower or "interface" in task_lower:
            criteria.append("UI is responsive and accessible")
            criteria.append("User interactions work as expected")
        
        if "performance" in task_lower:
            criteria.append("Performance targets met")
            criteria.append("No performance regressions")
        
        return criteria
    
    def _decompose_requirement(self, requirement: str) -> List[Dict[str, Any]]:
        """Decompose a requirement into tasks"""
        tasks = []
        
        # Basic task template
        base_task = {
            "description": requirement,
            "type": self._classify_task_type(requirement),
            "tags": self._extract_task_tags(requirement)
        }
        
        # Add subtasks based on task type
        task_type = base_task["type"]
        
        if task_type == "implementation":
            tasks.extend([
                {**base_task, "phase": "design", "description": f"Design {requirement}"},
                {**base_task, "phase": "implement", "description": f"Implement {requirement}"},
                {**base_task, "phase": "test", "description": f"Test {requirement}"}
            ])
        elif task_type == "integration":
            tasks.extend([
                {**base_task, "phase": "analysis", "description": f"Analyze integration points for {requirement}"},
                {**base_task, "phase": "implement", "description": f"Implement integration for {requirement}"},
                {**base_task, "phase": "validate", "description": f"Validate integration for {requirement}"}
            ])
        else:
            tasks.append(base_task)
        
        return tasks
    
    def _classify_task_type(self, requirement: str) -> str:
        """Classify the type of task"""
        req_lower = requirement.lower()
        
        if "integrate" in req_lower or "connect" in req_lower:
            return "integration"
        elif "test" in req_lower or "validate" in req_lower:
            return "testing"
        elif "document" in req_lower or "explain" in req_lower:
            return "documentation"
        elif "fix" in req_lower or "debug" in req_lower:
            return "bugfix"
        elif "optimize" in req_lower or "improve" in req_lower:
            return "optimization"
        else:
            return "implementation"
    
    def _extract_task_tags(self, requirement: str) -> List[str]:
        """Extract tags from requirement"""
        tags = []
        req_lower = requirement.lower()
        
        # Technology tags
        tech_keywords = ["api", "database", "ui", "backend", "frontend", "service"]
        tags.extend([kw for kw in tech_keywords if kw in req_lower])
        
        # Action tags
        action_keywords = ["create", "update", "delete", "read", "process"]
        tags.extend([kw for kw in action_keywords if kw in req_lower])
        
        return tags
    
    def _add_supporting_tasks(self, requirements: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Add supporting tasks like testing and documentation"""
        supporting = []
        
        # Always add testing tasks
        supporting.append({
            "description": "Write unit tests",
            "type": "testing",
            "phase": "testing",
            "tags": ["testing", "quality"]
        })
        
        # Add documentation if complex
        if len(requirements.get("functional", [])) > 3:
            supporting.append({
                "description": "Update documentation",
                "type": "documentation",
                "phase": "documentation",
                "tags": ["documentation"]
            })
        
        # Add validation task
        supporting.append({
            "description": "Validate implementation against requirements",
            "type": "validation",
            "phase": "validation",
            "tags": ["validation", "quality"]
        })
        
        return supporting
    
    def _order_tasks(self, tasks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Order tasks logically"""
        # Define phase order
        phase_order = {
            "design": 1,
            "analysis": 2,
            "implement": 3,
            "implementation": 3,
            "test": 4,
            "testing": 4,
            "validate": 5,
            "validation": 5,
            "documentation": 6
        }
        
        # Sort by phase
        def get_phase_order(task):
            phase = task.get("phase", task.get("type", "implement"))
            return phase_order.get(phase, 99)
        
        return sorted(tasks, key=get_phase_order)
    
    def _calculate_priority(self, task: Dict[str, Any]) -> str:
        """Calculate task priority"""
        # Critical tasks
        if task.get("type") in ["bugfix", "security"]:
            return "high"
        
        # Core functionality
        if task.get("phase") in ["implement", "implementation"]:
            return "high"
        
        # Supporting tasks
        if task.get("type") in ["testing", "validation"]:
            return "medium"
        
        # Documentation and optimization
        if task.get("type") in ["documentation", "optimization"]:
            return "low"
        
        return "medium"
    
    def _estimate_task_effort(self, task: Dict[str, Any]) -> str:
        """Estimate effort for a task"""
        task_type = task.get("type", "")
        
        # Simple estimates based on task type
        effort_map = {
            "implementation": "2-4 hours",
            "integration": "4-8 hours",
            "testing": "1-2 hours",
            "documentation": "1 hour",
            "bugfix": "1-4 hours",
            "optimization": "2-6 hours"
        }
        
        return effort_map.get(task_type, "2-4 hours")
    
    def _calculate_task_depth(self, tasks: List[Dict[str, Any]]) -> int:
        """Calculate maximum task depth"""
        # For now, return a simple estimate based on task count
        if len(tasks) < 5:
            return 1
        elif len(tasks) < 10:
            return 2
        else:
            return 3
    
    def _count_interdependencies(self, tasks: List[Dict[str, Any]]) -> int:
        """Count interdependencies between tasks"""
        # Estimate based on task types
        integration_tasks = sum(1 for t in tasks if t.get("type") == "integration")
        return integration_tasks * 2
    
    def _assess_technical_complexity(self, tasks: List[Dict[str, Any]]) -> float:
        """Assess technical complexity of tasks"""
        complexity_scores = {
            "implementation": 1.0,
            "integration": 2.0,
            "optimization": 2.5,
            "bugfix": 1.5,
            "testing": 0.5,
            "documentation": 0.3
        }
        
        total_score = sum(
            complexity_scores.get(t.get("type", "implementation"), 1.0)
            for t in tasks
        )
        
        return min(total_score / max(len(tasks), 1), 10.0)
    
    def _assess_uncertainty(self, tasks: List[Dict[str, Any]]) -> float:
        """Assess uncertainty in tasks"""
        # Look for uncertainty indicators
        uncertainty_keywords = ["might", "maybe", "possibly", "explore", "investigate"]
        
        uncertain_count = sum(
            1 for t in tasks
            if any(kw in t.get("description", "").lower() for kw in uncertainty_keywords)
        )
        
        return min(uncertain_count / max(len(tasks), 1) * 10, 10.0)
    
    def _infer_dependencies(
        self,
        task: Dict[str, Any],
        previous_tasks: List[Dict[str, Any]]
    ) -> List[str]:
        """Infer task dependencies"""
        deps = []
        
        # Testing depends on implementation
        if task.get("type") == "testing":
            for prev in previous_tasks:
                if prev.get("type") == "implementation":
                    deps.append(prev.get("id", ""))
        
        # Documentation depends on everything
        if task.get("type") == "documentation":
            deps.extend([t.get("id", "") for t in previous_tasks])
        
        # Integration depends on relevant implementations
        if task.get("type") == "integration":
            for prev in previous_tasks:
                if prev.get("type") == "implementation":
                    # Check if tags overlap
                    task_tags = set(task.get("tags", []))
                    prev_tags = set(prev.get("tags", []))
                    if task_tags.intersection(prev_tags):
                        deps.append(prev.get("id", ""))
        
        return [d for d in deps if d]  # Filter out empty strings
    
    def _classify_dependency_type(self, dependencies: List[str]) -> str:
        """Classify type of dependencies"""
        if len(dependencies) == 0:
            return "independent"
        elif len(dependencies) == 1:
            return "sequential"
        else:
            return "complex"
    
    def _assess_technical_risks(self, tasks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Assess technical risks"""
        risks = []
        
        # Integration complexity
        integration_count = sum(1 for t in tasks if t.get("type") == "integration")
        if integration_count > 2:
            risks.append({
                "type": "technical",
                "description": "Multiple integrations increase complexity",
                "severity": "medium",
                "mitigation": "Plan integration testing thoroughly"
            })
        
        # Performance risks
        if any("performance" in t.get("description", "").lower() for t in tasks):
            risks.append({
                "type": "technical",
                "description": "Performance requirements may be challenging",
                "severity": "medium",
                "mitigation": "Implement performance testing early"
            })
        
        return risks
    
    def _assess_dependency_risks(self, dependencies: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Assess dependency-related risks"""
        risks = []
        
        # Complex dependencies
        complex_deps = [d for d in dependencies if d.get("type") == "complex"]
        if complex_deps:
            risks.append({
                "type": "dependency",
                "description": f"{len(complex_deps)} tasks have complex dependencies",
                "severity": "high",
                "mitigation": "Consider parallelizing where possible"
            })
        
        return risks
    
    def _assess_resource_risks(self, tasks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Assess resource-related risks"""
        risks = []
        
        # Time risk
        total_effort_hours = len(tasks) * 3  # Rough estimate
        if total_effort_hours > 20:
            risks.append({
                "type": "resource",
                "description": "Significant time investment required",
                "severity": "medium",
                "mitigation": "Consider phased delivery"
            })
        
        return risks
    
    def _analyze_historical_risks(self, task_history: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Analyze risks from historical data"""
        risks = []
        
        # Look for patterns in failures
        failures = [t for t in task_history if not t.get("success", True)]
        if len(failures) > len(task_history) * 0.2:  # >20% failure rate
            risks.append({
                "type": "historical",
                "description": "Historical data shows high failure rate",
                "severity": "high",
                "mitigation": "Review past failures and adjust approach"
            })
        
        return risks
    
    def _calculate_risk_impact(self, risk: Dict[str, Any]) -> float:
        """Calculate risk impact score"""
        severity_scores = {
            "low": 1.0,
            "medium": 2.0,
            "high": 3.0
        }
        
        type_multipliers = {
            "technical": 1.2,
            "dependency": 1.5,
            "resource": 1.0,
            "historical": 1.3
        }
        
        base_score = severity_scores.get(risk.get("severity", "medium"), 2.0)
        multiplier = type_multipliers.get(risk.get("type", "technical"), 1.0)
        
        return base_score * multiplier
    
    def _check_context_alignment(
        self,
        requirements: Dict[str, Any],
        project_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Check if requirements align with project context"""
        alignment = {
            "aligned": True,
            "conflicts": [],
            "suggestions": []
        }
        
        # Check against project conventions
        if "conventions" in project_context:
            # Add relevant checks
            pass
        
        # Check against existing architecture
        if "architecture" in project_context:
            # Add relevant checks
            pass
        
        return alignment