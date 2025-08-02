"""Component Advisor agent that recommends tools, patterns, and models for agent generation"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from pydantic_ai import RunContext
from .base_agent import BaseAgent
from .dependencies import BaseDependencies
import json
import logfire
from pathlib import Path


class ComponentRecommendation(BaseModel):
    """A recommended component"""
    name: str = Field(description="Component name")
    type: str = Field(description="Component type: tool/pattern/example/prompt")
    relevance_score: float = Field(description="Relevance score 0-1")
    reason: str = Field(description="Why this component is recommended")
    usage_example: Optional[str] = Field(default=None, description="Example usage")
    dependencies: List[str] = Field(default_factory=list)


class ModelRecommendation(BaseModel):
    """Recommended AI model configuration"""
    model_id: str = Field(description="Model identifier")
    reason: str = Field(description="Why this model is recommended")
    estimated_cost: Optional[float] = Field(default=None)
    performance_score: float = Field(description="Expected performance 0-1")
    alternatives: List[str] = Field(default_factory=list)


class ArchitectureRecommendation(BaseModel):
    """Recommended architecture pattern"""
    pattern_name: str = Field(description="Architecture pattern name")
    description: str = Field(description="Pattern description")
    benefits: List[str] = Field(description="Benefits of this pattern")
    considerations: List[str] = Field(description="Things to consider")
    example_implementation: Optional[str] = Field(default=None)


class ComponentAdvisorOutput(BaseModel):
    """Complete component advisor recommendations"""
    task_analysis: str = Field(description="Analysis of the task")
    recommended_components: List[ComponentRecommendation] = Field(description="Recommended components")
    recommended_models: List[ModelRecommendation] = Field(description="Recommended AI models")
    architecture_patterns: List[ArchitectureRecommendation] = Field(description="Recommended patterns")
    implementation_steps: List[str] = Field(description="Suggested implementation steps")
    potential_challenges: List[str] = Field(description="Potential challenges to consider")
    estimated_complexity: str = Field(description="simple/moderate/complex")


class ComponentAdvisorDependencies(BaseDependencies):
    """Dependencies for component advisor agent"""
    component_library_path: str = Field(default="src/library")
    model_catalog: Dict[str, Any] = Field(default_factory=dict)
    pattern_registry: Dict[str, Any] = Field(default_factory=dict)


class ComponentAdvisorAgent(BaseAgent[ComponentAdvisorDependencies, ComponentAdvisorOutput]):
    """Agent that analyzes requirements and recommends components for agent generation"""
    
    def __init__(self):
        """Initialize the component advisor agent"""
        super().__init__(
            model='openai:gpt-4',
            deps_type=ComponentAdvisorDependencies,
            result_type=ComponentAdvisorOutput,
            enable_logfire=True
        )
        
        # Initialize knowledge bases
        self.component_knowledge = self._load_component_knowledge()
        self.pattern_knowledge = self._load_pattern_knowledge()
        self.model_knowledge = self._load_model_knowledge()
    
    def get_system_prompt(self) -> str:
        """Get the system prompt for component advisor agent"""
        return """You are an expert AI component advisor that helps developers build agents efficiently.
        
        Your responsibilities:
        1. Analyze task requirements thoroughly
        2. Recommend relevant components from the library
        3. Suggest optimal AI models for the task
        4. Recommend architecture patterns
        5. Identify potential challenges
        6. Provide actionable implementation guidance
        
        Guidelines:
        - Prioritize reusability and maintainability
        - Consider performance and cost trade-offs
        - Recommend battle-tested components when available
        - Suggest patterns that match the task complexity
        - Be specific and actionable in recommendations
        
        Always provide clear reasoning for your recommendations."""
    
    def _register_tools(self):
        """Register tools for the component advisor agent"""
        self.agent.tool(self.analyze_task_requirements)
        self.agent.tool(self.search_components)
        self.agent.tool(self.evaluate_models)
        self.agent.tool(self.recommend_patterns)
        self.agent.tool(self.assess_complexity)
        self.agent.tool(self.generate_implementation_plan)
    
    def _load_component_knowledge(self) -> Dict[str, Any]:
        """Load knowledge about available components"""
        # Extended component library based on Multi-Agent-Channel's capabilities
        return {
            "tools": {
                "github_tool": {
                    "description": "Interact with GitHub API",
                    "use_cases": ["repo management", "issue tracking", "PR automation", "code analysis"],
                    "dependencies": ["PyGithub", "httpx"]
                },
                "web_search_tool": {
                    "description": "Search the web for information",
                    "use_cases": ["research", "fact checking", "current events", "documentation lookup"],
                    "dependencies": ["duckduckgo-search", "beautifulsoup4"]
                },
                "file_tool": {
                    "description": "Read and write files",
                    "use_cases": ["data processing", "code generation", "logging", "configuration"],
                    "dependencies": ["aiofiles"]
                },
                "database_tool": {
                    "description": "Interact with databases",
                    "use_cases": ["data storage", "querying", "analytics", "persistence"],
                    "dependencies": ["sqlalchemy", "asyncpg"]
                },
                "validation_tool": {
                    "description": "Validate code and data",
                    "use_cases": ["code validation", "data validation", "testing", "quality assurance"],
                    "dependencies": ["pytest", "mypy", "ruff"]
                },
                "mcp_adapter": {
                    "description": "Integrate MCP servers",
                    "use_cases": ["external tools", "service integration", "protocol adaptation"],
                    "dependencies": ["httpx", "websockets"]
                },
                "context_tool": {
                    "description": "Manage project context",
                    "use_cases": ["context loading", "memory management", "state persistence"],
                    "dependencies": ["pyyaml", "json"]
                }
            },
            "patterns": {
                "react_pattern": {
                    "description": "Reasoning and Acting pattern",
                    "use_cases": ["complex reasoning", "multi-step tasks", "research", "analysis"],
                    "benefits": ["transparent reasoning", "better accuracy", "step-by-step thinking"]
                },
                "rag_pattern": {
                    "description": "Retrieval Augmented Generation",
                    "use_cases": ["knowledge-based tasks", "documentation", "Q&A", "search"],
                    "benefits": ["accurate information", "reduced hallucination", "context awareness"]
                },
                "chain_pattern": {
                    "description": "Chain of thought reasoning",
                    "use_cases": ["math problems", "logical reasoning", "analysis", "planning"],
                    "benefits": ["step-by-step reasoning", "verifiable logic", "explainability"]
                },
                "parallel_pattern": {
                    "description": "Parallel agent execution",
                    "use_cases": ["multi-domain tasks", "performance optimization", "independent subtasks"],
                    "benefits": ["faster execution", "scalability", "resource efficiency"]
                }
            },
            "examples": {
                "github_monitor": {
                    "description": "GitHub repository monitor",
                    "components": ["github_tool", "notification_tool", "scheduler"],
                    "pattern": "monitoring_pattern"
                },
                "code_reviewer": {
                    "description": "Automated code reviewer",
                    "components": ["file_tool", "validation_tool", "github_tool"],
                    "pattern": "analysis_pattern"
                },
                "research_assistant": {
                    "description": "Research and analysis assistant",
                    "components": ["web_search_tool", "rag_pattern", "synthesis_tool"],
                    "pattern": "research_pattern"
                }
            }
        }
    
    def _load_pattern_knowledge(self) -> Dict[str, Any]:
        """Load knowledge about architecture patterns"""
        return {
            "single_agent": {
                "description": "Single agent handles all tasks",
                "when_to_use": ["simple tasks", "low complexity", "single domain", "quick prototypes"],
                "benefits": ["simple implementation", "fast execution", "easy debugging"],
                "drawbacks": ["limited scalability", "single point of failure", "no specialization"]
            },
            "multi_agent": {
                "description": "Multiple specialized agents collaborate",
                "when_to_use": ["complex tasks", "multiple domains", "parallel work", "production systems"],
                "benefits": ["scalability", "specialization", "fault tolerance", "modularity"],
                "drawbacks": ["coordination overhead", "complexity", "resource usage"]
            },
            "hierarchical": {
                "description": "Agents organized in hierarchy",
                "when_to_use": ["large projects", "team simulation", "delegation", "complex workflows"],
                "benefits": ["clear responsibility", "scalable", "manageable", "delegation"],
                "drawbacks": ["communication overhead", "potential bottlenecks", "complexity"]
            },
            "reactive": {
                "description": "Event-driven agent responses",
                "when_to_use": ["monitoring", "real-time systems", "webhooks", "notifications"],
                "benefits": ["efficient", "responsive", "scalable", "event-driven"],
                "drawbacks": ["state management", "debugging difficulty", "ordering issues"]
            }
        }
    
    def _load_model_knowledge(self) -> Dict[str, Any]:
        """Load knowledge about AI models"""
        # Extended with Multi-Agent-Channel's 100+ model support
        return {
            "openai:gpt-4": {
                "strengths": ["reasoning", "code generation", "analysis", "creativity"],
                "weaknesses": ["cost", "speed", "rate limits"],
                "cost_per_1k": 0.03,
                "use_cases": ["complex tasks", "high accuracy needs", "code generation"]
            },
            "openai:gpt-3.5-turbo": {
                "strengths": ["speed", "cost-effective", "general tasks", "good availability"],
                "weaknesses": ["complex reasoning", "latest knowledge", "nuanced tasks"],
                "cost_per_1k": 0.002,
                "use_cases": ["simple tasks", "high volume", "prototyping", "chat"]
            },
            "anthropic:claude-3-opus": {
                "strengths": ["long context", "analysis", "safety", "nuanced understanding"],
                "weaknesses": ["availability", "cost", "speed"],
                "cost_per_1k": 0.075,
                "use_cases": ["document analysis", "research", "safety-critical", "complex reasoning"]
            },
            "anthropic:claude-3-sonnet": {
                "strengths": ["balance", "efficiency", "reasoning", "context"],
                "weaknesses": ["specialized tasks", "real-time needs"],
                "cost_per_1k": 0.015,
                "use_cases": ["general purpose", "production", "cost-conscious", "analysis"]
            },
            "meta:llama-3-70b": {
                "strengths": ["open source", "customizable", "privacy", "on-premise"],
                "weaknesses": ["infrastructure", "fine-tuning needed", "support"],
                "cost_per_1k": 0.001,
                "use_cases": ["privacy-sensitive", "custom domains", "high volume", "research"]
            },
            "google:gemini-pro": {
                "strengths": ["multimodal", "fast", "Google integration", "large context"],
                "weaknesses": ["consistency", "availability", "documentation"],
                "cost_per_1k": 0.005,
                "use_cases": ["multimodal tasks", "Google ecosystem", "visual analysis", "speed"]
            }
        }
    
    async def analyze_task_requirements(
        self,
        ctx: RunContext[ComponentAdvisorDependencies],
        task_description: str,
        requirements: List[str]
    ) -> Dict[str, Any]:
        """Analyze task requirements in detail
        
        Args:
            ctx: Run context
            task_description: Description of the task
            requirements: List of specific requirements
            
        Returns:
            Detailed task analysis
        """
        logfire.info("analyzing_task_requirements", task=task_description)
        
        analysis = {
            "task_type": self._classify_task(task_description),
            "domains": self._identify_domains(task_description),
            "complexity_indicators": self._find_complexity_indicators(task_description),
            "required_capabilities": self._extract_capabilities(requirements),
            "performance_requirements": self._extract_performance_requirements(requirements),
            "integration_points": self._identify_integrations(task_description, requirements)
        }
        
        return analysis
    
    def _classify_task(self, task: str) -> str:
        """Classify the type of task"""
        task_lower = task.lower()
        
        if any(word in task_lower for word in ["monitor", "track", "watch", "observe"]):
            return "monitoring"
        elif any(word in task_lower for word in ["analyze", "research", "investigate", "examine"]):
            return "analysis"
        elif any(word in task_lower for word in ["generate", "create", "build", "produce"]):
            return "generation"
        elif any(word in task_lower for word in ["automate", "process", "workflow", "pipeline"]):
            return "automation"
        elif any(word in task_lower for word in ["test", "validate", "verify", "check"]):
            return "validation"
        elif any(word in task_lower for word in ["integrate", "connect", "bridge", "adapter"]):
            return "integration"
        else:
            return "general"
    
    def _identify_domains(self, task: str) -> List[str]:
        """Identify domains involved in the task"""
        domains = []
        task_lower = task.lower()
        
        domain_keywords = {
            "code": ["code", "programming", "development", "github", "git", "repository"],
            "data": ["data", "database", "analytics", "processing", "etl", "storage"],
            "web": ["web", "internet", "search", "scraping", "api", "http"],
            "security": ["security", "vulnerability", "authentication", "encryption", "auth"],
            "devops": ["deploy", "ci/cd", "docker", "kubernetes", "infrastructure"],
            "ml": ["machine learning", "ml", "ai", "model", "training", "prediction"],
            "monitoring": ["monitor", "alert", "notification", "tracking", "metrics"],
            "documentation": ["document", "docs", "readme", "api docs", "knowledge base"]
        }
        
        for domain, keywords in domain_keywords.items():
            if any(keyword in task_lower for keyword in keywords):
                domains.append(domain)
        
        return domains or ["general"]
    
    def _find_complexity_indicators(self, task: str) -> List[str]:
        """Find indicators of task complexity"""
        indicators = []
        task_lower = task.lower()
        
        complexity_patterns = {
            "multiple_components": ["multiple", "various", "several", "different", "diverse"],
            "integration_required": ["integrate", "combine", "coordinate", "connect", "bridge"],
            "real_time_processing": ["real-time", "live", "streaming", "instant", "immediate"],
            "scalability_needed": ["scale", "high-volume", "performance", "concurrent", "parallel"],
            "state_management": ["state", "persist", "maintain", "track", "history"],
            "error_handling": ["robust", "fault-tolerant", "resilient", "recovery", "retry"],
            "security_concerns": ["secure", "encrypt", "authenticate", "authorize", "protect"]
        }
        
        for indicator, keywords in complexity_patterns.items():
            if any(keyword in task_lower for keyword in keywords):
                indicators.append(indicator)
        
        return indicators
    
    def _extract_capabilities(self, requirements: List[str]) -> List[str]:
        """Extract required capabilities from requirements"""
        capabilities = set()
        
        capability_patterns = {
            "api_integration": ["api", "endpoint", "rest", "graphql", "webhook"],
            "data_handling": ["database", "data", "storage", "query", "persist"],
            "authentication": ["auth", "login", "user", "permission", "access"],
            "testing": ["test", "validate", "verify", "check", "assert"],
            "monitoring": ["monitor", "track", "alert", "notify", "watch"],
            "scheduling": ["schedule", "cron", "periodic", "timer", "interval"],
            "caching": ["cache", "memory", "redis", "performance", "speed"],
            "logging": ["log", "audit", "trace", "debug", "record"]
        }
        
        for req in requirements:
            req_lower = req.lower()
            for capability, keywords in capability_patterns.items():
                if any(keyword in req_lower for keyword in keywords):
                    capabilities.add(capability)
        
        return list(capabilities)
    
    def _extract_performance_requirements(self, requirements: List[str]) -> Dict[str, Any]:
        """Extract performance requirements"""
        perf_reqs = {}
        
        for req in requirements:
            req_lower = req.lower()
            
            # Latency requirements
            if any(word in req_lower for word in ["latency", "speed", "fast", "quick"]):
                perf_reqs["low_latency"] = True
                if "ms" in req_lower or "millisecond" in req_lower:
                    perf_reqs["target_latency_ms"] = 100  # Default target
            
            # Throughput requirements  
            if any(word in req_lower for word in ["throughput", "requests per", "rps", "qps"]):
                perf_reqs["high_throughput"] = True
            
            # Concurrency requirements
            if any(word in req_lower for word in ["concurrent", "parallel", "simultaneous"]):
                perf_reqs["high_concurrency"] = True
            
            # Reliability requirements
            if any(word in req_lower for word in ["reliable", "availability", "uptime", "sla"]):
                perf_reqs["high_availability"] = True
                if "99" in req_lower:
                    perf_reqs["target_availability"] = 0.99
        
        return perf_reqs
    
    def _identify_integrations(self, task: str, requirements: List[str]) -> List[str]:
        """Identify required integrations"""
        integrations = set()
        combined_text = f"{task} {' '.join(requirements)}".lower()
        
        integration_patterns = {
            "github": ["github", "gh", "repository", "pull request", "issue"],
            "slack": ["slack", "channel", "notification", "message"],
            "database": ["postgres", "mysql", "mongodb", "redis", "database"],
            "cloud": ["aws", "azure", "gcp", "cloud", "s3", "lambda"],
            "monitoring": ["prometheus", "grafana", "datadog", "newrelic"],
            "ci_cd": ["jenkins", "github actions", "gitlab", "circleci"],
            "api": ["rest", "graphql", "webhook", "http", "api"]
        }
        
        for integration, keywords in integration_patterns.items():
            if any(keyword in combined_text for keyword in keywords):
                integrations.add(integration)
        
        return list(integrations)
    
    async def search_components(
        self,
        ctx: RunContext[ComponentAdvisorDependencies],
        task_analysis: Dict[str, Any]
    ) -> List[ComponentRecommendation]:
        """Search for relevant components based on task analysis
        
        Args:
            ctx: Run context
            task_analysis: Analysis of the task
            
        Returns:
            List of recommended components
        """
        logfire.info("searching_components")
        
        recommendations = []
        
        # Search tools
        for tool_name, tool_info in self.component_knowledge["tools"].items():
            relevance = self._calculate_relevance(
                task_analysis,
                tool_info["use_cases"]
            )
            
            if relevance > 0.3:  # Threshold for recommendation
                recommendations.append(ComponentRecommendation(
                    name=tool_name,
                    type="tool",
                    relevance_score=relevance,
                    reason=f"Useful for {', '.join(tool_info['use_cases'][:2])}",
                    dependencies=tool_info["dependencies"],
                    usage_example=self._generate_usage_example(tool_name, task_analysis)
                ))
        
        # Search patterns
        for pattern_name, pattern_info in self.component_knowledge["patterns"].items():
            relevance = self._calculate_relevance(
                task_analysis,
                pattern_info["use_cases"]
            )
            
            if relevance > 0.3:
                recommendations.append(ComponentRecommendation(
                    name=pattern_name,
                    type="pattern",
                    relevance_score=relevance,
                    reason=f"Benefits: {', '.join(pattern_info['benefits'][:2])}",
                    dependencies=[]
                ))
        
        # Search examples
        task_type = task_analysis.get("task_type", "")
        for example_name, example_info in self.component_knowledge.get("examples", {}).items():
            if any(domain in task_analysis.get("domains", []) for domain in example_info.get("components", [])):
                recommendations.append(ComponentRecommendation(
                    name=example_name,
                    type="example",
                    relevance_score=0.5,
                    reason=f"Similar implementation: {example_info['description']}",
                    dependencies=[]
                ))
        
        # Sort by relevance
        recommendations.sort(key=lambda x: x.relevance_score, reverse=True)
        
        return recommendations[:8]  # Top 8 recommendations
    
    def _calculate_relevance(
        self,
        task_analysis: Dict[str, Any],
        component_use_cases: List[str]
    ) -> float:
        """Calculate relevance score for a component"""
        score = 0.0
        
        # Check task type match
        task_type = task_analysis.get("task_type", "")
        for use_case in component_use_cases:
            if task_type in use_case or use_case in task_type:
                score += 0.3
        
        # Check domain match
        domains = task_analysis.get("domains", [])
        for domain in domains:
            for use_case in component_use_cases:
                if domain in use_case or use_case in domain:
                    score += 0.2
        
        # Check capability match
        capabilities = task_analysis.get("required_capabilities", [])
        for capability in capabilities:
            for use_case in component_use_cases:
                if capability in use_case or use_case in capability:
                    score += 0.15
        
        # Check integration match
        integrations = task_analysis.get("integration_points", [])
        for integration in integrations:
            for use_case in component_use_cases:
                if integration in use_case:
                    score += 0.1
        
        return min(score, 1.0)  # Cap at 1.0
    
    def _generate_usage_example(self, tool_name: str, task_analysis: Dict[str, Any]) -> str:
        """Generate a usage example for a tool"""
        examples = {
            "github_tool": """
# Example usage
from tools import GitHubTool

github = GitHubTool(token="your_token")
repos = await github.list_repos("organization")
issues = await github.get_issues("owner/repo", state="open")
""",
            "web_search_tool": """
# Example usage
from tools import WebSearchTool

search = WebSearchTool()
results = await search.search("latest AI developments")
for result in results:
    print(f"{result.title}: {result.url}")
""",
            "validation_tool": """
# Example usage
from tools import ValidationTool

validator = ValidationTool()
results = await validator.validate_code("agent.py")
if results.passed:
    print("Code validation passed!")
"""
        }
        
        return examples.get(tool_name, f"# {tool_name} usage example pending")
    
    async def evaluate_models(
        self,
        ctx: RunContext[ComponentAdvisorDependencies],
        task_analysis: Dict[str, Any],
        performance_requirements: Dict[str, Any]
    ) -> List[ModelRecommendation]:
        """Evaluate and recommend AI models
        
        Args:
            ctx: Run context
            task_analysis: Task analysis
            performance_requirements: Performance requirements
            
        Returns:
            List of model recommendations
        """
        logfire.info("evaluating_models")
        
        recommendations = []
        
        for model_id, model_info in self.model_knowledge.items():
            score = 0.0
            reasons = []
            
            # Evaluate based on task type
            task_type = task_analysis.get("task_type", "")
            if task_type == "analysis" and "analysis" in model_info["strengths"]:
                score += 0.3
                reasons.append("Excellent for analysis tasks")
            elif task_type == "generation" and "code generation" in model_info["strengths"]:
                score += 0.3
                reasons.append("Strong code generation capabilities")
            elif task_type == "monitoring" and "speed" in model_info["strengths"]:
                score += 0.3
                reasons.append("Fast response for real-time monitoring")
            
            # Evaluate based on complexity
            complexity_indicators = task_analysis.get("complexity_indicators", [])
            if len(complexity_indicators) > 3:
                # Complex task - prefer more capable models
                if "reasoning" in model_info["strengths"]:
                    score += 0.3
                    reasons.append("Handles complex reasoning well")
            else:
                # Simple task - consider cost-effectiveness
                if model_info.get("cost_per_1k", 1) < 0.01:
                    score += 0.2
                    reasons.append("Cost-effective for simple tasks")
            
            # Evaluate based on performance requirements
            if performance_requirements.get("low_latency"):
                if "speed" in model_info["strengths"]:
                    score += 0.3
                    reasons.append("Low latency performance")
                elif "speed" in model_info.get("weaknesses", []):
                    score -= 0.2
            
            if performance_requirements.get("high_availability"):
                if "availability" not in model_info.get("weaknesses", []):
                    score += 0.1
                    reasons.append("Good availability")
            
            # Context window considerations
            if "state_management" in complexity_indicators:
                if "long context" in model_info["strengths"]:
                    score += 0.2
                    reasons.append("Large context window for state management")
            
            if score > 0:
                recommendations.append(ModelRecommendation(
                    model_id=model_id,
                    reason="; ".join(reasons),
                    estimated_cost=model_info.get("cost_per_1k", 0),
                    performance_score=score,
                    alternatives=self._find_model_alternatives(model_id)
                ))
        
        # Sort by performance score
        recommendations.sort(key=lambda x: x.performance_score, reverse=True)
        
        return recommendations[:4]  # Top 4 models
    
    def _find_model_alternatives(self, model_id: str) -> List[str]:
        """Find alternative models"""
        alternatives = []
        
        # Group models by capability tier
        high_tier = ["openai:gpt-4", "anthropic:claude-3-opus"]
        mid_tier = ["anthropic:claude-3-sonnet", "google:gemini-pro"]
        low_tier = ["openai:gpt-3.5-turbo", "meta:llama-3-70b"]
        
        if model_id in high_tier:
            alternatives = [m for m in high_tier if m != model_id] + mid_tier[:1]
        elif model_id in mid_tier:
            alternatives = [m for m in mid_tier if m != model_id] + [high_tier[0], low_tier[0]]
        else:
            alternatives = [m for m in low_tier if m != model_id] + mid_tier[:1]
        
        return alternatives[:3]
    
    async def recommend_patterns(
        self,
        ctx: RunContext[ComponentAdvisorDependencies],
        task_analysis: Dict[str, Any],
        recommended_components: List[ComponentRecommendation]
    ) -> List[ArchitectureRecommendation]:
        """Recommend architecture patterns
        
        Args:
            ctx: Run context
            task_analysis: Task analysis
            recommended_components: Already recommended components
            
        Returns:
            Architecture recommendations
        """
        logfire.info("recommending_patterns")
        
        recommendations = []
        complexity_indicators = task_analysis.get("complexity_indicators", [])
        domains = task_analysis.get("domains", [])
        
        # Analyze pattern suitability
        for pattern_name, pattern_info in self.pattern_knowledge.items():
            suitability_score = 0
            
            # Single agent pattern
            if pattern_name == "single_agent":
                if len(complexity_indicators) <= 2 and len(domains) <= 1:
                    suitability_score = 0.8
                else:
                    suitability_score = 0.3
            
            # Multi-agent pattern
            elif pattern_name == "multi_agent":
                if len(complexity_indicators) >= 2 or len(domains) >= 2:
                    suitability_score = 0.9
                else:
                    suitability_score = 0.4
            
            # Hierarchical pattern
            elif pattern_name == "hierarchical":
                if "integration_required" in complexity_indicators or len(domains) >= 3:
                    suitability_score = 0.7
                else:
                    suitability_score = 0.3
            
            # Reactive pattern
            elif pattern_name == "reactive":
                if task_analysis.get("task_type") == "monitoring" or "real_time_processing" in complexity_indicators:
                    suitability_score = 0.9
                else:
                    suitability_score = 0.2
            
            if suitability_score > 0.4:
                recommendations.append(ArchitectureRecommendation(
                    pattern_name=pattern_name,
                    description=pattern_info["description"],
                    benefits=pattern_info["benefits"],
                    considerations=pattern_info["drawbacks"],
                    example_implementation=self._generate_pattern_example(pattern_name)
                ))
        
        # Sort by suitability
        recommendations.sort(key=lambda x: x.pattern_name == "multi_agent", reverse=True)
        
        return recommendations[:3]
    
    def _generate_pattern_example(self, pattern_name: str) -> str:
        """Generate example implementation for a pattern"""
        examples = {
            "single_agent": """
# Single agent pattern
agent = TaskAgent(model="gpt-4")
result = await agent.run(user_input)
""",
            "multi_agent": """
# Multi-agent pattern with orchestrator
orchestrator = LangGraphOrchestrator()
result = await orchestrator.run(user_input, thread_id="task_123")
""",
            "hierarchical": """
# Hierarchical pattern
manager = ManagerAgent()
workers = [WorkerAgent(specialty) for specialty in ["analysis", "generation"]]
result = await manager.delegate(task, workers)
""",
            "reactive": """
# Reactive pattern
monitor = ReactiveAgent()
monitor.on_event("repository_update", handle_update)
await monitor.start()
"""
        }
        
        return examples.get(pattern_name, "# Pattern implementation example")
    
    async def assess_complexity(
        self,
        ctx: RunContext[ComponentAdvisorDependencies],
        task_analysis: Dict[str, Any]
    ) -> str:
        """Assess overall task complexity
        
        Args:
            ctx: Run context  
            task_analysis: Task analysis
            
        Returns:
            Complexity assessment
        """
        complexity_score = 0
        
        # Factor in complexity indicators (weighted heavily)
        indicators = task_analysis.get("complexity_indicators", [])
        complexity_score += len(indicators) * 0.25
        
        # Factor in number of domains
        domains = task_analysis.get("domains", [])
        complexity_score += len(domains) * 0.15
        
        # Factor in capabilities
        capabilities = task_analysis.get("required_capabilities", [])
        complexity_score += len(capabilities) * 0.1
        
        # Factor in integrations
        integrations = task_analysis.get("integration_points", [])
        complexity_score += len(integrations) * 0.15
        
        # Factor in performance requirements
        perf_reqs = task_analysis.get("performance_requirements", {})
        complexity_score += len(perf_reqs) * 0.1
        
        # Determine complexity level
        if complexity_score < 0.4:
            return "simple"
        elif complexity_score < 0.8:
            return "moderate"
        else:
            return "complex"
    
    async def generate_implementation_plan(
        self,
        ctx: RunContext[ComponentAdvisorDependencies],
        task_description: str,
        recommendations: Dict[str, Any]
    ) -> List[str]:
        """Generate step-by-step implementation plan
        
        Args:
            ctx: Run context
            task_description: Task description
            recommendations: All recommendations
            
        Returns:
            Implementation steps
        """
        components = recommendations.get('components', [])
        models = recommendations.get('models', [])
        patterns = recommendations.get('patterns', [])
        
        steps = [
            "1. Set up project structure using the Multi-Agent-Channel framework",
            "2. Install dependencies: `pip install -r requirements.txt`"
        ]
        
        # Add pattern-specific setup
        if patterns and patterns[0].pattern_name == "multi_agent":
            steps.append("3. Configure LangGraph orchestrator for multi-agent coordination")
        else:
            steps.append("3. Create main agent class inheriting from BaseAgent")
        
        # Add component integration steps
        if components:
            tool_names = [c.name for c in components if c.type == "tool"][:3]
            steps.append(f"4. Integrate recommended tools: {', '.join(tool_names)}")
        
        # Add model configuration
        if models:
            steps.append(f"5. Configure primary model: {models[0].model_id}")
        
        # Add pattern implementation
        if patterns:
            steps.append(f"6. Implement {patterns[0].pattern_name} architecture pattern")
        
        # Standard steps
        steps.extend([
            "7. Add error handling and retry logic",
            "8. Implement logging and monitoring",
            "9. Write unit and integration tests",
            "10. Create comprehensive documentation",
            "11. Set up CI/CD pipeline",
            "12. Deploy and monitor performance"
        ])
        
        return steps
    
    async def identify_challenges(
        self,
        ctx: RunContext[ComponentAdvisorDependencies],
        task_analysis: Dict[str, Any]
    ) -> List[str]:
        """Identify potential challenges
        
        Args:
            ctx: Run context
            task_analysis: Task analysis
            
        Returns:
            List of potential challenges
        """
        challenges = []
        
        # Complexity-based challenges
        complexity_indicators = task_analysis.get("complexity_indicators", [])
        if "state_management" in complexity_indicators:
            challenges.append("Managing state across agent interactions")
        if "real_time_processing" in complexity_indicators:
            challenges.append("Ensuring low-latency responses")
        if "scalability_needed" in complexity_indicators:
            challenges.append("Handling high concurrency and load")
        
        # Integration challenges
        integrations = task_analysis.get("integration_points", [])
        if len(integrations) > 2:
            challenges.append("Coordinating multiple external integrations")
        if "github" in integrations:
            challenges.append("Handling GitHub API rate limits")
        
        # Domain-specific challenges
        domains = task_analysis.get("domains", [])
        if "security" in domains:
            challenges.append("Implementing robust security measures")
        if "ml" in domains:
            challenges.append("Managing model versioning and updates")
        
        # Performance challenges
        perf_reqs = task_analysis.get("performance_requirements", {})
        if perf_reqs.get("low_latency") and perf_reqs.get("high_throughput"):
            challenges.append("Balancing latency and throughput requirements")
        
        return challenges or ["Standard implementation complexity"]


# Example usage
async def example_component_advisor():
    """Example of using the component advisor agent"""
    agent = ComponentAdvisorAgent()
    
    deps = ComponentAdvisorDependencies(
        user_id="demo",
        session_id="component_advisor_demo"
    )
    
    result = await agent.run(
        """I need to build an agent that:
        1. Monitors multiple GitHub repositories for security vulnerabilities
        2. Analyzes code changes in real-time
        3. Generates security reports
        4. Sends notifications via Slack
        5. Maintains a history of vulnerabilities found
        Requirements: Low latency (<500ms), High availability (99.9%), Scalable to 1000+ repos
        """,
        deps
    )
    
    print("\n=== COMPONENT ADVISOR RECOMMENDATIONS ===\n")
    
    print("📋 Task Analysis:", result.task_analysis)
    print(f"📊 Complexity: {result.estimated_complexity}")
    
    print("\n🔧 Recommended Components:")
    for comp in result.recommended_components:
        print(f"  - {comp.name} ({comp.type})")
        print(f"    Relevance: {comp.relevance_score:.2f}")
        print(f"    Reason: {comp.reason}")
        if comp.usage_example:
            print(f"    Example: {comp.usage_example[:100]}...")
    
    print("\n🤖 Recommended Models:")
    for model in result.recommended_models:
        print(f"  - {model.model_id}")
        print(f"    Score: {model.performance_score:.2f}")
        print(f"    Reason: {model.reason}")
        print(f"    Est. Cost: ${model.estimated_cost:.3f}/1k tokens")
    
    print("\n🏗️  Architecture Patterns:")
    for pattern in result.architecture_patterns:
        print(f"  - {pattern.pattern_name}: {pattern.description}")
        print(f"    Benefits: {', '.join(pattern.benefits[:2])}")
    
    print("\n📝 Implementation Steps:")
    for i, step in enumerate(result.implementation_steps, 1):
        print(f"  {step}")
    
    print("\n⚠️  Potential Challenges:")
    for challenge in result.potential_challenges:
        print(f"  - {challenge}")


if __name__ == "__main__":
    import asyncio
    asyncio.run(example_component_advisor())