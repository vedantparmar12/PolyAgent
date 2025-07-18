"""Intelligent task routing for workflow management"""

from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
from pydantic_ai import Agent
from ..agents.models import TaskComplexity
import re


class RoutingDecision(BaseModel):
    """Routing decision output"""
    complexity: TaskComplexity = Field(description="Assessed task complexity")
    reasoning: str = Field(description="Reasoning for the routing decision")
    required_capabilities: List[str] = Field(description="Required agent capabilities")
    estimated_tokens: int = Field(description="Estimated tokens for completion")
    recommended_agents: List[str] = Field(description="Recommended agents for the task")


class TaskRouter:
    """Routes tasks to appropriate agents based on complexity and requirements"""
    
    def __init__(self, model: str = "openai:gpt-4"):
        """Initialize the task router
        
        Args:
            model: The AI model to use for routing decisions
        """
        self.model = model
        self._init_routing_agent()
        self._init_patterns()
    
    def _init_routing_agent(self):
        """Initialize the routing agent"""
        self.routing_agent = Agent(
            model=self.model,
            result_type=RoutingDecision,
            system_prompt=self._get_routing_prompt()
        )
    
    def _get_routing_prompt(self) -> str:
        """Get the system prompt for routing"""
        return """You are an intelligent task router that analyzes user requests and determines:
        
        1. Task Complexity:
           - SIMPLE: Single-step tasks, direct questions, basic operations
           - COMPLEX: Multi-step tasks, requires coordination, involves multiple files
           - RESEARCH: Requires extensive information gathering, analysis, or exploration
        
        2. Required Capabilities:
           - coding: Code generation, debugging, refactoring
           - research: Information gathering, web search, documentation lookup
           - analysis: Data analysis, pattern recognition, insights
           - reasoning: Complex logic, planning, decision making
           - validation: Testing, linting, type checking
        
        3. Agent Recommendations:
           - advisor: For context and examples
           - scoper: For task breakdown and planning
           - coder: For implementation
           - refiner: For code improvement
           - researcher: For information gathering
        
        Analyze the user's request carefully and provide accurate routing decisions."""
    
    def _init_patterns(self):
        """Initialize patterns for quick routing"""
        self.simple_patterns = [
            r"^(what|how|when|where|why|who)\s+",
            r"^(explain|describe|define)\s+",
            r"^(show|list|display)\s+",
            r"^(fix|correct|update)\s+\w+\s+(in|on)\s+",
            r"^(add|remove|change)\s+\w+\s+",
        ]
        
        self.complex_patterns = [
            r"(implement|create|build|develop)\s+.*(system|application|feature)",
            r"(refactor|redesign|optimize)\s+",
            r"(integrate|connect|combine)\s+",
            r"(debug|troubleshoot|investigate)\s+",
            r"(test|validate|verify)\s+.*(thoroughly|comprehensive)",
        ]
        
        self.research_patterns = [
            r"(research|investigate|explore|analyze)\s+",
            r"(compare|evaluate|assess)\s+",
            r"(find|search|look for)\s+.*(information|data|examples)",
            r"(gather|collect|compile)\s+",
            r"(study|examine|review)\s+",
        ]
    
    async def assess_complexity(self, user_input: str) -> TaskComplexity:
        """Assess the complexity of a task
        
        Args:
            user_input: The user's input/query
            
        Returns:
            The assessed task complexity
        """
        # Quick pattern matching first
        complexity = self._quick_assess(user_input)
        
        if complexity:
            return complexity
        
        # Use AI for more nuanced assessment
        result = await self.routing_agent.run(
            f"Analyze this task and determine its complexity: {user_input}"
        )
        
        return result.data.complexity
    
    def _quick_assess(self, user_input: str) -> Optional[TaskComplexity]:
        """Quick assessment using pattern matching
        
        Args:
            user_input: The user's input
            
        Returns:
            Task complexity if pattern matched, None otherwise
        """
        lower_input = user_input.lower()
        
        # Check simple patterns
        for pattern in self.simple_patterns:
            if re.match(pattern, lower_input):
                return TaskComplexity.SIMPLE
        
        # Check research patterns
        for pattern in self.research_patterns:
            if re.search(pattern, lower_input):
                return TaskComplexity.RESEARCH
        
        # Check complex patterns
        for pattern in self.complex_patterns:
            if re.search(pattern, lower_input):
                return TaskComplexity.COMPLEX
        
        return None
    
    async def get_routing_decision(self, user_input: str) -> RoutingDecision:
        """Get complete routing decision
        
        Args:
            user_input: The user's input/query
            
        Returns:
            Complete routing decision with recommendations
        """
        result = await self.routing_agent.run(
            f"Analyze this task and provide routing decision: {user_input}"
        )
        
        return result.data
    
    def estimate_token_usage(self, user_input: str, complexity: TaskComplexity) -> int:
        """Estimate token usage for a task
        
        Args:
            user_input: The user's input
            complexity: The task complexity
            
        Returns:
            Estimated token usage
        """
        base_tokens = len(user_input.split()) * 2  # Rough estimate
        
        multipliers = {
            TaskComplexity.SIMPLE: 10,
            TaskComplexity.COMPLEX: 50,
            TaskComplexity.RESEARCH: 100
        }
        
        return base_tokens * multipliers.get(complexity, 20)
    
    def get_recommended_agents(self, complexity: TaskComplexity, capabilities: List[str]) -> List[str]:
        """Get recommended agents based on complexity and capabilities
        
        Args:
            complexity: Task complexity
            capabilities: Required capabilities
            
        Returns:
            List of recommended agent names
        """
        recommendations = []
        
        # Base recommendations by complexity
        if complexity == TaskComplexity.SIMPLE:
            recommendations.append("coder")
        elif complexity == TaskComplexity.COMPLEX:
            recommendations.extend(["advisor", "scoper", "coder", "refiner"])
        elif complexity == TaskComplexity.RESEARCH:
            recommendations.extend(["researcher", "advisor", "analyzer"])
        
        # Add based on capabilities
        capability_agents = {
            "coding": ["coder", "refiner"],
            "research": ["researcher", "advisor"],
            "analysis": ["analyzer", "scoper"],
            "reasoning": ["scoper", "advisor"],
            "validation": ["validator", "refiner"]
        }
        
        for capability in capabilities:
            if capability in capability_agents:
                recommendations.extend(capability_agents[capability])
        
        # Remove duplicates while preserving order
        seen = set()
        unique_recommendations = []
        for agent in recommendations:
            if agent not in seen:
                seen.add(agent)
                unique_recommendations.append(agent)
        
        return unique_recommendations
    
    def should_use_streaming(self, complexity: TaskComplexity) -> bool:
        """Determine if streaming should be used
        
        Args:
            complexity: Task complexity
            
        Returns:
            Whether to use streaming
        """
        # Use streaming for complex and research tasks
        return complexity in [TaskComplexity.COMPLEX, TaskComplexity.RESEARCH]


class ConversationContextAnalyzer:
    """Analyzes conversation context for better routing"""
    
    def __init__(self):
        self.context_window = 5  # Number of previous messages to consider
    
    def analyze_context(
        self,
        current_input: str,
        conversation_history: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analyze conversation context
        
        Args:
            current_input: Current user input
            conversation_history: Previous conversation messages
            
        Returns:
            Context analysis results
        """
        recent_history = conversation_history[-self.context_window:] if conversation_history else []
        
        # Analyze patterns in conversation
        is_followup = self._is_followup_question(current_input, recent_history)
        topic_shift = self._detect_topic_shift(current_input, recent_history)
        requires_context = self._requires_previous_context(current_input)
        
        # Extract relevant context
        relevant_context = self._extract_relevant_context(current_input, recent_history)
        
        return {
            "is_followup": is_followup,
            "topic_shift": topic_shift,
            "requires_context": requires_context,
            "relevant_context": relevant_context,
            "conversation_depth": len(conversation_history)
        }
    
    def _is_followup_question(
        self,
        current_input: str,
        recent_history: List[Dict[str, Any]]
    ) -> bool:
        """Check if current input is a follow-up question"""
        followup_indicators = [
            "it", "this", "that", "these", "those",
            "the same", "also", "additionally", "furthermore",
            "what about", "how about", "and"
        ]
        
        lower_input = current_input.lower()
        
        # Check for pronouns and references
        for indicator in followup_indicators:
            if indicator in lower_input:
                return True
        
        # Check if it starts with a continuation
        if lower_input.startswith(("and ", "but ", "also ", "what about ", "how about ")):
            return True
        
        return False
    
    def _detect_topic_shift(
        self,
        current_input: str,
        recent_history: List[Dict[str, Any]]
    ) -> bool:
        """Detect if there's a topic shift"""
        if not recent_history:
            return False
        
        # Simple heuristic: Check for completely different keywords
        current_keywords = set(current_input.lower().split())
        
        for msg in recent_history:
            if msg.get("role") == "user":
                prev_keywords = set(msg.get("content", "").lower().split())
                overlap = current_keywords.intersection(prev_keywords)
                
                # If very little overlap, might be topic shift
                if len(overlap) < 2:
                    return True
        
        return False
    
    def _requires_previous_context(self, current_input: str) -> bool:
        """Check if the input requires previous context"""
        context_indicators = [
            "it", "this", "that", "the above", "the previous",
            "as mentioned", "like before", "the same way"
        ]
        
        lower_input = current_input.lower()
        
        for indicator in context_indicators:
            if indicator in lower_input:
                return True
        
        return False
    
    def _extract_relevant_context(
        self,
        current_input: str,
        recent_history: List[Dict[str, Any]]
    ) -> List[str]:
        """Extract relevant context from history"""
        relevant = []
        
        # Extract key information from recent messages
        for msg in recent_history:
            if msg.get("role") == "assistant":
                content = msg.get("content", "")
                
                # Extract code blocks
                code_blocks = re.findall(r'```[\s\S]*?```', content)
                if code_blocks:
                    relevant.extend(code_blocks)
                
                # Extract file paths
                file_paths = re.findall(r'[./\w]+\.\w+', content)
                if file_paths:
                    relevant.extend(file_paths)
        
        return relevant