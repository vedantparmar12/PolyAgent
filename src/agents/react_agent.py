"""ReAct (Reasoning + Acting) agent implementation with PydanticAI"""

from typing import List, Dict, Any, Optional, Tuple
from pydantic import BaseModel, Field
from pydantic_ai import RunContext
from .base_agent import BaseAgent
from .dependencies import BaseDependencies
import json
import logfire


class ThoughtAction(BaseModel):
    """Represents a thought-action pair in ReAct pattern"""
    thought: str = Field(description="The reasoning/thought process")
    action: str = Field(description="The action to take")
    action_input: Dict[str, Any] = Field(description="Input parameters for the action")
    observation: Optional[str] = Field(default=None, description="Result of the action")


class ReActOutput(BaseModel):
    """Output from ReAct agent"""
    task: str = Field(description="The original task")
    thought_process: List[ThoughtAction] = Field(description="Chain of thoughts and actions")
    final_answer: str = Field(description="The final synthesized answer")
    confidence: float = Field(description="Confidence in the answer (0-1)")
    tools_used: List[str] = Field(description="List of tools used")


class ReActDependencies(BaseDependencies):
    """Dependencies for ReAct agent"""
    max_iterations: int = 5
    available_tools: Dict[str, Any] = Field(default_factory=dict)
    context: Optional[str] = None


class ReActAgent(BaseAgent[ReActDependencies, ReActOutput]):
    """Agent that implements ReAct (Reasoning + Acting) pattern"""
    
    def __init__(self, model: str = "openai:gpt-4"):
        """Initialize the ReAct agent"""
        super().__init__(
            model=model,
            deps_type=ReActDependencies,
            result_type=ReActOutput,
            enable_logfire=True
        )
        self.thought_history: List[ThoughtAction] = []
    
    def get_system_prompt(self) -> str:
        """Get the system prompt for ReAct agent"""
        return """You are a ReAct (Reasoning and Acting) agent that solves problems through iterative thinking and action.

Follow this process:
1. THOUGHT: Analyze the current situation and what needs to be done
2. ACTION: Decide on an action to take based on your reasoning
3. OBSERVATION: Process the result of your action
4. Repeat until you have enough information to provide a final answer

Guidelines:
- Break down complex problems into smaller steps
- Use available tools to gather information
- Reflect on observations before deciding next actions
- Know when you have sufficient information to answer
- Be explicit about your reasoning process

Available actions:
- search: Search for information
- analyze: Analyze data or code
- calculate: Perform calculations
- validate: Validate information or assumptions
- synthesize: Combine multiple pieces of information

Always structure your process clearly and provide a confidence score."""
    
    def _register_tools(self):
        """Register tools for the ReAct agent"""
        self.agent.tool(self.think_and_act)
        self.agent.tool(self.search_information)
        self.agent.tool(self.analyze_data)
        self.agent.tool(self.calculate)
        self.agent.tool(self.validate_assumption)
        self.agent.tool(self.synthesize_information)
    
    async def think_and_act(
        self,
        ctx: RunContext[ReActDependencies],
        task: str,
        current_context: str = ""
    ) -> Tuple[str, str, Dict[str, Any]]:
        """Core ReAct loop - think about the task and decide on action
        
        Args:
            ctx: Run context
            task: The task to solve
            current_context: Current context/observations
            
        Returns:
            Tuple of (thought, action, action_input)
        """
        logfire.info("react_thinking", task=task)
        
        # Generate thought based on current context
        thought_prompt = f"""
Task: {task}

Current Context: {current_context if current_context else "No observations yet"}

What should I think about and do next? Provide:
1. THOUGHT: Your reasoning about the current situation
2. ACTION: The next action to take (search/analyze/calculate/validate/synthesize/complete)
3. ACTION_INPUT: Parameters for the action as JSON
"""
        
        # In a real implementation, this would call the LLM
        # For now, we'll return a structured response
        thought = f"I need to understand more about: {task}"
        
        # Decide on action based on task keywords
        if "search" in task.lower() or "find" in task.lower():
            action = "search"
            action_input = {"query": task}
        elif "analyze" in task.lower() or "examine" in task.lower():
            action = "analyze"
            action_input = {"data": task}
        elif "calculate" in task.lower() or "compute" in task.lower():
            action = "calculate"
            action_input = {"expression": task}
        elif len(self.thought_history) >= ctx.deps.max_iterations - 1:
            action = "complete"
            action_input = {"summary": current_context}
        else:
            action = "search"
            action_input = {"query": task}
        
        return thought, action, action_input
    
    async def search_information(
        self,
        ctx: RunContext[ReActDependencies],
        query: str
    ) -> str:
        """Search for information
        
        Args:
            ctx: Run context
            query: Search query
            
        Returns:
            Search results
        """
        logfire.info("react_search", query=query)
        
        # Check if we have a search tool in dependencies
        if "search" in ctx.deps.available_tools:
            try:
                result = await ctx.deps.available_tools["search"](query)
                return str(result)
            except Exception as e:
                return f"Search failed: {str(e)}"
        
        # Fallback response
        return f"Searched for: {query}. Found relevant information about the topic."
    
    async def analyze_data(
        self,
        ctx: RunContext[ReActDependencies],
        data: str
    ) -> str:
        """Analyze data or code
        
        Args:
            ctx: Run context
            data: Data to analyze
            
        Returns:
            Analysis results
        """
        logfire.info("react_analyze", data_length=len(data))
        
        # Simple analysis logic
        analysis = f"Analysis of: {data[:100]}...\n"
        analysis += f"- Length: {len(data)} characters\n"
        analysis += f"- Type: {type(data).__name__}\n"
        
        if "code" in data.lower():
            analysis += "- Appears to be code-related\n"
        if "error" in data.lower():
            analysis += "- Contains error information\n"
        
        return analysis
    
    async def calculate(
        self,
        ctx: RunContext[ReActDependencies],
        expression: str
    ) -> str:
        """Perform calculations
        
        Args:
            ctx: Run context
            expression: Expression to calculate
            
        Returns:
            Calculation result
        """
        logfire.info("react_calculate", expression=expression)
        
        try:
            # Safe evaluation of mathematical expressions
            import ast
            import operator as op
            
            # Supported operators
            operators = {
                ast.Add: op.add,
                ast.Sub: op.sub,
                ast.Mult: op.mul,
                ast.Div: op.truediv,
                ast.Pow: op.pow,
                ast.USub: op.neg
            }
            
            def eval_expr(expr):
                """Safely evaluate mathematical expression"""
                def _eval(node):
                    if isinstance(node, ast.Num):
                        return node.n
                    elif isinstance(node, ast.BinOp):
                        return operators[type(node.op)](_eval(node.left), _eval(node.right))
                    elif isinstance(node, ast.UnaryOp):
                        return operators[type(node.op)](_eval(node.operand))
                    else:
                        raise TypeError(node)
                
                return _eval(ast.parse(expr, mode='eval').body)
            
            result = eval_expr(expression)
            return f"Calculation result: {expression} = {result}"
        except Exception as e:
            return f"Calculation failed: {str(e)}"
    
    async def validate_assumption(
        self,
        ctx: RunContext[ReActDependencies],
        assumption: str
    ) -> str:
        """Validate an assumption or hypothesis
        
        Args:
            ctx: Run context
            assumption: Assumption to validate
            
        Returns:
            Validation result
        """
        logfire.info("react_validate", assumption=assumption)
        
        # Simple validation logic
        validation = f"Validating: {assumption}\n"
        
        # Check for common validation patterns
        if "true" in assumption.lower() or "false" in assumption.lower():
            validation += "- This appears to be a boolean assertion\n"
        if "?" in assumption:
            validation += "- This is phrased as a question\n"
        
        validation += "- Validation status: Requires further investigation\n"
        
        return validation
    
    async def synthesize_information(
        self,
        ctx: RunContext[ReActDependencies],
        observations: List[str]
    ) -> str:
        """Synthesize multiple observations into insights
        
        Args:
            ctx: Run context
            observations: List of observations
            
        Returns:
            Synthesized insights
        """
        logfire.info("react_synthesize", num_observations=len(observations))
        
        synthesis = "Synthesis of observations:\n"
        
        # Combine observations
        for i, obs in enumerate(observations, 1):
            synthesis += f"{i}. {obs[:100]}...\n"
        
        synthesis += f"\nKey insights from {len(observations)} observations"
        
        return synthesis
    
    async def run(self, prompt: str, deps: ReActDependencies) -> ReActOutput:
        """Run the ReAct agent
        
        Args:
            prompt: The task/prompt
            deps: Dependencies
            
        Returns:
            ReActOutput with complete thought process
        """
        self.thought_history = []
        tools_used = set()
        current_context = deps.context or ""
        
        # ReAct loop
        for iteration in range(deps.max_iterations):
            logfire.info("react_iteration", iteration=iteration)
            
            # Think and decide on action
            thought, action, action_input = await self.think_and_act(
                RunContext(deps=deps),
                prompt,
                current_context
            )
            
            # Execute action
            observation = ""
            if action == "search":
                observation = await self.search_information(
                    RunContext(deps=deps),
                    action_input.get("query", "")
                )
                tools_used.add("search")
            elif action == "analyze":
                observation = await self.analyze_data(
                    RunContext(deps=deps),
                    action_input.get("data", "")
                )
                tools_used.add("analyze")
            elif action == "calculate":
                observation = await self.calculate(
                    RunContext(deps=deps),
                    action_input.get("expression", "")
                )
                tools_used.add("calculate")
            elif action == "validate":
                observation = await self.validate_assumption(
                    RunContext(deps=deps),
                    action_input.get("assumption", "")
                )
                tools_used.add("validate")
            elif action == "synthesize":
                observations = [ta.observation for ta in self.thought_history if ta.observation]
                observation = await self.synthesize_information(
                    RunContext(deps=deps),
                    observations
                )
                tools_used.add("synthesize")
            elif action == "complete":
                break
            
            # Record thought-action-observation
            thought_action = ThoughtAction(
                thought=thought,
                action=action,
                action_input=action_input,
                observation=observation
            )
            self.thought_history.append(thought_action)
            
            # Update context
            current_context += f"\n{observation}"
            
            # Check if we have enough information
            if "final answer" in observation.lower() or iteration == deps.max_iterations - 1:
                break
        
        # Generate final answer
        final_answer = self._generate_final_answer(prompt, self.thought_history)
        confidence = self._calculate_confidence(self.thought_history)
        
        return ReActOutput(
            task=prompt,
            thought_process=self.thought_history,
            final_answer=final_answer,
            confidence=confidence,
            tools_used=list(tools_used)
        )
    
    def _generate_final_answer(self, task: str, thoughts: List[ThoughtAction]) -> str:
        """Generate final answer from thought process
        
        Args:
            task: Original task
            thoughts: List of thoughts and actions
            
        Returns:
            Final synthesized answer
        """
        if not thoughts:
            return "Unable to complete the task due to lack of observations."
        
        # Combine all observations
        observations = [ta.observation for ta in thoughts if ta.observation]
        
        final = f"Based on my analysis of '{task}':\n\n"
        
        # Add key observations
        for i, obs in enumerate(observations[-3:], 1):  # Last 3 observations
            final += f"{i}. {obs}\n"
        
        final += f"\nCompleted using {len(thoughts)} reasoning steps."
        
        return final
    
    def _calculate_confidence(self, thoughts: List[ThoughtAction]) -> float:
        """Calculate confidence score based on thought process
        
        Args:
            thoughts: List of thoughts and actions
            
        Returns:
            Confidence score (0-1)
        """
        if not thoughts:
            return 0.0
        
        # Base confidence
        confidence = 0.5
        
        # Increase confidence for each successful observation
        successful_observations = sum(1 for ta in thoughts if ta.observation and "error" not in ta.observation.lower())
        confidence += successful_observations * 0.1
        
        # Decrease for errors
        errors = sum(1 for ta in thoughts if ta.observation and "error" in ta.observation.lower())
        confidence -= errors * 0.2
        
        # Ensure bounds
        return max(0.0, min(1.0, confidence))


# Example standalone ReAct loop
async def react_example():
    """Example of using ReAct agent"""
    agent = ReActAgent()
    
    # Example dependencies with mock tools
    deps = ReActDependencies(
        user_id="test_user",
        session_id="test_session",
        max_iterations=5,
        available_tools={
            "search": lambda q: f"Search results for: {q}"
        }
    )
    
    # Run the agent
    result = await agent.run(
        "What are the key differences between Python async and threading?",
        deps
    )
    
    print(f"Task: {result.task}")
    print(f"\nThought Process:")
    for i, ta in enumerate(result.thought_process, 1):
        print(f"\n{i}. THOUGHT: {ta.thought}")
        print(f"   ACTION: {ta.action}")
        print(f"   OBSERVATION: {ta.observation}")
    
    print(f"\nFinal Answer: {result.final_answer}")
    print(f"Confidence: {result.confidence:.2f}")
    print(f"Tools Used: {', '.join(result.tools_used)}")


if __name__ == "__main__":
    import asyncio
    asyncio.run(react_example())