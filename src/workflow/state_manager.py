"""LangGraph state management for agentic workflows"""

from typing import TypedDict, List, Optional, Dict, Any, Annotated
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver
from pydantic import BaseModel
from ..agents.models import TaskComplexity, FinalOutput, ValidationResult
import os


class AgentState(TypedDict):
    """Core state for the agentic workflow with Pydantic AI integration"""
    user_input: str
    conversation_history: List[Dict[str, Any]]
    current_task: Optional[str]
    task_complexity: Optional[TaskComplexity]
    agent_outputs: Dict[str, BaseModel]  # Stores structured Pydantic outputs
    validation_results: Dict[str, ValidationResult]
    synthesis_result: Optional[FinalOutput]
    error_count: int
    refinement_cycle: int
    dependencies: Dict[str, Any]  # Injected dependencies
    messages: List[str]  # For tracking agent communication


class PydanticWorkflowConfig(BaseModel):
    """Configuration for Pydantic AI workflow"""
    max_parallel_agents: int = 4
    max_refinement_cycles: int = 3
    validation_timeout: int = 300
    enable_self_correction: bool = True
    enable_logfire: bool = True
    model_provider: str = "openai:gpt-4"
    checkpoint_dir: str = "./checkpoints"


class WorkflowStateManager:
    """Manages the LangGraph state and workflow execution"""
    
    def __init__(self, config: PydanticWorkflowConfig = None):
        """Initialize the workflow state manager
        
        Args:
            config: Workflow configuration
        """
        self.config = config or PydanticWorkflowConfig()
        self.graph = None
        self.checkpointer = None
        self._build_graph()
    
    def _build_graph(self):
        """Build the LangGraph state graph"""
        # Create checkpointer for persistence
        os.makedirs(self.config.checkpoint_dir, exist_ok=True)
        self.checkpointer = SqliteSaver.from_conn_string(
            f"{self.config.checkpoint_dir}/workflow.db"
        )
        
        # Create the state graph
        self.graph = StateGraph(AgentState)
        
        # Add nodes for each step in the workflow
        self.graph.add_node("route_task", self._route_task)
        self.graph.add_node("simple_agent", self._run_simple_agent)
        self.graph.add_node("generate_questions", self._generate_questions)
        self.graph.add_node("parallel_agents", self._run_parallel_agents)
        self.graph.add_node("validation", self._run_validation)
        self.graph.add_node("self_correction", self._run_self_correction)
        self.graph.add_node("synthesis", self._run_synthesis)
        
        # Add edges
        self.graph.add_edge("route_task", "simple_agent", self._is_simple_task)
        self.graph.add_edge("route_task", "generate_questions", self._is_complex_task)
        self.graph.add_edge("generate_questions", "parallel_agents")
        self.graph.add_edge("simple_agent", "validation")
        self.graph.add_edge("parallel_agents", "validation")
        self.graph.add_edge("validation", "synthesis", self._validation_passed)
        self.graph.add_edge("validation", "self_correction", self._validation_failed)
        self.graph.add_edge("self_correction", "validation", self._continue_refinement)
        self.graph.add_edge("self_correction", "synthesis", self._max_refinement_reached)
        self.graph.add_edge("synthesis", END)
        
        # Set entry point
        self.graph.set_entry_point("route_task")
        
        # Compile the graph
        self.runnable = self.graph.compile(checkpointer=self.checkpointer)
    
    async def _route_task(self, state: AgentState) -> AgentState:
        """Route the task based on complexity"""
        # Import here to avoid circular imports
        from ..workflow.router import TaskRouter
        
        router = TaskRouter()
        complexity = await router.assess_complexity(state["user_input"])
        
        state["task_complexity"] = complexity
        state["messages"].append(f"Task routed as: {complexity}")
        
        return state
    
    async def _run_simple_agent(self, state: AgentState) -> AgentState:
        """Run a single agent for simple tasks"""
        # Import here to avoid circular imports
        from ..agents.coder_agent import CoderAgent
        from ..agents.dependencies import CoderDependencies
        
        # Create dependencies
        deps = CoderDependencies(
            user_id=state["dependencies"].get("user_id", "system"),
            session_id=state["dependencies"].get("session_id", "default"),
            api_keys=state["dependencies"].get("api_keys", {}),
            http_client=state["dependencies"].get("http_client"),
            workspace_path=state["dependencies"].get("workspace_path", ".")
        )
        
        # Run the agent
        agent = CoderAgent()
        result = await agent.run(state["user_input"], deps)
        
        # Store the result
        state["agent_outputs"]["coder"] = result
        state["messages"].append("Simple agent completed task")
        
        return state
    
    async def _generate_questions(self, state: AgentState) -> AgentState:
        """Generate specialized questions for complex tasks"""
        # Import here to avoid circular imports
        from ..orchestration.question_generator import QuestionGenerator
        
        generator = QuestionGenerator()
        questions = await generator.generate_questions(
            state["user_input"],
            state["task_complexity"]
        )
        
        state["agent_outputs"]["questions"] = questions
        state["messages"].append(f"Generated {len(questions)} specialized questions")
        
        return state
    
    async def _run_parallel_agents(self, state: AgentState) -> AgentState:
        """Run multiple agents in parallel"""
        # Import here to avoid circular imports
        from ..orchestration.coordinator import AgentCoordinator
        
        coordinator = AgentCoordinator(
            max_parallel=self.config.max_parallel_agents
        )
        
        # Get questions from state
        questions = state["agent_outputs"].get("questions", {})
        
        # Run agents in parallel
        results = await coordinator.run_parallel_agents(
            state["user_input"],
            questions,
            state["dependencies"]
        )
        
        # Store results
        state["agent_outputs"].update(results)
        state["messages"].append(f"Completed {len(results)} parallel agents")
        
        return state
    
    async def _run_validation(self, state: AgentState) -> AgentState:
        """Run validation gates on agent outputs"""
        # Import here to avoid circular imports
        from ..validation.gates import ValidationGates
        
        gates = ValidationGates()
        
        # Validate each agent output
        validation_results = {}
        
        for agent_name, output in state["agent_outputs"].items():
            if hasattr(output, "generated_code"):
                # Validate code output
                result = await gates.validate_all(output.generated_code)
                validation_results[agent_name] = ValidationResult(
                    test_passed=result.get("pytest", False),
                    lint_passed=result.get("ruff", False),
                    type_check_passed=result.get("mypy", False),
                    errors=result.get("errors", []),
                    warnings=result.get("warnings", [])
                )
        
        state["validation_results"] = validation_results
        state["messages"].append("Validation completed")
        
        return state
    
    async def _run_self_correction(self, state: AgentState) -> AgentState:
        """Run self-correction on failed validations"""
        if not self.config.enable_self_correction:
            state["messages"].append("Self-correction disabled")
            return state
        
        # Import here to avoid circular imports
        from ..validation.self_correction import SelfCorrectionEngine
        
        engine = SelfCorrectionEngine()
        
        # Increment refinement cycle
        state["refinement_cycle"] += 1
        
        # Apply corrections
        for agent_name, validation_result in state["validation_results"].items():
            if not all([
                validation_result.test_passed,
                validation_result.lint_passed,
                validation_result.type_check_passed
            ]):
                # Get the original output
                original_output = state["agent_outputs"].get(agent_name)
                
                if hasattr(original_output, "generated_code"):
                    # Apply corrections
                    corrected_code = await engine.correction_loop(
                        original_output.generated_code,
                        validation_result,
                        max_cycles=1  # Single correction per refinement cycle
                    )
                    
                    # Update the output
                    original_output.generated_code = corrected_code
                    state["agent_outputs"][agent_name] = original_output
        
        state["messages"].append(f"Self-correction cycle {state['refinement_cycle']} completed")
        
        return state
    
    async def _run_synthesis(self, state: AgentState) -> AgentState:
        """Synthesize final output from all agent results"""
        # Import here to avoid circular imports
        from ..agents.synthesis_agent import SynthesisAgent
        from ..agents.dependencies import SynthesisDependencies
        
        # Create dependencies
        deps = SynthesisDependencies(
            user_id=state["dependencies"].get("user_id", "system"),
            session_id=state["dependencies"].get("session_id", "default"),
            api_keys=state["dependencies"].get("api_keys", {}),
            http_client=state["dependencies"].get("http_client"),
            agent_outputs=state["agent_outputs"]
        )
        
        # Run synthesis
        agent = SynthesisAgent()
        synthesis_result = await agent.run(
            f"Synthesize results for: {state['user_input']}",
            deps
        )
        
        state["synthesis_result"] = synthesis_result
        state["messages"].append("Synthesis completed")
        
        return state
    
    def _is_simple_task(self, state: AgentState) -> bool:
        """Check if the task is simple"""
        return state.get("task_complexity") == TaskComplexity.SIMPLE
    
    def _is_complex_task(self, state: AgentState) -> bool:
        """Check if the task is complex"""
        return state.get("task_complexity") in [
            TaskComplexity.COMPLEX,
            TaskComplexity.RESEARCH
        ]
    
    def _validation_passed(self, state: AgentState) -> bool:
        """Check if validation passed"""
        if not state.get("validation_results"):
            return True
        
        # Check if all validations passed
        for result in state["validation_results"].values():
            if not all([
                result.test_passed,
                result.lint_passed,
                result.type_check_passed
            ]):
                return False
        
        return True
    
    def _validation_failed(self, state: AgentState) -> bool:
        """Check if validation failed"""
        return not self._validation_passed(state)
    
    def _continue_refinement(self, state: AgentState) -> bool:
        """Check if we should continue refinement"""
        return (
            state["refinement_cycle"] < self.config.max_refinement_cycles and
            self.config.enable_self_correction
        )
    
    def _max_refinement_reached(self, state: AgentState) -> bool:
        """Check if max refinement cycles reached"""
        return (
            state["refinement_cycle"] >= self.config.max_refinement_cycles or
            not self.config.enable_self_correction
        )
    
    async def run_workflow(
        self,
        user_input: str,
        dependencies: Dict[str, Any],
        thread_id: str = "default"
    ) -> AgentState:
        """Run the complete workflow
        
        Args:
            user_input: The user's input/query
            dependencies: Dependencies to inject
            thread_id: Thread ID for conversation continuity
            
        Returns:
            The final state after workflow completion
        """
        # Initialize state
        initial_state: AgentState = {
            "user_input": user_input,
            "conversation_history": [],
            "current_task": user_input,
            "task_complexity": None,
            "agent_outputs": {},
            "validation_results": {},
            "synthesis_result": None,
            "error_count": 0,
            "refinement_cycle": 0,
            "dependencies": dependencies,
            "messages": []
        }
        
        # Run the workflow
        config = {"configurable": {"thread_id": thread_id}}
        final_state = await self.runnable.ainvoke(initial_state, config)
        
        return final_state
    
    def get_conversation_history(self, thread_id: str = "default") -> List[Dict[str, Any]]:
        """Get conversation history for a thread
        
        Args:
            thread_id: The thread ID
            
        Returns:
            List of conversation history
        """
        if self.checkpointer:
            checkpoint = self.checkpointer.get(
                {"configurable": {"thread_id": thread_id}}
            )
            if checkpoint and checkpoint.get("channel_values"):
                return checkpoint["channel_values"].get("conversation_history", [])
        
        return []