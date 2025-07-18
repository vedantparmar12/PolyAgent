"""Pydantic AI agent base classes"""

from abc import ABC, abstractmethod
from typing import TypeVar, Generic, Type, Any, Optional, Dict, List, Union
from pydantic import BaseModel
from pydantic_ai import Agent
import logfire
from ..monitoring.metrics import MetricsCollector
from ..core.model_provider import ModelProvider, ModelConfig, ModelInfo


# Type variables for dependency and result types
TDeps = TypeVar('TDeps')
TResult = TypeVar('TResult', bound=BaseModel)


class BaseAgent(ABC, Generic[TDeps, TResult]):
    """Base class for all Pydantic AI agents"""
    
    def __init__(
        self,
        model: Union[str, ModelInfo, None] = None,
        deps_type: Type[TDeps] = None,
        result_type: Type[TResult] = None,
        system_prompt: str = None,
        enable_logfire: bool = True,
        model_config: Optional[ModelConfig] = None
    ):
        """Initialize the base agent
        
        Args:
            model: The AI model to use (model ID string, ModelInfo object, or None for default)
            deps_type: The dependency type for this agent
            result_type: The result type this agent produces
            system_prompt: The system prompt for the agent
            enable_logfire: Whether to enable Pydantic Logfire monitoring
            model_config: Model configuration (includes API keys and preferences)
        """
        # Initialize model provider
        self.model_config = model_config or ModelConfig()
        self.model_provider = ModelProvider(config=self.model_config.model_dump())
        
        # Determine model to use
        if model is None:
            # Use default from config
            model_id = self.model_config.default_model
        elif isinstance(model, ModelInfo):
            model_id = model.id
        else:
            model_id = model
        
        # Get model string for Pydantic AI
        self.model = self.model_provider.get_model_string(model_id)
        self.model_info = self.model_provider.get_model_info(model_id)
        
        self.deps_type = deps_type
        self.result_type = result_type
        self._system_prompt = system_prompt or self.get_system_prompt()
        
        # Initialize Pydantic AI agent
        self.agent = Agent(
            model=self.model,
            deps_type=self.deps_type,
            result_type=self.result_type,
            system_prompt=self._system_prompt,
        )
        
        # Configure monitoring
        if enable_logfire:
            logfire.configure()
            self.metrics = MetricsCollector()
        
        # Register tools
        self._register_tools()
    
    @abstractmethod
    def get_system_prompt(self) -> str:
        """Get the system prompt for this agent"""
        pass
    
    @abstractmethod
    def _register_tools(self):
        """Register tools for this agent"""
        pass
    
    async def run(self, prompt: str, deps: TDeps) -> TResult:
        """Run the agent with the given prompt and dependencies
        
        Args:
            prompt: The user prompt
            deps: The dependencies for this run
            
        Returns:
            The structured result from the agent
        """
        try:
            # Log the start of execution
            logfire.info(
                "agent_execution_started",
                agent_type=self.__class__.__name__,
                prompt_length=len(prompt)
            )
            
            # Run the agent
            result = await self.agent.run(prompt, deps=deps)
            
            # Log successful completion
            logfire.info(
                "agent_execution_completed",
                agent_type=self.__class__.__name__,
                success=True
            )
            
            return result.data
            
        except Exception as e:
            # Log the error
            logfire.error(
                "agent_execution_failed",
                agent_type=self.__class__.__name__,
                error=str(e)
            )
            raise
    
    async def run_stream(self, prompt: str, deps: TDeps):
        """Run the agent with streaming output
        
        Args:
            prompt: The user prompt
            deps: The dependencies for this run
            
        Yields:
            Streaming responses from the agent
        """
        async with self.agent.run_stream(prompt, deps=deps) as stream:
            async for chunk in stream:
                yield chunk


class MultiAgentCoordinator:
    """Coordinator for running multiple agents in parallel"""
    
    def __init__(self, max_parallel: int = 4):
        """Initialize the coordinator
        
        Args:
            max_parallel: Maximum number of agents to run in parallel
        """
        self.max_parallel = max_parallel
        self.active_agents: Dict[str, BaseAgent] = {}
    
    def register_agent(self, name: str, agent: BaseAgent):
        """Register an agent with the coordinator
        
        Args:
            name: The name to register the agent under
            agent: The agent instance
        """
        self.active_agents[name] = agent
    
    async def run_agents(
        self,
        agent_tasks: List[Dict[str, Any]],
        parallel: bool = True
    ) -> Dict[str, Any]:
        """Run multiple agents with their tasks
        
        Args:
            agent_tasks: List of dicts with 'agent_name', 'prompt', and 'deps'
            parallel: Whether to run agents in parallel
            
        Returns:
            Dictionary mapping agent names to their results
        """
        import asyncio
        
        results = {}
        
        if parallel:
            # Run agents in parallel
            tasks = []
            for task in agent_tasks:
                agent_name = task['agent_name']
                agent = self.active_agents.get(agent_name)
                
                if not agent:
                    raise ValueError(f"Agent '{agent_name}' not registered")
                
                tasks.append(
                    self._run_single_agent(
                        agent_name,
                        agent,
                        task['prompt'],
                        task['deps']
                    )
                )
            
            # Execute all tasks concurrently
            agent_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            for i, result in enumerate(agent_results):
                agent_name = agent_tasks[i]['agent_name']
                if isinstance(result, Exception):
                    results[agent_name] = {
                        'success': False,
                        'error': str(result)
                    }
                else:
                    results[agent_name] = {
                        'success': True,
                        'data': result
                    }
        else:
            # Run agents sequentially
            for task in agent_tasks:
                agent_name = task['agent_name']
                agent = self.active_agents.get(agent_name)
                
                if not agent:
                    raise ValueError(f"Agent '{agent_name}' not registered")
                
                try:
                    result = await agent.run(task['prompt'], task['deps'])
                    results[agent_name] = {
                        'success': True,
                        'data': result
                    }
                except Exception as e:
                    results[agent_name] = {
                        'success': False,
                        'error': str(e)
                    }
        
        return results
    
    async def _run_single_agent(
        self,
        agent_name: str,
        agent: BaseAgent,
        prompt: str,
        deps: Any
    ) -> Any:
        """Run a single agent and return its result
        
        Args:
            agent_name: Name of the agent
            agent: The agent instance
            prompt: The prompt for the agent
            deps: Dependencies for the agent
            
        Returns:
            The agent's result
        """
        logfire.info(
            "parallel_agent_started",
            agent_name=agent_name
        )
        
        result = await agent.run(prompt, deps)
        
        logfire.info(
            "parallel_agent_completed",
            agent_name=agent_name
        )
        
        return result


class AgentRegistry:
    """Registry for managing available agents"""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.agents = {}
        return cls._instance
    
    def register(self, name: str, agent_class: Type[BaseAgent]):
        """Register an agent class
        
        Args:
            name: The name to register the agent under
            agent_class: The agent class
        """
        self.agents[name] = agent_class
    
    def get(self, name: str) -> Optional[Type[BaseAgent]]:
        """Get an agent class by name
        
        Args:
            name: The name of the agent
            
        Returns:
            The agent class if found, None otherwise
        """
        return self.agents.get(name)
    
    def list_agents(self) -> List[str]:
        """List all registered agent names
        
        Returns:
            List of agent names
        """
        return list(self.agents.keys())