"""Agent generation system inspired by Archon's approach"""

from typing import Dict, Any, List, Optional, Tuple
from pydantic import BaseModel, Field
from pydantic_ai import Agent
import yaml
import json
from pathlib import Path
import logfire
from datetime import datetime
from ..agents.base_agent import BaseAgent
from ..workflow.state_manager import AgentState
from langgraph.graph import StateGraph, END


class AgentRequirements(BaseModel):
    """Requirements for agent generation"""
    task_description: str = Field(description="What the agent should do")
    capabilities: List[str] = Field(description="Required capabilities")
    tools_needed: List[str] = Field(description="Tools the agent should use")
    model_preference: Optional[str] = Field(default=None, description="Preferred AI model")
    performance_requirements: Optional[Dict[str, Any]] = Field(default=None)
    examples: Optional[List[str]] = Field(default=None, description="Example use cases")


class AgentComponent(BaseModel):
    """A reusable agent component"""
    name: str
    type: str  # 'tool', 'prompt', 'pattern', 'example'
    description: str
    code: str
    dependencies: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class GeneratedAgent(BaseModel):
    """Complete generated agent package"""
    name: str
    description: str
    agent_code: str
    tools_code: str
    prompts_code: str
    config: Dict[str, Any]
    requirements: List[str]
    tests: str
    documentation: str
    version: str = "1.0.0"
    created_at: datetime = Field(default_factory=datetime.now)


class AgentGenerationState(AgentState):
    """Extended state for agent generation workflow"""
    requirements: AgentRequirements
    recommended_components: List[AgentComponent]
    generated_code: Dict[str, str]
    validation_results: Dict[str, bool]
    refinement_suggestions: List[str]
    final_agent: Optional[GeneratedAgent]


class AgentGenerator:
    """Generates complete agent implementations based on requirements"""
    
    def __init__(self, library_path: str = "src/library"):
        """Initialize the agent generator
        
        Args:
            library_path: Path to component library
        """
        self.library_path = Path(library_path)
        self.component_registry = self._load_component_registry()
        self._build_workflow()
        
    def _load_component_registry(self) -> Dict[str, List[AgentComponent]]:
        """Load available components from library"""
        registry = {
            "tools": [],
            "prompts": [],
            "patterns": [],
            "examples": []
        }
        
        # Load components from library (simplified for now)
        # In full implementation, this would scan the library directory
        return registry
    
    def _build_workflow(self):
        """Build the LangGraph workflow for agent generation"""
        workflow = StateGraph(AgentGenerationState)
        
        # Add nodes
        workflow.add_node("analyze_requirements", self.analyze_requirements)
        workflow.add_node("recommend_components", self.recommend_components)
        workflow.add_node("generate_agent_code", self.generate_agent_code)
        workflow.add_node("generate_tools", self.generate_tools)
        workflow.add_node("generate_prompts", self.generate_prompts)
        workflow.add_node("validate_agent", self.validate_agent)
        workflow.add_node("refine_agent", self.refine_agent)
        workflow.add_node("package_agent", self.package_agent)
        
        # Add edges
        workflow.add_edge("analyze_requirements", "recommend_components")
        workflow.add_edge("recommend_components", "generate_agent_code")
        workflow.add_edge("generate_agent_code", "generate_tools")
        workflow.add_edge("generate_tools", "generate_prompts")
        workflow.add_edge("generate_prompts", "validate_agent")
        
        # Conditional edges
        workflow.add_conditional_edges(
            "validate_agent",
            self.check_validation,
            {
                "refine": "refine_agent",
                "complete": "package_agent"
            }
        )
        
        workflow.add_edge("refine_agent", "generate_agent_code")
        workflow.add_edge("package_agent", END)
        
        # Set entry point
        workflow.set_entry_point("analyze_requirements")
        
        # Compile
        self.app = workflow.compile()
    
    async def analyze_requirements(self, state: AgentGenerationState) -> AgentGenerationState:
        """Analyze and understand agent requirements"""
        logfire.info("analyzing_agent_requirements")
        
        # Create scope reasoner
        scope_agent = Agent(
            model="openai:gpt-4",
            system_prompt="""You are an expert at analyzing agent requirements.
            Break down the requirements into:
            1. Core functionality needed
            2. Required tools and integrations
            3. Performance constraints
            4. Suggested implementation approach"""
        )
        
        result = await scope_agent.run(
            f"Analyze these agent requirements: {state['requirements'].model_dump()}"
        )
        
        state['messages'].append(f"Requirements analyzed: {result.data}")
        return state
    
    async def recommend_components(self, state: AgentGenerationState) -> AgentGenerationState:
        """Recommend relevant components from library"""
        logfire.info("recommending_components")
        
        # Create advisor agent
        advisor_agent = Agent(
            model="openai:gpt-4",
            system_prompt="""You are an expert at recommending reusable components.
            Based on the requirements, suggest:
            1. Relevant tools from the library
            2. Appropriate prompt templates
            3. Useful patterns to follow
            4. Similar example agents"""
        )
        
        # In full implementation, this would search the component registry
        recommendations = []
        
        # Add some example recommendations
        if "github" in str(state['requirements']).lower():
            recommendations.append(AgentComponent(
                name="GitHubTool",
                type="tool",
                description="Tool for interacting with GitHub API",
                code="""class GitHubTool:
    def __init__(self, token: str):
        self.token = token
    
    async def get_repo_info(self, owner: str, repo: str):
        # Implementation here
        pass""",
                dependencies=["httpx", "pydantic"]
            ))
        
        state['recommended_components'] = recommendations
        state['messages'].append(f"Recommended {len(recommendations)} components")
        return state
    
    async def generate_agent_code(self, state: AgentGenerationState) -> AgentGenerationState:
        """Generate the main agent code"""
        logfire.info("generating_agent_code")
        
        # Create code generation agent
        code_agent = Agent(
            model="openai:gpt-4",
            system_prompt="""You are an expert at generating Pydantic AI agent code.
            Generate clean, well-structured agent implementations that:
            1. Follow Pydantic AI best practices
            2. Include proper error handling
            3. Have comprehensive docstrings
            4. Use type hints throughout"""
        )
        
        # Generate agent code
        prompt = f"""Generate a Pydantic AI agent for: {state['requirements'].task_description}
        
        Use these recommended components: {state['recommended_components']}
        
        The agent should:
        - Inherit from BaseAgent
        - Implement all required methods
        - Use appropriate tools
        - Have proper error handling"""
        
        result = await code_agent.run(prompt)
        
        # Store generated code
        if 'generated_code' not in state:
            state['generated_code'] = {}
        
        state['generated_code']['agent'] = result.data
        state['messages'].append("Generated main agent code")
        return state
    
    async def generate_tools(self, state: AgentGenerationState) -> AgentGenerationState:
        """Generate tool implementations"""
        logfire.info("generating_tools")
        
        tools_agent = Agent(
            model="openai:gpt-4",
            system_prompt="""You are an expert at creating tools for Pydantic AI agents.
            Generate tool implementations that are:
            1. Type-safe with Pydantic models
            2. Async-first
            3. Well-documented
            4. Properly error-handled"""
        )
        
        prompt = f"""Generate tools for the agent based on requirements: {state['requirements'].tools_needed}
        
        Each tool should:
        - Be a proper Pydantic AI tool
        - Have clear input/output types
        - Include error handling
        - Be well documented"""
        
        result = await tools_agent.run(prompt)
        
        state['generated_code']['tools'] = result.data
        state['messages'].append("Generated agent tools")
        return state
    
    async def generate_prompts(self, state: AgentGenerationState) -> AgentGenerationState:
        """Generate optimized prompts"""
        logfire.info("generating_prompts")
        
        prompt_agent = Agent(
            model="openai:gpt-4",
            system_prompt="""You are an expert at crafting effective prompts.
            Generate prompts that are:
            1. Clear and specific
            2. Include relevant context
            3. Guide the model effectively
            4. Handle edge cases"""
        )
        
        prompt = f"""Generate system and task prompts for: {state['requirements'].task_description}
        
        Create:
        1. Main system prompt
        2. Task-specific prompts
        3. Error handling prompts
        4. Refinement prompts"""
        
        result = await prompt_agent.run(prompt)
        
        state['generated_code']['prompts'] = result.data
        state['messages'].append("Generated agent prompts")
        return state
    
    async def validate_agent(self, state: AgentGenerationState) -> AgentGenerationState:
        """Validate the generated agent"""
        logfire.info("validating_agent")
        
        validation_results = {
            "syntax_valid": True,  # Would run actual syntax check
            "imports_valid": True,  # Would check imports
            "structure_valid": True,  # Would validate structure
            "tools_valid": True,  # Would validate tools
        }
        
        # In full implementation, would:
        # 1. Check Python syntax
        # 2. Validate imports and dependencies
        # 3. Ensure proper inheritance
        # 4. Check tool implementations
        
        state['validation_results'] = validation_results
        
        # Add refinement suggestions if needed
        if not all(validation_results.values()):
            state['refinement_suggestions'] = [
                "Fix syntax errors in agent code",
                "Ensure all imports are available",
                "Validate tool method signatures"
            ]
        
        state['messages'].append("Validation completed")
        return state
    
    def check_validation(self, state: AgentGenerationState) -> str:
        """Check if validation passed"""
        if all(state.get('validation_results', {}).values()):
            return "complete"
        return "refine"
    
    async def refine_agent(self, state: AgentGenerationState) -> AgentGenerationState:
        """Refine the agent based on validation feedback"""
        logfire.info("refining_agent")
        
        refiner_agent = Agent(
            model="openai:gpt-4",
            system_prompt="""You are an expert at refining and improving agent code.
            Fix issues and enhance the implementation based on feedback."""
        )
        
        prompt = f"""Refine the agent code based on these issues:
        {state.get('refinement_suggestions', [])}
        
        Current code:
        {state['generated_code'].get('agent', '')}"""
        
        result = await refiner_agent.run(prompt)
        
        # Update code with refinements
        state['generated_code']['agent'] = result.data
        state['messages'].append("Refined agent code")
        return state
    
    async def package_agent(self, state: AgentGenerationState) -> AgentGenerationState:
        """Package the complete agent"""
        logfire.info("packaging_agent")
        
        # Generate configuration
        config = {
            "name": f"Generated{state['requirements'].task_description.replace(' ', '')}Agent",
            "version": "1.0.0",
            "model": state['requirements'].model_preference or "openai:gpt-4",
            "dependencies": self._extract_dependencies(state['generated_code'])
        }
        
        # Generate tests
        tests = self._generate_tests(state)
        
        # Generate documentation
        documentation = self._generate_documentation(state)
        
        # Create final package
        final_agent = GeneratedAgent(
            name=config['name'],
            description=state['requirements'].task_description,
            agent_code=state['generated_code'].get('agent', ''),
            tools_code=state['generated_code'].get('tools', ''),
            prompts_code=state['generated_code'].get('prompts', ''),
            config=config,
            requirements=self._extract_dependencies(state['generated_code']),
            tests=tests,
            documentation=documentation
        )
        
        state['final_agent'] = final_agent
        state['messages'].append("Agent package created successfully")
        return state
    
    def _extract_dependencies(self, code: Dict[str, str]) -> List[str]:
        """Extract Python dependencies from generated code"""
        dependencies = set()
        
        # Basic extraction - in full implementation would use AST
        for code_block in code.values():
            if "import " in code_block:
                lines = code_block.split('\n')
                for line in lines:
                    if line.strip().startswith('import ') or line.strip().startswith('from '):
                        # Extract module name
                        parts = line.split()
                        if parts[0] == 'import':
                            dependencies.add(parts[1].split('.')[0])
                        elif parts[0] == 'from':
                            dependencies.add(parts[1].split('.')[0])
        
        return list(dependencies)
    
    def _generate_tests(self, state: AgentGenerationState) -> str:
        """Generate test suite for the agent"""
        return f"""import pytest
from {state['final_agent'].name} import {state['final_agent'].name}

class Test{state['final_agent'].name}:
    def test_initialization(self):
        agent = {state['final_agent'].name}()
        assert agent is not None
    
    async def test_basic_functionality(self):
        agent = {state['final_agent'].name}()
        result = await agent.run("Test query")
        assert result is not None
"""
    
    def _generate_documentation(self, state: AgentGenerationState) -> str:
        """Generate documentation for the agent"""
        return f"""# {state['final_agent'].name}

## Description
{state['final_agent'].description}

## Requirements
{state['requirements'].model_dump()}

## Usage
```python
from {state['final_agent'].name} import {state['final_agent'].name}

agent = {state['final_agent'].name}()
result = await agent.run("Your query here")
```

## Configuration
{yaml.dump(state['final_agent'].config)}

## Generated Components
- Main agent code
- Tool implementations  
- Optimized prompts
- Test suite
"""
    
    async def generate_agent(
        self,
        requirements: AgentRequirements,
        output_dir: Optional[Path] = None
    ) -> GeneratedAgent:
        """Generate a complete agent based on requirements
        
        Args:
            requirements: Agent requirements specification
            output_dir: Directory to save generated files
            
        Returns:
            Generated agent package
        """
        # Initialize state
        initial_state: AgentGenerationState = {
            'requirements': requirements,
            'recommended_components': [],
            'generated_code': {},
            'validation_results': {},
            'refinement_suggestions': [],
            'final_agent': None,
            'messages': [],
            # Required base state fields
            'user_input': requirements.task_description,
            'conversation_id': '',
            'working_memory': [],
            'relevant_memories': [],
            'task_complexity': 'complex',
            'subtasks': [],
            'agent_outputs': {},
            'agent_errors': {},
            'current_step': 'start',
            'retry_count': 0,
            'should_continue': True,
            'final_response': '',
            'synthesis_complete': False
        }
        
        # Run workflow
        final_state = await self.app.ainvoke(initial_state)
        
        # Save to files if output directory provided
        if output_dir and final_state['final_agent']:
            await self._save_agent_files(final_state['final_agent'], output_dir)
        
        return final_state['final_agent']
    
    async def _save_agent_files(self, agent: GeneratedAgent, output_dir: Path):
        """Save generated agent to files"""
        output_dir = Path(output_dir)
        agent_dir = output_dir / agent.name.lower()
        agent_dir.mkdir(parents=True, exist_ok=True)
        
        # Save agent code
        (agent_dir / "agent.py").write_text(agent.agent_code)
        
        # Save tools
        (agent_dir / "agent_tools.py").write_text(agent.tools_code)
        
        # Save prompts
        (agent_dir / "agent_prompts.py").write_text(agent.prompts_code)
        
        # Save config
        (agent_dir / "config.yaml").write_text(yaml.dump(agent.config))
        
        # Save requirements
        (agent_dir / "requirements.txt").write_text('\n'.join(agent.requirements))
        
        # Save tests
        (agent_dir / "test_agent.py").write_text(agent.tests)
        
        # Save documentation
        (agent_dir / "README.md").write_text(agent.documentation)
        
        logfire.info(f"Agent saved to {agent_dir}")


# Example usage
async def example_usage():
    """Example of using the agent generator"""
    generator = AgentGenerator()
    
    requirements = AgentRequirements(
        task_description="Monitor GitHub repositories for security vulnerabilities",
        capabilities=[
            "Access GitHub API",
            "Analyze code for vulnerabilities",
            "Generate security reports",
            "Send notifications"
        ],
        tools_needed=[
            "github_api",
            "security_scanner",
            "report_generator",
            "notification_sender"
        ],
        model_preference="openai:gpt-4",
        examples=[
            "Scan repo for SQL injection vulnerabilities",
            "Check for exposed API keys",
            "Monitor dependency vulnerabilities"
        ]
    )
    
    # Generate agent
    agent = await generator.generate_agent(
        requirements,
        output_dir=Path("generated_agents")
    )
    
    print(f"Generated agent: {agent.name}")
    print(f"Files saved to: generated_agents/{agent.name.lower()}/")


if __name__ == "__main__":
    import asyncio
    asyncio.run(example_usage())