"""Unified LangGraph-based orchestrator for multi-agent workflows"""

import json
import yaml
import asyncio
from typing import TypedDict, List, Dict, Any, Annotated, Optional
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.prebuilt import ToolExecutor
import operator
from agent import OpenRouterAgent
from src.workflow.memory import ConversationMemory, WorkingMemory


class AgentState(TypedDict):
    """Enhanced state for LangGraph workflow"""
    # Core state
    user_input: str
    messages: Annotated[List[str], operator.add]
    
    # Task decomposition
    task_complexity: str
    subtasks: List[str]
    
    # Agent results
    agent_outputs: Dict[str, Any]
    agent_errors: Dict[str, str]
    
    # Memory and context
    conversation_id: str
    working_memory: List[Dict[str, Any]]
    relevant_memories: List[Dict[str, Any]]
    
    # Workflow control
    current_step: str
    retry_count: int
    should_continue: bool
    
    # Final output
    final_response: str
    synthesis_complete: bool


class LangGraphOrchestrator:
    """Orchestrator using LangGraph for efficient multi-agent coordination"""
    
    def __init__(self, config_path="config.yaml"):
        """Initialize the LangGraph orchestrator
        
        Args:
            config_path: Path to configuration file
        """
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Initialize components
        self.memory = ConversationMemory()
        self.working_memory = WorkingMemory(capacity=20)
        
        # Initialize checkpointer
        self.checkpointer = SqliteSaver.from_conn_string("checkpoints/orchestrator.db")
        
        # Build the graph
        self._build_graph()
    
    def _build_graph(self):
        """Build the LangGraph state graph"""
        # Create the graph
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("analyze_complexity", self.analyze_complexity)
        workflow.add_node("decompose_task", self.decompose_task)
        workflow.add_node("load_memory", self.load_relevant_memory)
        workflow.add_node("route_simple", self.handle_simple_task)
        workflow.add_node("route_complex", self.handle_complex_task)
        workflow.add_node("validate_outputs", self.validate_outputs)
        workflow.add_node("synthesize", self.synthesize_results)
        workflow.add_node("save_memory", self.save_to_memory)
        
        # Add conditional edges
        workflow.add_conditional_edges(
            "analyze_complexity",
            self.route_by_complexity,
            {
                "simple": "route_simple",
                "complex": "decompose_task"
            }
        )
        
        workflow.add_edge("decompose_task", "load_memory")
        workflow.add_edge("load_memory", "route_complex")
        workflow.add_edge("route_simple", "validate_outputs")
        workflow.add_edge("route_complex", "validate_outputs")
        
        workflow.add_conditional_edges(
            "validate_outputs",
            self.check_validation,
            {
                "retry": "route_complex",
                "success": "synthesize"
            }
        )
        
        workflow.add_edge("synthesize", "save_memory")
        workflow.add_edge("save_memory", END)
        
        # Set entry point
        workflow.set_entry_point("analyze_complexity")
        
        # Compile with checkpointer
        self.app = workflow.compile(checkpointer=self.checkpointer)
    
    async def analyze_complexity(self, state: AgentState) -> AgentState:
        """Analyze task complexity using ReAct-style reasoning"""
        # Create a reasoning agent
        agent = OpenRouterAgent(silent=True)
        
        prompt = f"""Analyze the complexity of this task and provide reasoning:

Task: {state['user_input']}

Think step by step:
1. What type of task is this? (simple query, complex implementation, research, etc.)
2. How many sub-components or steps might this require?
3. Would this benefit from multiple specialized agents?

Respond with:
- COMPLEXITY: simple OR complex
- REASONING: Your step-by-step analysis
- SUGGESTED_APPROACH: Brief recommendation"""

        response = agent.run(prompt)
        
        # Parse complexity
        complexity = "simple"
        if "COMPLEXITY: complex" in response.lower():
            complexity = "complex"
        
        state['task_complexity'] = complexity
        state['messages'].append(f"Task analyzed as: {complexity}")
        state['messages'].append(f"Analysis: {response}")
        
        return state
    
    def route_by_complexity(self, state: AgentState) -> str:
        """Route based on task complexity"""
        return state['task_complexity']
    
    async def decompose_task(self, state: AgentState) -> AgentState:
        """Decompose complex task into subtasks"""
        agent = OpenRouterAgent(silent=True)
        
        prompt = f"""Decompose this complex task into 3-5 focused subtasks:

Task: {state['user_input']}

Previous analysis: {state['messages'][-1] if state['messages'] else 'None'}

Create specific, actionable subtasks that can be handled by specialized agents.
Return as JSON array of strings."""

        response = agent.run(prompt)
        
        try:
            # Extract JSON from response
            import re
            json_match = re.search(r'\[.*\]', response, re.DOTALL)
            if json_match:
                subtasks = json.loads(json_match.group())
            else:
                # Fallback decomposition
                subtasks = [
                    f"Research and analyze: {state['user_input']}",
                    f"Identify key components for: {state['user_input']}",
                    f"Develop solution approach for: {state['user_input']}",
                    f"Validate and refine: {state['user_input']}"
                ]
        except:
            subtasks = [state['user_input']]  # Fallback to original
        
        state['subtasks'] = subtasks
        state['messages'].append(f"Decomposed into {len(subtasks)} subtasks")
        
        return state
    
    async def load_relevant_memory(self, state: AgentState) -> AgentState:
        """Load relevant memories for context"""
        # Get relevant memories
        memories = self.memory.get_relevant_memories(
            state['conversation_id'],
            query=state['user_input'],
            entry_types=['fact', 'insight', 'summary'],
            limit=5
        )
        
        state['relevant_memories'] = memories
        
        # Add to working memory
        self.working_memory.add({
            'type': 'task_start',
            'task': state['user_input'],
            'subtasks': state.get('subtasks', [])
        })
        
        state['working_memory'] = self.working_memory.get_recent(10)
        state['messages'].append(f"Loaded {len(memories)} relevant memories")
        
        return state
    
    async def handle_simple_task(self, state: AgentState) -> AgentState:
        """Handle simple task with single agent"""
        agent = OpenRouterAgent(silent=True, context_aware=True)
        
        # Add memory context to prompt
        context = ""
        if state.get('relevant_memories'):
            context = "\n\nRelevant context from memory:\n"
            for mem in state['relevant_memories']:
                context += f"- {mem['content']}\n"
        
        full_prompt = state['user_input'] + context
        
        try:
            response = agent.run(full_prompt)
            state['agent_outputs']['main'] = response
            state['messages'].append("Simple task completed successfully")
        except Exception as e:
            state['agent_errors']['main'] = str(e)
            state['messages'].append(f"Error in simple task: {e}")
        
        return state
    
    async def handle_complex_task(self, state: AgentState) -> AgentState:
        """Handle complex task with multiple parallel agents"""
        subtasks = state.get('subtasks', [state['user_input']])
        
        # Create specialized agents for each subtask
        agents = []
        for i, subtask in enumerate(subtasks):
            agent = OpenRouterAgent(silent=True, context_aware=True)
            agents.append((f"agent_{i}", agent, subtask))
        
        # Add memory context
        context = ""
        if state.get('relevant_memories'):
            context = "\n\nRelevant context:\n"
            for mem in state['relevant_memories']:
                context += f"- {mem['content']}\n"
        
        # Run agents in parallel
        async def run_agent(agent_id, agent, task):
            try:
                result = await asyncio.to_thread(agent.run, task + context)
                return agent_id, result, None
            except Exception as e:
                return agent_id, None, str(e)
        
        # Execute all agents concurrently
        tasks = [run_agent(aid, agent, task) for aid, agent, task in agents]
        results = await asyncio.gather(*tasks)
        
        # Process results
        for agent_id, result, error in results:
            if error:
                state['agent_errors'][agent_id] = error
            else:
                state['agent_outputs'][agent_id] = result
        
        state['messages'].append(f"Completed {len(results)} parallel agents")
        
        # Update working memory
        self.working_memory.add({
            'type': 'agents_completed',
            'count': len(results),
            'success': len(state['agent_outputs']),
            'errors': len(state['agent_errors'])
        })
        
        return state
    
    async def validate_outputs(self, state: AgentState) -> AgentState:
        """Validate agent outputs using ReAct-style checking"""
        if not state['agent_outputs']:
            state['should_continue'] = False
            return state
        
        # Create validation agent
        agent = OpenRouterAgent(silent=True)
        
        outputs_text = "\n\n".join([
            f"=== {agent_id} ===\n{output}"
            for agent_id, output in state['agent_outputs'].items()
        ])
        
        prompt = f"""Validate these agent outputs for the task:

Original Task: {state['user_input']}

Agent Outputs:
{outputs_text}

Check for:
1. Completeness - Do the outputs fully address the task?
2. Accuracy - Are there any contradictions or errors?
3. Quality - Is the output high quality and useful?

Respond with:
- VALIDATION: pass OR retry
- ISSUES: List any issues found
- SUGGESTIONS: How to improve if retry needed"""

        response = agent.run(prompt)
        
        # Parse validation result
        if "VALIDATION: pass" in response:
            state['should_continue'] = False
            state['messages'].append("Validation passed")
        else:
            state['retry_count'] = state.get('retry_count', 0) + 1
            if state['retry_count'] >= 3:
                state['should_continue'] = False
                state['messages'].append("Max retries reached, proceeding with current outputs")
            else:
                state['should_continue'] = True
                state['messages'].append(f"Validation failed, retry {state['retry_count']}")
        
        return state
    
    def check_validation(self, state: AgentState) -> str:
        """Check if validation passed or needs retry"""
        if state.get('should_continue', False) and state.get('retry_count', 0) < 3:
            return "retry"
        return "success"
    
    async def synthesize_results(self, state: AgentState) -> AgentState:
        """Synthesize final response using ReAct-style reasoning"""
        agent = OpenRouterAgent(silent=True)
        
        # Prepare outputs for synthesis
        outputs_text = "\n\n".join([
            f"=== Agent {agent_id} ===\n{output}"
            for agent_id, output in state['agent_outputs'].items()
        ])
        
        # Include memory context
        memory_context = ""
        if state.get('relevant_memories'):
            memory_context = "\n\nHistorical context:\n"
            for mem in state['relevant_memories']:
                memory_context += f"- {mem['content']}\n"
        
        prompt = f"""Synthesize a comprehensive response from these agent outputs:

Original Task: {state['user_input']}
{memory_context}

Agent Outputs:
{outputs_text}

Create a unified, coherent response that:
1. Combines insights from all agents
2. Resolves any contradictions
3. Provides a complete answer to the original task
4. Highlights key findings and recommendations

Use clear formatting and structure."""

        response = agent.run(prompt)
        
        state['final_response'] = response
        state['synthesis_complete'] = True
        state['messages'].append("Synthesis completed")
        
        return state
    
    async def save_to_memory(self, state: AgentState) -> AgentState:
        """Save important information to long-term memory"""
        # Save the conversation
        self.memory.add_message(
            state['conversation_id'],
            'user',
            state['user_input']
        )
        
        self.memory.add_message(
            state['conversation_id'],
            'assistant',
            state['final_response']
        )
        
        # Extract and save facts using agent
        agent = OpenRouterAgent(silent=True)
        
        prompt = f"""Extract 3-5 important facts or insights from this conversation:

User Query: {state['user_input']}

Response: {state['final_response']}

Return as JSON array of strings, each being a concise fact or insight."""

        try:
            response = agent.run(prompt)
            import re
            json_match = re.search(r'\[.*\]', response, re.DOTALL)
            if json_match:
                facts = json.loads(json_match.group())
                self.memory.extract_facts(state['conversation_id'], facts)
        except:
            pass  # Fact extraction is optional
        
        # Create summary
        if len(state['final_response']) > 500:
            summary = f"Task: {state['user_input'][:100]}... Key points: {state['final_response'][:200]}..."
            self.memory.summarize_conversation(
                state['conversation_id'],
                summary,
                importance=0.7
            )
        
        state['messages'].append("Saved to memory")
        
        return state
    
    async def run(self, user_input: str, thread_id: str = "default") -> str:
        """Run the orchestrated workflow
        
        Args:
            user_input: The user's input/query
            thread_id: Thread ID for conversation continuity
            
        Returns:
            The final synthesized response
        """
        # Create conversation
        conversation_id = self.memory.create_conversation(thread_id)
        
        # Initialize state
        initial_state: AgentState = {
            'user_input': user_input,
            'messages': [],
            'task_complexity': '',
            'subtasks': [],
            'agent_outputs': {},
            'agent_errors': {},
            'conversation_id': conversation_id,
            'working_memory': [],
            'relevant_memories': [],
            'current_step': 'start',
            'retry_count': 0,
            'should_continue': True,
            'final_response': '',
            'synthesis_complete': False
        }
        
        # Run the workflow
        config = {"configurable": {"thread_id": thread_id}}
        
        # Stream events for real-time updates
        async for event in self.app.astream_events(initial_state, config, version="v1"):
            if event["event"] == "on_chain_start":
                print(f"\n🔄 Starting: {event['name']}")
            elif event["event"] == "on_chain_end":
                print(f"✅ Completed: {event['name']}")
        
        # Get final state
        final_state = await self.app.ainvoke(initial_state, config)
        
        return final_state['final_response']
    
    def get_conversation_history(self, conversation_id: str) -> List[Dict[str, Any]]:
        """Get conversation history"""
        return self.memory.get_conversation_history(conversation_id)
    
    def get_workflow_graph(self):
        """Get the workflow graph for visualization"""
        return self.app.get_graph()


# Example usage with CLI interface
async def main():
    """Example usage of LangGraph orchestrator"""
    print("🚀 LangGraph Multi-Agent Orchestrator")
    print("=" * 50)
    
    orchestrator = LangGraphOrchestrator()
    
    # Example queries
    examples = [
        "What is the weather like today?",
        "Implement a Python web scraper with error handling and rate limiting",
        "Research and compare the top 5 JavaScript frameworks for building modern web applications"
    ]
    
    print("\nExample queries:")
    for i, example in enumerate(examples, 1):
        print(f"{i}. {example}")
    
    while True:
        user_input = input("\n💭 Enter your query (or 'quit' to exit): ").strip()
        
        if user_input.lower() in ['quit', 'exit', 'bye']:
            print("👋 Goodbye!")
            break
        
        if not user_input:
            continue
        
        print("\n🤖 Processing with LangGraph orchestrator...")
        
        try:
            response = await orchestrator.run(user_input)
            print(f"\n📝 Response:\n{response}")
        except Exception as e:
            print(f"\n❌ Error: {e}")


if __name__ == "__main__":
    asyncio.run(main())