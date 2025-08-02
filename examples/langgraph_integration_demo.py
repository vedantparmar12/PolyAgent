"""Demo of LangGraph integration with ReAct agents and memory

This example demonstrates:
1. LangGraph orchestration with state persistence
2. ReAct pattern for agent reasoning
3. Long-term memory integration
4. Parallel agent execution
5. PydanticAI structured outputs
"""

import asyncio
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langgraph_orchestrator import LangGraphOrchestrator
from src.agents.react_agent import ReActAgent, ReActDependencies
from src.workflow.memory import ConversationMemory
from src.agents.base_agent import MultiAgentCoordinator
from pydantic import BaseModel
from typing import List, Dict, Any
import yaml


class DemoConfig(BaseModel):
    """Configuration for the demo"""
    enable_visualization: bool = True
    enable_streaming: bool = True
    thread_id: str = "demo_thread"
    max_parallel_agents: int = 4


async def visualize_workflow(orchestrator: LangGraphOrchestrator):
    """Visualize the LangGraph workflow"""
    try:
        from IPython.display import Image, display
        graph = orchestrator.get_workflow_graph()
        display(Image(graph.draw_mermaid_png()))
    except:
        print("\n📊 Workflow Graph Structure:")
        graph = orchestrator.app.get_graph()
        print(f"Nodes: {list(graph.nodes.keys())}")
        print(f"Edges: {graph.edges}")


async def demo_simple_task(orchestrator: LangGraphOrchestrator):
    """Demo simple task handling"""
    print("\n" + "="*60)
    print("🔹 DEMO 1: Simple Task Handling")
    print("="*60)
    
    query = "What is the capital of France?"
    print(f"\n📝 Query: {query}")
    
    response = await orchestrator.run(query, "demo_simple")
    print(f"\n✅ Response: {response}")


async def demo_complex_task(orchestrator: LangGraphOrchestrator):
    """Demo complex task with parallel agents"""
    print("\n" + "="*60)
    print("🔹 DEMO 2: Complex Task with Parallel Agents")
    print("="*60)
    
    query = """Create a Python REST API with the following requirements:
    1. User authentication with JWT
    2. CRUD operations for a blog post model
    3. Rate limiting and caching
    4. Comprehensive error handling
    5. Unit tests with pytest"""
    
    print(f"\n📝 Query: {query[:100]}...")
    
    response = await orchestrator.run(query, "demo_complex")
    print(f"\n✅ Response:\n{response}")


async def demo_react_agent():
    """Demo ReAct agent with reasoning"""
    print("\n" + "="*60)
    print("🔹 DEMO 3: ReAct Agent with Reasoning")
    print("="*60)
    
    # Create ReAct agent
    agent = ReActAgent()
    
    # Create dependencies with mock tools
    deps = ReActDependencies(
        user_id="demo_user",
        session_id="demo_session",
        max_iterations=5,
        available_tools={
            "search": lambda q: f"Found information about: {q}. Python async uses event loops for concurrency, while threading uses OS threads.",
            "analyze": lambda d: f"Analysis shows: {d} involves different concurrency models with distinct performance characteristics."
        }
    )
    
    query = "Compare Python async/await with threading for web scraping"
    print(f"\n📝 Query: {query}")
    
    result = await agent.run(query, deps)
    
    print(f"\n🧠 Thought Process:")
    for i, thought_action in enumerate(result.thought_process, 1):
        print(f"\n  Step {i}:")
        print(f"  💭 Thought: {thought_action.thought}")
        print(f"  🎯 Action: {thought_action.action}")
        print(f"  👀 Observation: {thought_action.observation[:100]}...")
    
    print(f"\n✅ Final Answer: {result.final_answer}")
    print(f"📊 Confidence: {result.confidence:.2%}")
    print(f"🛠️  Tools Used: {', '.join(result.tools_used)}")


async def demo_memory_integration(orchestrator: LangGraphOrchestrator):
    """Demo memory integration and context awareness"""
    print("\n" + "="*60)
    print("🔹 DEMO 4: Memory Integration")
    print("="*60)
    
    thread_id = "memory_demo"
    
    # First query - establish context
    query1 = "My project uses FastAPI with PostgreSQL. What's the best way to handle database connections?"
    print(f"\n📝 Query 1: {query1}")
    
    response1 = await orchestrator.run(query1, thread_id)
    print(f"\n✅ Response 1: {response1[:200]}...")
    
    # Second query - should use context from first
    query2 = "How should I structure the models for this setup?"
    print(f"\n📝 Query 2: {query2}")
    
    response2 = await orchestrator.run(query2, thread_id)
    print(f"\n✅ Response 2: {response2[:200]}...")
    
    # Show memory retrieval
    memory = orchestrator.memory
    conversation_id = memory._generate_conversation_id(thread_id, None)
    history = memory.get_conversation_history(conversation_id, limit=4)
    
    print(f"\n📚 Conversation History:")
    for msg in history:
        print(f"  - {msg['role']}: {msg['content'][:100]}...")


async def demo_multi_agent_coordination():
    """Demo multi-agent coordination with PydanticAI"""
    print("\n" + "="*60)
    print("🔹 DEMO 5: Multi-Agent Coordination")
    print("="*60)
    
    # Create coordinator
    coordinator = MultiAgentCoordinator(max_parallel=3)
    
    # Register different agents
    react_agent = ReActAgent(model="openai:gpt-3.5-turbo")
    coordinator.register_agent("analyzer", react_agent)
    coordinator.register_agent("validator", react_agent)
    coordinator.register_agent("synthesizer", react_agent)
    
    # Define tasks for each agent
    agent_tasks = [
        {
            "agent_name": "analyzer",
            "prompt": "Analyze the pros and cons of microservices architecture",
            "deps": ReActDependencies(
                user_id="demo",
                session_id="multi_agent_demo",
                max_iterations=3
            )
        },
        {
            "agent_name": "validator",
            "prompt": "Validate common misconceptions about microservices",
            "deps": ReActDependencies(
                user_id="demo",
                session_id="multi_agent_demo",
                max_iterations=3
            )
        },
        {
            "agent_name": "synthesizer",
            "prompt": "Synthesize best practices for microservices adoption",
            "deps": ReActDependencies(
                user_id="demo",
                session_id="multi_agent_demo",
                max_iterations=3
            )
        }
    ]
    
    print("\n🚀 Running 3 agents in parallel...")
    
    results = await coordinator.run_agents(agent_tasks, parallel=True)
    
    print("\n📊 Results:")
    for agent_name, result in results.items():
        if result['success']:
            output = result['data']
            print(f"\n  {agent_name.upper()}:")
            print(f"    Task: {output.task}")
            print(f"    Confidence: {output.confidence:.2%}")
            print(f"    Tools: {', '.join(output.tools_used)}")
            print(f"    Answer: {output.final_answer[:150]}...")
        else:
            print(f"\n  {agent_name.upper()}: ❌ {result['error']}")


async def main():
    """Main demo function"""
    print("""
    🚀 LangGraph Multi-Agent System Demo
    ===================================
    
    This demo showcases:
    ✅ LangGraph orchestration with state management
    ✅ ReAct pattern for agent reasoning
    ✅ Long-term memory with SQLite persistence
    ✅ Parallel agent execution
    ✅ PydanticAI structured outputs
    """)
    
    # Initialize orchestrator
    print("🔧 Initializing LangGraph orchestrator...")
    orchestrator = LangGraphOrchestrator()
    
    # Visualize workflow
    await visualize_workflow(orchestrator)
    
    # Run demos
    demos = [
        ("Simple Task", demo_simple_task),
        ("Complex Task", demo_complex_task),
        ("ReAct Agent", demo_react_agent),
        ("Memory Integration", demo_memory_integration),
        ("Multi-Agent Coordination", demo_multi_agent_coordination)
    ]
    
    for name, demo_func in demos:
        try:
            if demo_func.__name__ == "demo_react_agent" or demo_func.__name__ == "demo_multi_agent_coordination":
                await demo_func()
            else:
                await demo_func(orchestrator)
        except Exception as e:
            print(f"\n❌ Error in {name}: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*60)
    print("✅ Demo completed successfully!")
    print("="*60)


if __name__ == "__main__":
    # Create necessary directories
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("memory", exist_ok=True)
    
    # Run the demo
    asyncio.run(main())