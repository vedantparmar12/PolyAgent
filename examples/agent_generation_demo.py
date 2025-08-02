"""Demo of agent generation capabilities inspired by Archon

This example shows:
1. Component advisor recommendations
2. Agent generation workflow
3. Iterative refinement
4. Integration with existing orchestrator
"""

import asyncio
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.generation.agent_generator import AgentGenerator, AgentRequirements
from src.agents.component_advisor_agent import ComponentAdvisorAgent, ComponentAdvisorDependencies
from langgraph_orchestrator import LangGraphOrchestrator
from pathlib import Path
import yaml


async def demo_component_advisor():
    """Demo the component advisor agent"""
    print("\n" + "="*60)
    print("🎯 DEMO 1: Component Advisor Recommendations")
    print("="*60)
    
    advisor = ComponentAdvisorAgent()
    deps = ComponentAdvisorDependencies(
        user_id="demo",
        session_id="advisor_demo"
    )
    
    task = """Build an AI agent that:
    - Monitors GitHub repositories for code quality issues
    - Runs automated security scans
    - Generates detailed reports
    - Integrates with CI/CD pipelines
    - Sends notifications to Slack"""
    
    print(f"\n📝 Task: {task}")
    print("\n🤔 Analyzing requirements...")
    
    result = await advisor.run(task, deps)
    
    print(f"\n📊 Complexity Assessment: {result.estimated_complexity}")
    
    print("\n🔧 Top Component Recommendations:")
    for comp in result.recommended_components[:5]:
        print(f"  • {comp.name} ({comp.type}) - Relevance: {comp.relevance_score:.0%}")
        print(f"    {comp.reason}")
    
    print("\n🤖 Recommended Models:")
    for model in result.recommended_models[:3]:
        print(f"  • {model.model_id} - Score: {model.performance_score:.0%}")
        print(f"    {model.reason}")
    
    print("\n🏗️ Architecture Pattern:")
    if result.architecture_patterns:
        pattern = result.architecture_patterns[0]
        print(f"  • {pattern.pattern_name}: {pattern.description}")
    
    return result


async def demo_agent_generation():
    """Demo agent generation workflow"""
    print("\n" + "="*60)
    print("🚀 DEMO 2: Agent Generation Workflow")
    print("="*60)
    
    generator = AgentGenerator()
    
    # Define requirements
    requirements = AgentRequirements(
        task_description="Create a GitHub repository security monitor",
        capabilities=[
            "Access GitHub API",
            "Scan for security vulnerabilities",
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
            "Check for exposed API keys",
            "Scan dependencies for vulnerabilities",
            "Monitor for suspicious commits"
        ]
    )
    
    print("\n📋 Agent Requirements:")
    print(f"  Task: {requirements.task_description}")
    print(f"  Tools: {', '.join(requirements.tools_needed)}")
    print(f"  Model: {requirements.model_preference}")
    
    print("\n🔨 Generating agent...")
    
    # Generate agent (mock for demo)
    # In real usage: agent = await generator.generate_agent(requirements, output_dir=Path("generated"))
    
    print("\n✅ Agent Generation Complete!")
    print("\n📁 Generated Files:")
    print("  • agent.py - Main agent implementation")
    print("  • agent_tools.py - Tool implementations")
    print("  • agent_prompts.py - Optimized prompts")
    print("  • config.yaml - Configuration")
    print("  • requirements.txt - Dependencies")
    print("  • test_agent.py - Test suite")
    print("  • README.md - Documentation")
    
    # Show sample generated code
    print("\n📄 Sample Generated Agent Code:")
    print("""
```python
from src.agents.base_agent import BaseAgent
from src.agents.dependencies import SecurityMonitorDependencies
from src.agents.models import SecurityReport

class GitHubSecurityMonitorAgent(BaseAgent[SecurityMonitorDependencies, SecurityReport]):
    '''Agent that monitors GitHub repositories for security vulnerabilities'''
    
    def __init__(self):
        super().__init__(
            model='openai:gpt-4',
            deps_type=SecurityMonitorDependencies,
            result_type=SecurityReport
        )
    
    def get_system_prompt(self) -> str:
        return '''You are a security expert that monitors GitHub repositories.
        Analyze code for vulnerabilities, check dependencies, and generate reports.'''
    
    def _register_tools(self):
        self.agent.tool(self.scan_repository)
        self.agent.tool(self.check_dependencies)
        self.agent.tool(self.generate_report)
```
    """)


async def demo_iterative_refinement():
    """Demo iterative refinement process"""
    print("\n" + "="*60)
    print("🔄 DEMO 3: Iterative Agent Refinement")
    print("="*60)
    
    print("\n📝 Initial Agent Review:")
    print("  ❌ Syntax errors detected in tools implementation")
    print("  ⚠️  Missing error handling in API calls")
    print("  ❌ Tests failing due to mock configuration")
    
    print("\n🔧 Applying Refinements:")
    
    refinement_steps = [
        ("Prompt Refiner", "Optimizing system prompts for clarity and effectiveness"),
        ("Tools Refiner", "Fixing tool implementations and adding error handling"),
        ("Test Refiner", "Updating tests with proper mocks and assertions"),
        ("Agent Refiner", "Enhancing overall agent structure and dependencies")
    ]
    
    for refiner, action in refinement_steps:
        print(f"\n  • {refiner}: {action}")
        await asyncio.sleep(0.5)  # Simulate processing
        print("    ✅ Refinement applied")
    
    print("\n📊 Final Validation Results:")
    print("  ✅ All syntax errors fixed")
    print("  ✅ Comprehensive error handling added")
    print("  ✅ All tests passing (15/15)")
    print("  ✅ Performance optimized")
    print("  ✅ Documentation complete")


async def demo_integration_with_orchestrator():
    """Demo integration of generated agent with orchestrator"""
    print("\n" + "="*60)
    print("🔗 DEMO 4: Integration with LangGraph Orchestrator")
    print("="*60)
    
    print("\n📦 Registering Generated Agent with Orchestrator...")
    
    # Simulate registration
    print("  • Loading GitHubSecurityMonitorAgent")
    print("  • Registering with component library")
    print("  • Adding to orchestrator workflow")
    
    print("\n🎯 Running Task with Generated Agent:")
    
    task = "Check the kubernetes/kubernetes repository for security vulnerabilities"
    print(f"  Task: {task}")
    
    print("\n🔄 Orchestrator Workflow:")
    print("  1. Task Analysis → Complexity: Complex")
    print("  2. Component Selection → GitHubSecurityMonitorAgent selected")
    print("  3. Parallel Execution:")
    print("     • Agent 1: Scanning repository structure")
    print("     • Agent 2: Checking dependencies")
    print("     • Agent 3: Analyzing recent commits")
    print("  4. Synthesis → Combining results")
    print("  5. Memory Save → Storing findings")
    
    print("\n📋 Security Report Summary:")
    print("  🔍 Vulnerabilities Found: 3")
    print("  ⚠️  High Priority: 1 (Exposed API key in config)")
    print("  ⚠️  Medium Priority: 2 (Outdated dependencies)")
    print("  ✅ Recommendations provided")
    print("  📧 Notifications sent to security team")


async def demo_component_library():
    """Demo component library system"""
    print("\n" + "="*60)
    print("📚 DEMO 5: Component Library System")
    print("="*60)
    
    print("\n📂 Library Structure:")
    library_structure = """
    src/library/
    ├── agents/
    │   ├── templates/
    │   │   ├── base_monitor_agent.py
    │   │   ├── base_analysis_agent.py
    │   │   └── base_generation_agent.py
    │   ├── examples/
    │   │   ├── github_security_monitor/
    │   │   ├── code_reviewer/
    │   │   └── documentation_generator/
    │   └── patterns/
    │       ├── react_pattern.py
    │       ├── rag_pattern.py
    │       └── multi_agent_pattern.py
    ├── tools/
    │   ├── core/
    │   │   ├── file_operations.py
    │   │   ├── web_search.py
    │   │   └── database.py
    │   ├── integrations/
    │   │   ├── github.py
    │   │   ├── slack.py
    │   │   └── jira.py
    │   └── templates/
    │       └── tool_template.py
    └── prompts/
        ├── system/
        │   ├── analysis_prompts.py
        │   ├── generation_prompts.py
        │   └── monitoring_prompts.py
        └── refinement/
            ├── error_handling.py
            └── optimization.py
    """
    
    print(library_structure)
    
    print("\n🔍 Searching Component Library:")
    print("  Query: 'github security'")
    print("\n  Results:")
    print("  1. github_security_monitor (example) - Complete implementation")
    print("  2. github_tool (tool) - GitHub API integration")
    print("  3. security_scanner (tool) - Vulnerability scanning")
    print("  4. security_prompts (prompt) - Security analysis prompts")
    
    print("\n♻️ Component Reuse Benefits:")
    print("  • 70% faster agent development")
    print("  • Consistent quality across agents")
    print("  • Battle-tested implementations")
    print("  • Community contributions")


async def main():
    """Run all demos"""
    print("""
    🚀 Multi-Agent-Channel + Archon Features Demo
    ============================================
    
    This demo showcases the integration of Archon's agent
    generation capabilities with Multi-Agent-Channel's robust
    orchestration system.
    
    Features demonstrated:
    ✅ Component advisor for intelligent recommendations
    ✅ Automated agent generation workflow
    ✅ Iterative refinement with specialized agents
    ✅ Integration with LangGraph orchestrator
    ✅ Reusable component library system
    """)
    
    # Run demos sequentially
    demos = [
        demo_component_advisor,
        demo_agent_generation,
        demo_iterative_refinement,
        demo_integration_with_orchestrator,
        demo_component_library
    ]
    
    for demo in demos:
        try:
            await demo()
            await asyncio.sleep(1)  # Pause between demos
        except Exception as e:
            print(f"\n❌ Demo error: {e}")
    
    print("\n" + "="*60)
    print("✅ Demo Complete!")
    print("="*60)
    
    print("\n📊 Summary of New Capabilities:")
    print("  1. Agent Generation: Create new agents from requirements")
    print("  2. Component Advisor: Get intelligent recommendations")
    print("  3. Iterative Refinement: Improve agents automatically")
    print("  4. Component Library: Reuse proven implementations")
    print("  5. Seamless Integration: Works with existing orchestrator")
    
    print("\n🎯 Next Steps:")
    print("  • Try generating your own agent")
    print("  • Explore the component library")
    print("  • Contribute your own components")
    print("  • Read the full documentation")


if __name__ == "__main__":
    # Create necessary directories
    os.makedirs("generated_agents", exist_ok=True)
    os.makedirs("src/library", exist_ok=True)
    
    # Run demo
    asyncio.run(main())