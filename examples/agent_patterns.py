"""Agent creation and usage patterns for Make It Heavy."""

from typing import Dict, Any
from agent import OpenRouterAgent

# Pattern 1: Creating a basic agent
def create_basic_agent():
    """Create a simple agent for single-task execution."""
    agent = OpenRouterAgent(
        config_path="config.yaml",
        silent=False  # Show progress for debugging
    )
    return agent

# Pattern 2: Creating a context-aware agent
def create_context_aware_agent():
    """Create an agent that loads and uses project context."""
    agent = OpenRouterAgent(
        config_path="config.yaml",
        silent=False,
        context_aware=True  # Enable context loading
    )
    
    # The agent automatically loads CLAUDE.md, PLANNING.md, etc.
    # Context is injected into system prompts
    return agent

# Pattern 3: Creating a silent agent for orchestration
def create_orchestration_agent():
    """Create a silent agent for use in multi-agent systems."""
    agent = OpenRouterAgent(
        config_path="config.yaml",
        silent=True,  # Suppress output for cleaner orchestration
        context_aware=True
    )
    return agent

# Pattern 4: Using an agent with error handling
def use_agent_safely(user_input: str) -> Dict[str, Any]:
    """Demonstrate proper error handling when using agents."""
    try:
        agent = create_context_aware_agent()
        
        # Run agent with timeout consideration
        response = agent.run(user_input)
        
        return {
            "status": "success",
            "response": response
        }
        
    except Exception as e:
        # Specific error handling
        if "api_key" in str(e).lower():
            return {
                "status": "error",
                "error": "API key not configured. Check config.yaml"
            }
        elif "rate limit" in str(e).lower():
            return {
                "status": "error", 
                "error": "Rate limit exceeded. Please wait and retry."
            }
        else:
            return {
                "status": "error",
                "error": f"Agent execution failed: {str(e)}"
            }

# Pattern 5: Custom agent with specific tools
def create_research_agent():
    """Create an agent optimized for research tasks."""
    agent = OpenRouterAgent(
        config_path="config.yaml",
        context_aware=True
    )
    
    # The agent automatically discovers tools
    # But you can check what's available
    available_tools = list(agent.tool_mapping.keys())
    print(f"Research agent has access to: {available_tools}")
    
    return agent

# Pattern 6: Agent with custom system prompt
def create_specialized_agent(specialty: str):
    """Create an agent with a specialized focus.
    
    Note: This modifies the config at runtime - use sparingly.
    """
    import yaml
    
    # Load config
    with open("config.yaml", 'r') as f:
        config = yaml.safe_load(f)
    
    # Enhance system prompt
    original_prompt = config['system_prompt']
    config['system_prompt'] = f"{original_prompt}\n\nSpecialty: You are an expert in {specialty}."
    
    # Save to temporary config
    temp_config = "temp_config.yaml"
    with open(temp_config, 'w') as f:
        yaml.dump(config, f)
    
    # Create agent with modified config
    agent = OpenRouterAgent(
        config_path=temp_config,
        context_aware=True
    )
    
    return agent

# Pattern 7: Streaming responses (for future enhancement)
def use_agent_with_streaming(user_input: str):
    """Demonstrate how to handle streaming responses.
    
    Note: Current implementation doesn't support streaming,
    but this shows the pattern for future enhancement.
    """
    agent = create_context_aware_agent()
    
    # Future pattern for streaming
    # for chunk in agent.run_streaming(user_input):
    #     print(chunk, end='', flush=True)
    
    # Current implementation
    response = agent.run(user_input)
    
    # Simulate streaming by printing progressively
    words = response.split()
    for i in range(0, len(words), 10):
        chunk = ' '.join(words[i:i+10])
        print(chunk)

# Anti-patterns to avoid
def anti_pattern_examples():
    """Examples of what NOT to do."""
    
    # ❌ Don't create agents in loops without need
    # for i in range(100):
    #     agent = OpenRouterAgent()  # Wasteful
    
    # ❌ Don't ignore context when available
    # agent = OpenRouterAgent(context_aware=False)  # Missing benefits
    
    # ❌ Don't catch all exceptions blindly
    # try:
    #     agent.run("task")
    # except:  # Too broad
    #     pass
    
    # ❌ Don't modify global config
    # OpenRouterAgent.config['api_key'] = 'new_key'  # Side effects
    
    pass

if __name__ == "__main__":
    # Example usage
    print("Creating context-aware agent...")
    agent = create_context_aware_agent()
    
    print("\nTesting agent with simple query...")
    result = use_agent_safely("What project rules should I follow?")
    print(f"Result: {result['status']}")
    
    if result['status'] == 'success':
        print(f"Agent says: {result['response'][:200]}...")