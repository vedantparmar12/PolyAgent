"""Example demonstrating multi-model support in the Enhanced Agentic Workflow"""

import asyncio
from src.agents.base_agent import BaseAgent
from src.agents.code_generator import CodeGeneratorAgent
from src.agents.research_agent import ResearchAgent
from src.core.model_provider import ModelProvider, ModelConfig, ModelRegistry
from src.ui.components.model_selector import ModelSelectorUI
import streamlit as st


async def demonstrate_multi_model_support():
    """Demonstrate using different models for different tasks"""
    
    # Configure model settings with API keys
    model_config = ModelConfig(
        # You can set these via environment variables or directly
        openrouter_api_key="your-openrouter-key",  # Access 100+ models
        openai_api_key="your-openai-key",
        anthropic_api_key="your-anthropic-key",
        google_api_key="your-google-key"
    )
    
    # Initialize model provider
    provider = ModelProvider(config=model_config.dict())
    
    print("=== Available Models ===")
    print("\nAll available models:")
    for model in provider.list_available_models():
        print(f"- {model.name} ({model.id}): {model.description}")
    
    print("\n=== Model Recommendations ===")
    
    # Get recommendations for different tasks
    coding_model = provider.recommend_model(
        task_type="coding",
        budget_priority=False,
        speed_priority=False
    )
    print(f"\nBest for coding: {coding_model.name}")
    
    fast_model = provider.recommend_model(
        task_type="general",
        budget_priority=False,
        speed_priority=True
    )
    print(f"Fastest model: {fast_model.name}")
    
    budget_model = provider.recommend_model(
        task_type="general",
        budget_priority=True,
        speed_priority=False
    )
    print(f"Most affordable: {budget_model.name}")
    
    # Example 1: Using GPT-4 for complex code generation
    print("\n=== Example 1: Code Generation with GPT-4 ===")
    code_agent = CodeGeneratorAgent(
        model="openai/gpt-4-turbo",  # Specific model
        model_config=model_config
    )
    
    # Example 2: Using Claude 3 Opus for deep analysis
    print("\n=== Example 2: Research with Claude 3 Opus ===")
    research_agent = ResearchAgent(
        model="anthropic/claude-3-opus",  # Best for analysis
        model_config=model_config
    )
    
    # Example 3: Using Mixtral via Groq for fast responses
    print("\n=== Example 3: Fast Response with Groq ===")
    fast_agent = BaseAgent(
        model="groq/mixtral-8x7b-32768",  # Ultra-fast inference
        model_config=model_config,
        system_prompt="You are a helpful assistant providing quick responses."
    )
    
    # Example 4: Using Llama 3 via OpenRouter
    print("\n=== Example 4: Open Source Model via OpenRouter ===")
    open_agent = BaseAgent(
        model="meta-llama/llama-3-70b-instruct",  # Via OpenRouter
        model_config=model_config,
        system_prompt="You are an AI assistant using open source models."
    )
    
    # Cost estimation example
    print("\n=== Cost Estimation ===")
    
    # Estimate costs for different models
    models_to_compare = [
        "openai/gpt-4",
        "anthropic/claude-3-opus",
        "openai/gpt-3.5-turbo",
        "groq/mixtral-8x7b-32768"
    ]
    
    for model_id in models_to_compare:
        cost = provider.estimate_cost(
            model_id=model_id,
            input_tokens=1000,
            output_tokens=500
        )
        if "error" not in cost:
            print(f"\n{model_id}:")
            print(f"  Input cost: ${cost['input_cost']:.4f}")
            print(f"  Output cost: ${cost['output_cost']:.4f}")
            print(f"  Total cost: ${cost['total_cost']:.4f}")


def streamlit_demo():
    """Streamlit demo for model selection UI"""
    st.set_page_config(
        page_title="Multi-Model Demo",
        page_icon="🤖",
        layout="wide"
    )
    
    st.title("🤖 Multi-Model Support Demo")
    st.markdown("""
    This demo shows how the Enhanced Agentic Workflow supports multiple AI models
    from various providers, similar to OpenRouter but integrated directly into the framework.
    """)
    
    # Initialize model selector UI
    model_selector = ModelSelectorUI()
    
    # Sidebar model selection
    with st.sidebar:
        selected_model = model_selector.render_sidebar()
        st.write(f"Selected: `{selected_model}`")
    
    # Main interface tabs
    tab1, tab2, tab3 = st.tabs(["Quick Demo", "Full Model Browser", "Cost Calculator"])
    
    with tab1:
        st.header("Quick Model Test")
        
        # User input
        user_prompt = st.text_area(
            "Enter your prompt:",
            value="Write a Python function to calculate fibonacci numbers",
            height=100
        )
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            model_choice = st.selectbox(
                "Choose a model:",
                [
                    "openai/gpt-4-turbo",
                    "anthropic/claude-3-opus",
                    "openai/gpt-3.5-turbo",
                    "meta-llama/llama-3-70b-instruct",
                    "groq/mixtral-8x7b-32768",
                    "google/gemini-pro"
                ]
            )
            
            if st.button("🚀 Run Model", type="primary"):
                with st.spinner(f"Running {model_choice}..."):
                    # Here you would actually run the model
                    st.success(f"Successfully ran {model_choice}!")
                    
                    # Show mock response
                    st.code("""
def fibonacci(n):
    if n <= 0:
        return []
    elif n == 1:
        return [0]
    elif n == 2:
        return [0, 1]
    
    fib = [0, 1]
    for i in range(2, n):
        fib.append(fib[i-1] + fib[i-2])
    return fib
                    """, language="python")
        
        with col2:
            # Model info
            model_provider = ModelProvider()
            model_info = model_provider.get_model_info(model_choice)
            
            if model_info:
                st.info(f"**{model_info.name}**\n\n{model_info.description}")
                
                col2_1, col2_2, col2_3 = st.columns(3)
                with col2_1:
                    st.metric("Speed", f"{model_info.speed_rank}/10")
                with col2_2:
                    st.metric("Quality", f"{model_info.quality_rank}/10")
                with col2_3:
                    avg_price = (model_info.pricing.get("input", 0) + model_info.pricing.get("output", 0)) / 2
                    st.metric("Avg Price", f"${avg_price:.3f}/1K")
    
    with tab2:
        # Full model browser interface
        model_selector.render_full_interface()
    
    with tab3:
        st.header("Model Comparison by Use Case")
        
        use_case = st.selectbox(
            "Select your use case:",
            [
                "General Chat",
                "Code Generation",
                "Data Analysis",
                "Creative Writing",
                "Translation",
                "Summarization",
                "Vision Tasks"
            ]
        )
        
        # Mock recommendations based on use case
        recommendations = {
            "General Chat": ["openai/gpt-4-turbo", "anthropic/claude-3-sonnet", "google/gemini-pro"],
            "Code Generation": ["openai/gpt-4", "anthropic/claude-3-opus", "openai/gpt-4-turbo"],
            "Data Analysis": ["anthropic/claude-3-opus", "openai/gpt-4", "google/gemini-pro"],
            "Creative Writing": ["anthropic/claude-3-opus", "openai/gpt-4", "meta-llama/llama-3-70b"],
            "Translation": ["google/gemini-pro", "openai/gpt-3.5-turbo", "anthropic/claude-3-haiku"],
            "Summarization": ["anthropic/claude-3-haiku", "openai/gpt-3.5-turbo", "groq/mixtral-8x7b"],
            "Vision Tasks": ["openai/gpt-4-turbo", "anthropic/claude-3-opus", "google/gemini-pro"]
        }
        
        st.write(f"### Recommended models for {use_case}:")
        
        for i, model_id in enumerate(recommendations.get(use_case, []), 1):
            model_info = model_provider.get_model_info(model_id)
            if model_info:
                with st.expander(f"{i}. {model_info.name}", expanded=(i==1)):
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        st.write(model_info.description)
                        st.write(f"**Context:** {model_info.context_window:,} tokens")
                        st.write(f"**Capabilities:** {', '.join([c.value for c in model_info.capabilities])}")
                    
                    with col2:
                        st.metric("Speed", f"{model_info.speed_rank}/10")
                        st.metric("Quality", f"{model_info.quality_rank}/10")
                        avg_price = (model_info.pricing.get("input", 0) + model_info.pricing.get("output", 0)) / 2
                        st.metric("Avg Price", f"${avg_price:.3f}/1K")


if __name__ == "__main__":
    # Run command line demo
    print("Enhanced Agentic Workflow - Multi-Model Support Demo")
    print("=" * 50)
    
    # Check if running in Streamlit
    try:
        import streamlit as st
        # If streamlit is available and we're in a streamlit context
        if hasattr(st, "runtime") and hasattr(st.runtime, "exists"):
            streamlit_demo()
        else:
            # Run async demo
            asyncio.run(demonstrate_multi_model_support())
    except ImportError:
        # Run async demo if streamlit not available
        asyncio.run(demonstrate_multi_model_support())