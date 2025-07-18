"""Demo: Using Enhanced Agentic Workflow with OpenRouter

This demo shows how to use different AI models through OpenRouter
with the Enhanced Agentic Workflow Architecture.
"""

import asyncio
import os
from src.agents.code_generator import CodeGeneratorAgent, CodeGeneratorDeps, CodeResult
from src.agents.research_agent import ResearchAgent, ResearchDeps, ResearchResult
from src.agents.tools_refiner import ToolsRefinerAgent, ToolsRefinerDeps, ToolsRefinerResult
from src.core.model_provider import ModelConfig
from src.ui.components.model_selector import ModelSelectorUI
import streamlit as st


# Configuration
OPENROUTER_API_KEY = "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"


async def demo_code_generation():
    """Demo: Code generation with different models"""
    print("\n" + "="*60)
    print("🚀 Code Generation Demo")
    print("="*60)
    
    # Configure with OpenRouter
    model_config = ModelConfig(
        openrouter_api_key=OPENROUTER_API_KEY,
        default_model="anthropic/claude-3.5-sonnet"
    )
    
    # Test prompt
    prompt = "Create a Python class for managing a todo list with add, remove, and list operations"
    
    # Dependencies for code generation
    deps = CodeGeneratorDeps(
        language="python",
        framework=None,
        output_format="class",
        include_tests=True,
        include_docs=True,
        style_guide="PEP8",
        performance_optimize=False,
        security_check=True
    )
    
    # Test 1: Claude 3.5 Sonnet (via OpenRouter)
    print("\n📝 Using Claude 3.5 Sonnet:")
    try:
        agent = CodeGeneratorAgent(
            model="anthropic/claude-3.5-sonnet",
            model_config=model_config,
            enable_logfire=False
        )
        
        result = await agent.run(prompt, deps)
        print(f"✅ Generated {len(result.code.splitlines())} lines of code")
        print(f"Language: {result.language}")
        print(f"Has tests: {result.has_tests}")
        print("\nCode preview:")
        print(result.code[:300] + "..." if len(result.code) > 300 else result.code)
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
    
    # Test 2: Fast model - Claude 3.5 Haiku
    print("\n\n📝 Using Claude 3.5 Haiku (Fast):")
    try:
        agent = CodeGeneratorAgent(
            model="anthropic/claude-3.5-haiku",
            model_config=model_config,
            enable_logfire=False
        )
        
        result = await agent.run("Write a simple hello world function", deps)
        print(f"✅ Generated code quickly!")
        print(result.code)
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")


async def demo_research():
    """Demo: Research with different models"""
    print("\n" + "="*60)
    print("🔬 Research Agent Demo")
    print("="*60)
    
    model_config = ModelConfig(
        openrouter_api_key=OPENROUTER_API_KEY
    )
    
    prompt = "Research the latest developments in quantum computing"
    
    deps = ResearchDeps(
        topic="quantum computing",
        depth="comprehensive",
        include_sources=True,
        max_sources=5,
        focus_areas=["recent breakthroughs", "practical applications"],
        time_range="2023-2024",
        output_format="report"
    )
    
    print("\n📚 Using Claude 3.7 Sonnet for deep research:")
    try:
        agent = ResearchAgent(
            model="anthropic/claude-3.7-sonnet",
            model_config=model_config,
            enable_logfire=False
        )
        
        result = await agent.run(prompt, deps)
        print(f"✅ Research complete!")
        print(f"Summary: {result.summary[:200]}...")
        print(f"Key findings: {len(result.key_findings)}")
        print(f"Sources: {len(result.sources)}")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")


async def demo_multi_agent_workflow():
    """Demo: Multi-agent workflow with different models"""
    print("\n" + "="*60)
    print("🤝 Multi-Agent Workflow Demo")
    print("="*60)
    
    model_config = ModelConfig(
        openrouter_api_key=OPENROUTER_API_KEY
    )
    
    # Create agents with different models for different tasks
    print("\n🎯 Setting up specialized agents:")
    
    # Research agent with Claude 3.7 Sonnet (best for analysis)
    research_agent = ResearchAgent(
        model="anthropic/claude-3.7-sonnet",
        model_config=model_config,
        enable_logfire=False
    )
    print("  ✓ Research Agent: Claude 3.7 Sonnet")
    
    # Code generator with Claude 3.5 Sonnet (good balance)
    code_agent = CodeGeneratorAgent(
        model="anthropic/claude-3.5-sonnet",
        model_config=model_config,
        enable_logfire=False
    )
    print("  ✓ Code Agent: Claude 3.5 Sonnet")
    
    # Tools refiner with fast model
    tools_agent = ToolsRefinerAgent(
        model="anthropic/claude-3.5-haiku",
        model_config=model_config,
        enable_logfire=False
    )
    print("  ✓ Tools Agent: Claude 3.5 Haiku (Fast)")
    
    print("\n🔄 Running workflow...")
    
    # Step 1: Research
    research_prompt = "Research best practices for Python async programming"
    research_deps = ResearchDeps(
        topic="Python async programming",
        depth="detailed",
        include_sources=True,
        max_sources=3
    )
    
    print("\n1️⃣ Research Phase:")
    research_result = await research_agent.run(research_prompt, research_deps)
    print(f"   ✅ Found {len(research_result.key_findings)} key findings")
    
    # Step 2: Generate code based on research
    code_prompt = f"Based on these findings: {research_result.summary[:200]}... Create an example async Python application"
    code_deps = CodeGeneratorDeps(
        language="python",
        framework="asyncio",
        output_format="script",
        include_docs=True
    )
    
    print("\n2️⃣ Code Generation Phase:")
    code_result = await code_agent.run(code_prompt, code_deps)
    print(f"   ✅ Generated {len(code_result.code.splitlines())} lines of code")
    
    # Step 3: Refine with tools
    tools_prompt = "Add error handling and logging to this async code"
    tools_deps = ToolsRefinerDeps(
        code=code_result.code,
        language="python",
        requested_tools=["error_handling", "logging"],
        preserve_functionality=True,
        add_tests=False
    )
    
    print("\n3️⃣ Tools Refinement Phase:")
    tools_result = await tools_agent.run(tools_prompt, tools_deps)
    print(f"   ✅ Applied {len(tools_result.tools_applied)} tools")
    print(f"   ✅ Made {tools_result.changes_made} changes")
    
    print("\n✨ Workflow Complete!")


def streamlit_app():
    """Streamlit UI Demo"""
    st.set_page_config(
        page_title="Enhanced Agentic Workflow - OpenRouter Demo",
        page_icon="🤖",
        layout="wide"
    )
    
    st.title("🤖 Enhanced Agentic Workflow with OpenRouter")
    st.markdown("""
    This demo shows how to use multiple AI models through OpenRouter
    with the Enhanced Agentic Workflow Architecture.
    """)
    
    # Initialize model selector
    model_config = ModelConfig(
        openrouter_api_key=OPENROUTER_API_KEY
    )
    
    model_selector = ModelSelectorUI(model_config=model_config)
    
    # Sidebar
    with st.sidebar:
        selected_model = model_selector.render_sidebar()
        
        st.divider()
        
        st.info(f"""
        **Current Model:** `{selected_model}`
        
        **Available Models:**
        - Claude 3.7 Sonnet
        - Claude 3.5 Sonnet  
        - Claude 3.5 Haiku
        - GPT-4 Turbo
        - Llama 3 70B
        - Mixtral 8x7B
        - And 100+ more!
        """)
    
    # Main content
    tab1, tab2, tab3 = st.tabs(["Code Generation", "Research", "Multi-Agent"])
    
    with tab1:
        st.header("Code Generation Demo")
        
        code_prompt = st.text_area(
            "Enter your code generation prompt:",
            value="Create a Python web scraper with error handling",
            height=100
        )
        
        col1, col2 = st.columns(2)
        with col1:
            language = st.selectbox("Language", ["python", "javascript", "typescript", "go", "rust"])
            include_tests = st.checkbox("Include tests", value=True)
        
        with col2:
            include_docs = st.checkbox("Include documentation", value=True)
            security_check = st.checkbox("Security check", value=True)
        
        if st.button("Generate Code", type="primary"):
            with st.spinner(f"Generating with {selected_model}..."):
                # Here you would run the actual agent
                st.success("Code generated successfully!")
                st.code("""
# Example generated code
import requests
from bs4 import BeautifulSoup
import logging

class WebScraper:
    '''A robust web scraper with error handling'''
    
    def __init__(self, timeout=10):
        self.timeout = timeout
        self.logger = logging.getLogger(__name__)
    
    def scrape(self, url):
        try:
            response = requests.get(url, timeout=self.timeout)
            response.raise_for_status()
            return BeautifulSoup(response.content, 'html.parser')
        except Exception as e:
            self.logger.error(f"Failed to scrape {url}: {e}")
            return None
                """, language=language)
    
    with tab2:
        st.header("Research Agent Demo")
        
        research_topic = st.text_input(
            "Research topic:",
            value="Latest AI developments in 2024"
        )
        
        depth = st.select_slider(
            "Research depth",
            options=["quick", "standard", "detailed", "comprehensive"],
            value="detailed"
        )
        
        if st.button("Start Research", type="primary"):
            with st.spinner(f"Researching with {selected_model}..."):
                # Simulate research
                st.success("Research complete!")
                
                with st.expander("📊 Key Findings", expanded=True):
                    st.markdown("""
                    1. **AI Reasoning Capabilities** - Major improvements in logical reasoning
                    2. **Multimodal Models** - Better integration of vision and language
                    3. **Efficiency Gains** - Smaller models with better performance
                    4. **Safety Advances** - Improved alignment and safety measures
                    """)
                
                with st.expander("📚 Sources"):
                    st.markdown("""
                    - OpenAI GPT-4 Turbo Release Notes
                    - Anthropic Claude 3 Technical Report
                    - Google Gemini Research Papers
                    """)
    
    with tab3:
        st.header("Multi-Agent Workflow")
        
        st.markdown("""
        Configure a multi-agent workflow where different models handle different tasks:
        """)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("🔬 Research Agent")
            research_model = st.selectbox(
                "Model for research",
                ["anthropic/claude-3.7-sonnet", "anthropic/claude-3-opus", "openai/gpt-4"],
                key="research"
            )
        
        with col2:
            st.subheader("💻 Code Agent")
            code_model = st.selectbox(
                "Model for coding",
                ["anthropic/claude-3.5-sonnet", "openai/gpt-4-turbo", "meta-llama/llama-3-70b"],
                key="code"
            )
        
        with col3:
            st.subheader("🔧 Refiner Agent")
            refiner_model = st.selectbox(
                "Model for refinement",
                ["anthropic/claude-3.5-haiku", "groq/mixtral-8x7b-32768", "openai/gpt-3.5-turbo"],
                key="refiner"
            )
        
        workflow_task = st.text_area(
            "Describe your task:",
            value="Build a real-time chat application with websockets",
            height=100
        )
        
        if st.button("Run Multi-Agent Workflow", type="primary"):
            # Progress tracking
            progress = st.progress(0)
            status = st.empty()
            
            # Simulate workflow
            status.text("🔬 Research Agent: Gathering information...")
            progress.progress(33)
            
            st.success("✅ Research complete")
            
            status.text("💻 Code Agent: Generating implementation...")
            progress.progress(66)
            
            st.success("✅ Code generated")
            
            status.text("🔧 Refiner Agent: Optimizing and adding tools...")
            progress.progress(100)
            
            st.success("✅ Workflow complete!")
            
            # Show results
            st.balloons()
            
            with st.expander("View Results", expanded=True):
                st.markdown("""
                ### Workflow Summary
                
                1. **Research Phase** (Claude 3.7 Sonnet)
                   - Analyzed websocket protocols
                   - Identified best practices
                   - Found popular libraries
                
                2. **Code Generation** (Claude 3.5 Sonnet)
                   - Generated server implementation
                   - Created client library
                   - Added authentication
                
                3. **Refinement** (Claude 3.5 Haiku)
                   - Added error handling
                   - Implemented logging
                   - Optimized performance
                """)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "streamlit":
        streamlit_app()
    else:
        print("🚀 Enhanced Agentic Workflow - OpenRouter Demo")
        print("="*60)
        
        # Run demos
        asyncio.run(demo_code_generation())
        asyncio.run(demo_research())
        asyncio.run(demo_multi_agent_workflow())
        
        print("\n" + "="*60)
        print("✨ Demo Complete!")
        print("\nTo run the Streamlit UI demo:")
        print("  streamlit run demo_with_openrouter.py streamlit")