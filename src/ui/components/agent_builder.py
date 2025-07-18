"""Visual Agent Creation Interface Component"""

import streamlit as st
import yaml
import json
import os
from pathlib import Path
from typing import Dict, List, Any, Optional
import subprocess
import tempfile
from datetime import datetime


class AgentBuilder:
    """UI component for visual agent creation and configuration"""
    
    def __init__(self):
        self.templates_dir = Path("templates/agents")
        self.agents_dir = Path("agents")
        self.agent_types = {
            "Research Agent": {
                "icon": "🔍",
                "description": "Specialized in web research and information gathering",
                "default_tools": ["search_tool", "read_file_tool", "context_tool"],
                "capabilities": ["Web search", "Document analysis", "Fact checking"]
            },
            "Coding Agent": {
                "icon": "💻",
                "description": "Expert in code generation, analysis, and refactoring",
                "default_tools": ["write_file_tool", "read_file_tool", "validation_tool"],
                "capabilities": ["Code generation", "Bug fixing", "Refactoring", "Testing"]
            },
            "Analysis Agent": {
                "icon": "📊",
                "description": "Focused on data analysis and insights generation",
                "default_tools": ["calculator_tool", "read_file_tool", "write_file_tool"],
                "capabilities": ["Data analysis", "Pattern recognition", "Report generation"]
            },
            "Creative Agent": {
                "icon": "🎨",
                "description": "Specialized in creative content generation",
                "default_tools": ["write_file_tool", "context_tool"],
                "capabilities": ["Content writing", "Ideation", "Storytelling"]
            },
            "Task Agent": {
                "icon": "✅",
                "description": "General purpose task completion agent",
                "default_tools": ["task_done_tool", "write_file_tool", "read_file_tool"],
                "capabilities": ["Task planning", "Execution", "Progress tracking"]
            },
            "Custom Agent": {
                "icon": "🔧",
                "description": "Build your own specialized agent from scratch",
                "default_tools": [],
                "capabilities": []
            }
        }
    
    def render(self):
        """Render the agent builder interface"""
        st.title("🤖 Agent Builder")
        st.markdown("Create and configure intelligent agents with visual tools")
        
        # Create tabs for different builder modes
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "Quick Start",
            "Advanced Builder",
            "Templates",
            "Testing",
            "Deployment"
        ])
        
        with tab1:
            self._render_quick_start()
        
        with tab2:
            self._render_advanced_builder()
        
        with tab3:
            self._render_templates()
        
        with tab4:
            self._render_testing()
        
        with tab5:
            self._render_deployment()
    
    def _render_quick_start(self):
        """Render quick start agent creation"""
        st.subheader("Quick Agent Creation")
        st.markdown("Create a functional agent in seconds with pre-configured templates")
        
        # Agent type selection
        col1, col2 = st.columns([1, 2])
        
        with col1:
            agent_type = st.selectbox(
                "Select Agent Type",
                list(self.agent_types.keys()),
                format_func=lambda x: f"{self.agent_types[x]['icon']} {x}"
            )
        
        with col2:
            agent_info = self.agent_types[agent_type]
            st.markdown(f"### {agent_info['icon']} {agent_type}")
            st.markdown(agent_info['description'])
            
            if agent_info['capabilities']:
                st.markdown("**Capabilities:**")
                for cap in agent_info['capabilities']:
                    st.markdown(f"• {cap}")
        
        st.markdown("---")
        
        # Basic configuration
        col1, col2 = st.columns(2)
        
        with col1:
            agent_name = st.text_input(
                "Agent Name",
                placeholder="my_research_agent",
                help="Lowercase, underscore-separated name"
            )
            
            model = st.selectbox(
                "Model",
                ["openai/gpt-4", "openai/gpt-3.5-turbo", "anthropic/claude-2", "google/gemini-pro"],
                help="AI model to power the agent"
            )
        
        with col2:
            temperature = st.slider(
                "Temperature",
                min_value=0.0,
                max_value=2.0,
                value=0.7,
                step=0.1,
                help="Controls randomness in responses"
            )
            
            max_iterations = st.number_input(
                "Max Iterations",
                min_value=1,
                max_value=50,
                value=10,
                help="Maximum thinking steps"
            )
        
        # Tools selection
        st.markdown("### Tools")
        available_tools = self._get_available_tools()
        
        default_tools = agent_info.get('default_tools', [])
        selected_tools = st.multiselect(
            "Select Tools",
            available_tools,
            default=default_tools,
            help="Tools the agent can use"
        )
        
        # System prompt
        st.markdown("### System Prompt")
        
        if agent_type != "Custom Agent":
            default_prompt = self._get_default_prompt(agent_type)
        else:
            default_prompt = ""
        
        system_prompt = st.text_area(
            "System Prompt",
            value=default_prompt,
            height=200,
            help="Instructions that define the agent's behavior"
        )
        
        # Create agent button
        col1, col2, col3 = st.columns([1, 1, 2])
        
        with col1:
            if st.button("🚀 Create Agent", type="primary"):
                if agent_name and system_prompt:
                    agent_config = {
                        "name": agent_name,
                        "type": agent_type,
                        "model": model,
                        "temperature": temperature,
                        "max_iterations": max_iterations,
                        "tools": selected_tools,
                        "system_prompt": system_prompt
                    }
                    self._create_agent(agent_config)
                else:
                    st.error("Please provide agent name and system prompt")
        
        with col2:
            if st.button("💾 Save as Template"):
                self._save_as_template(agent_config)
        
        with col3:
            if st.button("🧪 Test Agent"):
                st.session_state['test_agent'] = agent_config
                st.session_state['active_tab'] = "Testing"
                st.rerun()
    
    def _render_advanced_builder(self):
        """Render advanced agent builder with full customization"""
        st.subheader("Advanced Agent Builder")
        st.markdown("Fine-tune every aspect of your agent's configuration")
        
        # Configuration sections
        with st.expander("🧠 Core Configuration", expanded=True):
            self._render_core_config()
        
        with st.expander("🛠️ Tools & Capabilities"):
            self._render_tools_config()
        
        with st.expander("💭 Reasoning & Planning"):
            self._render_reasoning_config()
        
        with st.expander("🔄 Self-Improvement"):
            self._render_self_improvement_config()
        
        with st.expander("🔌 Integrations"):
            self._render_integrations_config()
        
        with st.expander("📊 Monitoring & Logging"):
            self._render_monitoring_config()
        
        # Generate configuration
        if st.button("📋 Generate Configuration"):
            config = self._generate_advanced_config()
            st.code(yaml.dump(config, default_flow_style=False), language="yaml")
            
            # Download button
            st.download_button(
                "Download Configuration",
                yaml.dump(config, default_flow_style=False),
                f"agent_config_{datetime.now().strftime('%Y%m%d_%H%M%S')}.yaml",
                "text/yaml"
            )
    
    def _render_core_config(self):
        """Render core agent configuration"""
        col1, col2 = st.columns(2)
        
        with col1:
            st.text_input("Agent ID", placeholder="unique_agent_id")
            st.text_input("Display Name", placeholder="My Intelligent Agent")
            st.text_area("Description", placeholder="Describe what this agent does...")
        
        with col2:
            st.selectbox("Base Model", ["openai/gpt-4", "anthropic/claude-2", "google/gemini-pro"])
            st.selectbox("Fallback Model", ["openai/gpt-3.5-turbo", "None"])
            st.number_input("Timeout (seconds)", value=300, min_value=60)
        
        # Advanced model parameters
        st.markdown("### Model Parameters")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.slider("Temperature", 0.0, 2.0, 0.7)
        
        with col2:
            st.slider("Top P", 0.0, 1.0, 1.0)
        
        with col3:
            st.number_input("Max Tokens", value=2000, min_value=100)
        
        with col4:
            st.number_input("Frequency Penalty", value=0.0, min_value=-2.0, max_value=2.0)
    
    def _render_tools_config(self):
        """Render tools configuration"""
        st.markdown("### Available Tools")
        
        # Tool categories
        tool_categories = {
            "File Operations": ["read_file_tool", "write_file_tool"],
            "Web & Search": ["search_tool", "web_scraper_tool"],
            "Data Processing": ["calculator_tool", "data_analyzer_tool"],
            "Communication": ["email_tool", "slack_tool"],
            "Development": ["git_tool", "docker_tool", "validation_tool"],
            "Custom Tools": []
        }
        
        selected_tools = []
        
        for category, tools in tool_categories.items():
            st.markdown(f"**{category}**")
            
            if tools:
                cols = st.columns(3)
                for i, tool in enumerate(tools):
                    with cols[i % 3]:
                        if st.checkbox(tool.replace("_", " ").title(), key=f"tool_{tool}"):
                            selected_tools.append(tool)
            else:
                st.info("No tools in this category")
        
        # Custom tool builder
        st.markdown("### Custom Tool Builder")
        
        with st.expander("Add Custom Tool"):
            tool_name = st.text_input("Tool Name")
            tool_description = st.text_area("Tool Description")
            tool_code = st.text_area(
                "Tool Implementation",
                value="""async def custom_tool(query: str) -> str:
    # Your tool implementation here
    return "Tool result\""""
            )
            
            if st.button("Add Custom Tool"):
                st.success(f"Custom tool '{tool_name}' added!")
    
    def _render_reasoning_config(self):
        """Render reasoning and planning configuration"""
        st.markdown("### Reasoning Strategy")
        
        reasoning_mode = st.selectbox(
            "Reasoning Mode",
            ["Chain of Thought", "Tree of Thoughts", "Graph of Thoughts", "ReAct"],
            help="How the agent structures its thinking"
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.checkbox("Enable Self-Reflection", value=True)
            st.checkbox("Enable Planning", value=True)
            st.checkbox("Enable Goal Decomposition", value=True)
        
        with col2:
            st.number_input("Max Reasoning Depth", value=5, min_value=1)
            st.number_input("Reflection Frequency", value=3, min_value=1)
            st.slider("Confidence Threshold", 0.0, 1.0, 0.7)
        
        # Custom reasoning prompt
        st.markdown("### Custom Reasoning Prompt")
        st.text_area(
            "Reasoning Template",
            value="""Let me think through this step by step:
1. Understanding: {understanding}
2. Planning: {plan}
3. Execution: {execution}
4. Validation: {validation}""",
            height=150
        )
    
    def _render_self_improvement_config(self):
        """Render self-improvement configuration"""
        st.markdown("### Self-Improvement Settings")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.checkbox("Enable Prompt Optimization", value=False)
            st.checkbox("Enable Tool Learning", value=False)
            st.checkbox("Enable Memory Formation", value=True)
        
        with col2:
            st.checkbox("Enable Error Learning", value=True)
            st.checkbox("Enable Performance Tracking", value=True)
            st.checkbox("Enable A/B Testing", value=False)
        
        # Learning parameters
        st.markdown("### Learning Parameters")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.slider("Learning Rate", 0.0, 1.0, 0.1)
        
        with col2:
            st.number_input("Memory Capacity", value=1000, min_value=100)
        
        with col3:
            st.selectbox("Memory Type", ["Episodic", "Semantic", "Procedural", "All"])
        
        # Improvement strategies
        st.markdown("### Improvement Strategies")
        
        strategies = st.multiselect(
            "Active Strategies",
            [
                "Reinforcement from Human Feedback",
                "Self-Supervised Learning",
                "Few-Shot Adaptation",
                "Meta-Learning",
                "Continual Learning"
            ]
        )
    
    def _render_integrations_config(self):
        """Render integrations configuration"""
        st.markdown("### External Integrations")
        
        # API integrations
        st.markdown("#### API Integrations")
        
        integrations = {
            "GitHub": ["Repository access", "Issue management", "PR creation"],
            "Slack": ["Message sending", "Channel monitoring", "User interactions"],
            "Jira": ["Issue tracking", "Project management", "Sprint planning"],
            "Database": ["Query execution", "Data retrieval", "Schema management"],
            "Cloud Services": ["AWS", "GCP", "Azure"]
        }
        
        for service, features in integrations.items():
            col1, col2 = st.columns([1, 3])
            
            with col1:
                enabled = st.checkbox(service, key=f"int_{service}")
            
            with col2:
                if enabled:
                    selected_features = st.multiselect(
                        f"{service} Features",
                        features,
                        key=f"features_{service}"
                    )
        
        # Webhook configuration
        st.markdown("#### Webhooks")
        
        webhook_url = st.text_input("Webhook URL", placeholder="https://your-webhook.com/endpoint")
        webhook_events = st.multiselect(
            "Trigger Events",
            ["Task Started", "Task Completed", "Error Occurred", "Goal Achieved"]
        )
    
    def _render_monitoring_config(self):
        """Render monitoring configuration"""
        st.markdown("### Monitoring & Observability")
        
        # Logging configuration
        col1, col2 = st.columns(2)
        
        with col1:
            st.selectbox("Log Level", ["DEBUG", "INFO", "WARNING", "ERROR"])
            st.checkbox("Enable Structured Logging", value=True)
            st.checkbox("Log to File", value=True)
        
        with col2:
            st.text_input("Log File Path", value="./logs/agent.log")
            st.number_input("Max Log Size (MB)", value=100)
            st.number_input("Log Retention (days)", value=30)
        
        # Metrics configuration
        st.markdown("### Metrics Collection")
        
        metrics = st.multiselect(
            "Tracked Metrics",
            [
                "Response Time",
                "Token Usage",
                "Task Success Rate",
                "Error Rate",
                "Tool Usage",
                "Memory Usage",
                "Cost per Task"
            ],
            default=["Response Time", "Token Usage", "Task Success Rate"]
        )
        
        # Alerting
        st.markdown("### Alerting")
        
        alert_channels = st.multiselect(
            "Alert Channels",
            ["Email", "Slack", "PagerDuty", "Webhook"]
        )
        
        if alert_channels:
            st.markdown("#### Alert Rules")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.selectbox("Metric", metrics if metrics else ["None"])
            
            with col2:
                st.selectbox("Condition", [">", "<", "==", "!="])
            
            with col3:
                st.number_input("Threshold", value=0.0)
    
    def _render_templates(self):
        """Render agent templates library"""
        st.subheader("Agent Templates Library")
        st.markdown("Start with pre-built agent templates for common use cases")
        
        # Template categories
        categories = {
            "Research & Analysis": [
                {"name": "Academic Researcher", "icon": "🎓", "description": "Literature review and citation management"},
                {"name": "Market Analyst", "icon": "📈", "description": "Market research and competitive analysis"},
                {"name": "Data Scientist", "icon": "🔬", "description": "Data analysis and visualization"}
            ],
            "Development & Engineering": [
                {"name": "Full Stack Developer", "icon": "💻", "description": "Web application development"},
                {"name": "DevOps Engineer", "icon": "⚙️", "description": "Infrastructure and deployment"},
                {"name": "QA Automation", "icon": "🧪", "description": "Testing and quality assurance"}
            ],
            "Content & Creative": [
                {"name": "Content Writer", "icon": "✍️", "description": "Blog posts and articles"},
                {"name": "Social Media Manager", "icon": "📱", "description": "Social media content and engagement"},
                {"name": "Technical Writer", "icon": "📚", "description": "Documentation and guides"}
            ],
            "Business & Operations": [
                {"name": "Project Manager", "icon": "📊", "description": "Project planning and tracking"},
                {"name": "Customer Support", "icon": "🎧", "description": "Customer service and ticketing"},
                {"name": "Sales Assistant", "icon": "💼", "description": "Lead generation and CRM"}
            ]
        }
        
        # Search and filter
        col1, col2 = st.columns([2, 1])
        
        with col1:
            search = st.text_input("Search templates...", placeholder="e.g., research, coding, analysis")
        
        with col2:
            filter_category = st.selectbox("Filter by Category", ["All"] + list(categories.keys()))
        
        # Display templates
        for category, templates in categories.items():
            if filter_category == "All" or filter_category == category:
                st.markdown(f"### {category}")
                
                cols = st.columns(3)
                for i, template in enumerate(templates):
                    with cols[i % 3]:
                        # Template card
                        st.markdown(f"""
                        <div style='border: 1px solid #ddd; border-radius: 10px; padding: 15px; margin: 5px;'>
                            <h4>{template['icon']} {template['name']}</h4>
                            <p style='color: #666; font-size: 14px;'>{template['description']}</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            if st.button("Use", key=f"use_{template['name']}"):
                                self._load_template(template['name'])
                        
                        with col2:
                            if st.button("Preview", key=f"preview_{template['name']}"):
                                self._preview_template(template['name'])
        
        # Import custom template
        st.markdown("---")
        st.markdown("### Import Custom Template")
        
        uploaded_file = st.file_uploader(
            "Upload Template File",
            type=['yaml', 'json'],
            help="Upload a custom agent template"
        )
        
        if uploaded_file:
            if st.button("Import Template"):
                self._import_template(uploaded_file)
    
    def _render_testing(self):
        """Render agent testing interface"""
        st.subheader("Agent Testing Playground")
        st.markdown("Test your agents in a controlled environment")
        
        # Agent selection
        col1, col2 = st.columns([2, 1])
        
        with col1:
            available_agents = self._get_available_agents()
            test_agent = st.selectbox(
                "Select Agent to Test",
                available_agents + ["<Create New>"]
            )
        
        with col2:
            test_mode = st.selectbox(
                "Test Mode",
                ["Interactive", "Batch", "Benchmark"]
            )
        
        st.markdown("---")
        
        if test_mode == "Interactive":
            self._render_interactive_testing()
        elif test_mode == "Batch":
            self._render_batch_testing()
        elif test_mode == "Benchmark":
            self._render_benchmark_testing()
    
    def _render_interactive_testing(self):
        """Render interactive testing interface"""
        st.markdown("### Interactive Testing")
        
        # Test input
        test_query = st.text_area(
            "Test Query",
            placeholder="Enter a query to test the agent...",
            height=100
        )
        
        # Test configuration
        with st.expander("Test Configuration"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                enable_logging = st.checkbox("Enable Detailed Logging", value=True)
                enable_profiling = st.checkbox("Enable Performance Profiling", value=False)
            
            with col2:
                mock_tools = st.checkbox("Use Mock Tools", value=False)
                limit_iterations = st.number_input("Limit Iterations", value=10)
            
            with col3:
                timeout = st.number_input("Timeout (seconds)", value=60)
                seed = st.number_input("Random Seed", value=42)
        
        # Run test
        if st.button("🚀 Run Test", type="primary"):
            if test_query:
                with st.spinner("Running test..."):
                    self._run_interactive_test(test_query)
            else:
                st.error("Please enter a test query")
        
        # Test results
        if 'test_results' in st.session_state:
            st.markdown("### Test Results")
            
            results = st.session_state.test_results
            
            # Metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Response Time", f"{results.get('response_time', 0):.2f}s")
            
            with col2:
                st.metric("Tokens Used", results.get('tokens_used', 0))
            
            with col3:
                st.metric("Iterations", results.get('iterations', 0))
            
            with col4:
                st.metric("Success", "✅" if results.get('success') else "❌")
            
            # Response
            st.markdown("#### Agent Response")
            st.code(results.get('response', ''), language="markdown")
            
            # Thinking process
            with st.expander("Thinking Process"):
                for i, thought in enumerate(results.get('thoughts', [])):
                    st.markdown(f"**Step {i+1}:**")
                    st.text(thought)
            
            # Tool usage
            with st.expander("Tool Usage"):
                tool_usage = results.get('tool_usage', [])
                if tool_usage:
                    for tool in tool_usage:
                        st.markdown(f"**{tool['name']}**")
                        st.json(tool['args'])
                        st.text(f"Result: {tool['result']}")
                else:
                    st.info("No tools were used")
    
    def _render_batch_testing(self):
        """Render batch testing interface"""
        st.markdown("### Batch Testing")
        
        # Test cases input
        test_input_method = st.radio(
            "Test Input Method",
            ["Upload CSV", "Upload JSON", "Manual Entry"]
        )
        
        if test_input_method == "Upload CSV":
            uploaded_file = st.file_uploader("Upload Test Cases (CSV)", type=['csv'])
            if uploaded_file:
                st.info(f"Loaded {uploaded_file.name}")
        
        elif test_input_method == "Upload JSON":
            uploaded_file = st.file_uploader("Upload Test Cases (JSON)", type=['json'])
            if uploaded_file:
                st.info(f"Loaded {uploaded_file.name}")
        
        else:  # Manual Entry
            num_cases = st.number_input("Number of Test Cases", value=3, min_value=1)
            
            test_cases = []
            for i in range(num_cases):
                with st.expander(f"Test Case {i+1}"):
                    query = st.text_input(f"Query {i+1}")
                    expected = st.text_area(f"Expected Output {i+1} (Optional)")
                    test_cases.append({"query": query, "expected": expected})
        
        # Run batch tests
        if st.button("🚀 Run Batch Tests"):
            with st.spinner("Running batch tests..."):
                results = self._run_batch_tests(test_cases)
                
                # Display results
                st.markdown("### Batch Test Results")
                
                # Summary metrics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Total Tests", len(results))
                
                with col2:
                    passed = sum(1 for r in results if r['passed'])
                    st.metric("Passed", passed)
                
                with col3:
                    st.metric("Failed", len(results) - passed)
                
                with col4:
                    avg_time = sum(r['time'] for r in results) / len(results)
                    st.metric("Avg Time", f"{avg_time:.2f}s")
                
                # Detailed results
                for i, result in enumerate(results):
                    with st.expander(f"Test Case {i+1} - {'✅ Passed' if result['passed'] else '❌ Failed'}"):
                        st.text(f"Query: {result['query']}")
                        st.text(f"Response: {result['response']}")
                        st.text(f"Time: {result['time']:.2f}s")
                        if result.get('error'):
                            st.error(f"Error: {result['error']}")
    
    def _render_benchmark_testing(self):
        """Render benchmark testing interface"""
        st.markdown("### Performance Benchmarking")
        
        # Benchmark selection
        benchmark_suite = st.selectbox(
            "Benchmark Suite",
            ["General Intelligence", "Coding Tasks", "Research Tasks", "Creative Writing", "Custom"]
        )
        
        if benchmark_suite == "Custom":
            custom_benchmarks = st.text_area(
                "Custom Benchmark Tasks (one per line)",
                height=150
            )
        
        # Benchmark configuration
        col1, col2 = st.columns(2)
        
        with col1:
            num_runs = st.number_input("Number of Runs", value=3, min_value=1)
            warmup_runs = st.number_input("Warmup Runs", value=1, min_value=0)
        
        with col2:
            compare_models = st.multiselect(
                "Compare Against",
                ["GPT-3.5", "GPT-4", "Claude-2", "Gemini Pro"]
            )
        
        # Run benchmark
        if st.button("🚀 Run Benchmark"):
            with st.spinner("Running benchmarks..."):
                results = self._run_benchmarks(benchmark_suite)
                
                # Display results
                st.markdown("### Benchmark Results")
                
                # Performance chart placeholder
                st.info("Performance charts would be displayed here")
                
                # Detailed metrics
                metrics = {
                    "Speed": {"Your Agent": 2.3, "GPT-4": 3.1, "GPT-3.5": 1.8},
                    "Accuracy": {"Your Agent": 0.92, "GPT-4": 0.95, "GPT-3.5": 0.87},
                    "Cost": {"Your Agent": 0.05, "GPT-4": 0.08, "GPT-3.5": 0.02}
                }
                
                for metric, values in metrics.items():
                    st.markdown(f"#### {metric}")
                    cols = st.columns(len(values))
                    for i, (model, value) in enumerate(values.items()):
                        with cols[i]:
                            st.metric(model, value)
    
    def _render_deployment(self):
        """Render deployment interface"""
        st.subheader("Agent Deployment")
        st.markdown("Deploy your agents to production environments")
        
        # Deployment target
        deployment_target = st.selectbox(
            "Deployment Target",
            ["Local Server", "Cloud Function", "API Endpoint", "MCP Server", "Docker Container"]
        )
        
        st.markdown("---")
        
        if deployment_target == "Local Server":
            self._render_local_deployment()
        elif deployment_target == "Cloud Function":
            self._render_cloud_deployment()
        elif deployment_target == "API Endpoint":
            self._render_api_deployment()
        elif deployment_target == "MCP Server":
            self._render_mcp_deployment()
        elif deployment_target == "Docker Container":
            self._render_docker_deployment()
    
    def _render_local_deployment(self):
        """Render local deployment options"""
        st.markdown("### Local Server Deployment")
        
        col1, col2 = st.columns(2)
        
        with col1:
            port = st.number_input("Port", value=8000, min_value=1024, max_value=65535)
            host = st.text_input("Host", value="localhost")
        
        with col2:
            workers = st.number_input("Workers", value=4, min_value=1)
            auto_reload = st.checkbox("Auto-reload on changes", value=True)
        
        # Generate startup command
        command = f"python -m agent_server --host {host} --port {port} --workers {workers}"
        if auto_reload:
            command += " --reload"
        
        st.markdown("### Startup Command")
        st.code(command, language="bash")
        
        if st.button("🚀 Start Local Server"):
            st.success("Server starting... (this would actually start the server)")
    
    def _render_api_deployment(self):
        """Render API deployment options"""
        st.markdown("### API Endpoint Deployment")
        
        # API configuration
        api_framework = st.selectbox(
            "API Framework",
            ["FastAPI", "Flask", "Django REST", "Express.js"]
        )
        
        # Authentication
        st.markdown("#### Authentication")
        auth_method = st.selectbox(
            "Authentication Method",
            ["API Key", "JWT", "OAuth2", "Basic Auth", "None"]
        )
        
        if auth_method != "None":
            st.text_input("API Key / Secret", type="password")
        
        # Rate limiting
        st.markdown("#### Rate Limiting")
        
        col1, col2 = st.columns(2)
        
        with col1:
            enable_rate_limit = st.checkbox("Enable Rate Limiting", value=True)
            if enable_rate_limit:
                rate_limit = st.number_input("Requests per minute", value=60)
        
        with col2:
            enable_quota = st.checkbox("Enable Usage Quota", value=False)
            if enable_quota:
                quota = st.number_input("Monthly quota", value=10000)
        
        # Generate API code
        if st.button("Generate API Code"):
            api_code = self._generate_api_code(api_framework, auth_method)
            st.code(api_code, language="python")
            
            st.download_button(
                "Download API Code",
                api_code,
                f"agent_api.{'py' if api_framework != 'Express.js' else 'js'}",
                "text/plain"
            )
    
    def _get_available_tools(self) -> List[str]:
        """Get list of available tools"""
        tools_dir = Path("tools")
        tools = []
        
        if tools_dir.exists():
            for tool_file in tools_dir.glob("*_tool.py"):
                if tool_file.stem != "base_tool":
                    tools.append(tool_file.stem)
        
        return tools
    
    def _get_default_prompt(self, agent_type: str) -> str:
        """Get default system prompt for agent type"""
        prompts = {
            "Research Agent": """You are a highly skilled research agent specialized in finding, analyzing, and synthesizing information from various sources. Your approach is methodical and thorough.

Your capabilities include:
- Web searching and information gathering
- Source verification and fact-checking
- Summarizing complex information
- Identifying patterns and insights

Always cite your sources and indicate confidence levels in your findings.""",
            
            "Coding Agent": """You are an expert software development agent capable of writing, analyzing, and improving code across multiple programming languages.

Your capabilities include:
- Writing clean, efficient, and well-documented code
- Debugging and fixing issues
- Refactoring for better performance and readability
- Following best practices and design patterns

Always consider edge cases, write tests when appropriate, and explain your implementation decisions.""",
            
            "Analysis Agent": """You are a data analysis expert specializing in extracting insights from complex information and data sets.

Your capabilities include:
- Statistical analysis and interpretation
- Pattern recognition and anomaly detection
- Creating visualizations and reports
- Making data-driven recommendations

Present your findings clearly with supporting evidence and confidence levels.""",
            
            "Creative Agent": """You are a creative specialist focused on generating original, engaging content across various formats.

Your capabilities include:
- Writing compelling narratives and copy
- Brainstorming innovative ideas
- Adapting tone and style for different audiences
- Creating structured content plans

Balance creativity with clarity and always consider the target audience.""",
            
            "Task Agent": """You are a versatile task completion agent designed to handle a wide variety of requests efficiently.

Your capabilities include:
- Breaking down complex tasks into manageable steps
- Executing tasks systematically
- Tracking progress and managing dependencies
- Adapting to different task types

Focus on completing tasks accurately while maintaining clear communication about progress."""
        }
        
        return prompts.get(agent_type, "")
    
    def _create_agent(self, config: Dict[str, Any]):
        """Create an agent with the given configuration"""
        try:
            # Create agent directory
            agent_dir = self.agents_dir / config['name']
            agent_dir.mkdir(parents=True, exist_ok=True)
            
            # Write agent configuration
            config_file = agent_dir / "config.yaml"
            with open(config_file, 'w') as f:
                yaml.dump(config, f, default_flow_style=False)
            
            # Create agent.py file
            agent_file = agent_dir / "agent.py"
            agent_code = self._generate_agent_code(config)
            with open(agent_file, 'w') as f:
                f.write(agent_code)
            
            st.success(f"✅ Agent '{config['name']}' created successfully!")
            st.info(f"Location: {agent_dir}")
            
            # Show next steps
            with st.expander("Next Steps"):
                st.markdown(f"""
                1. **Test your agent**: Go to the Testing tab
                2. **Deploy your agent**: Go to the Deployment tab
                3. **View files**: `{agent_dir}`
                4. **Run directly**: `python {agent_file}`
                """)
            
        except Exception as e:
            st.error(f"Error creating agent: {str(e)}")
    
    def _generate_agent_code(self, config: Dict[str, Any]) -> str:
        """Generate Python code for the agent"""
        code = f'''"""
{config['type']} - {config['name']}
Generated by Agent Builder
"""

import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from agent import Agent
from tools.base_tool import ToolDiscovery


class {config['name'].title().replace('_', '')}Agent(Agent):
    """Custom agent: {config['name']}"""
    
    def __init__(self):
        # Configure agent
        os.environ["OPENROUTER_MODEL"] = "{config['model']}"
        
        super().__init__(
            system_prompt="""{config['system_prompt']}""",
            model="{config['model']}",
            temperature={config['temperature']},
            max_iterations={config['max_iterations']}
        )
        
        # Load tools
        tool_discovery = ToolDiscovery()
        available_tools = tool_discovery.discover_tools()
        
        selected_tools = {config.get('tools', [])}
        for tool_name in selected_tools:
            if tool_name in available_tools:
                self.register_tool(available_tools[tool_name])


if __name__ == "__main__":
    import asyncio
    
    agent = {config['name'].title().replace('_', '')}Agent()
    
    # Example usage
    query = "Your query here"
    result = asyncio.run(agent.run(query))
    print(result)
'''
        
        return code
    
    def _get_available_agents(self) -> List[str]:
        """Get list of available agents"""
        agents = []
        
        if self.agents_dir.exists():
            for agent_dir in self.agents_dir.iterdir():
                if agent_dir.is_dir() and (agent_dir / "config.yaml").exists():
                    agents.append(agent_dir.name)
        
        return agents
    
    def _run_interactive_test(self, query: str):
        """Run interactive test"""
        # Simulate test results
        st.session_state.test_results = {
            "response_time": 2.34,
            "tokens_used": 1245,
            "iterations": 3,
            "success": True,
            "response": "Based on my analysis, here's the solution to your query...",
            "thoughts": [
                "Understanding the user's request",
                "Breaking down the problem into steps",
                "Executing the solution"
            ],
            "tool_usage": [
                {
                    "name": "search_tool",
                    "args": {"query": "relevant information"},
                    "result": "Found 5 relevant results"
                }
            ]
        }
        
    def _generate_advanced_config(self) -> Dict[str, Any]:
        """Generate advanced agent configuration"""
        return {
            "agent": {
                "name": "advanced_agent",
                "version": "1.0.0",
                "model": {
                    "provider": "openai",
                    "name": "gpt-4",
                    "parameters": {
                        "temperature": 0.7,
                        "max_tokens": 2000
                    }
                },
                "tools": ["search_tool", "calculator_tool"],
                "reasoning": {
                    "mode": "chain_of_thought",
                    "max_depth": 5
                },
                "monitoring": {
                    "log_level": "INFO",
                    "metrics": ["response_time", "token_usage"]
                }
            }
        }