"""Enhanced Streamlit Dashboard for Agent Creator System"""

import streamlit as st
import yaml
import os
import sys
import subprocess
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Import agent components
from agent import Agent
from orchestrator import MultiAgentOrchestrator
from tools.base_tool import ToolDiscovery

# Import UI components
from components.environment_manager import EnvironmentManager
from components.database_setup import DatabaseSetup
from components.agent_builder import AgentBuilder
from components.tool_library_ui import ToolLibraryUI
from components.mcp_config_ui import MCPConfigUI

# Page configuration
st.set_page_config(
    page_title="Agent Creator Dashboard",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main {
        padding: 0;
    }
    .stButton > button {
        width: 100%;
        margin-top: 10px;
    }
    .status-success {
        background-color: #d4edda;
        color: #155724;
        padding: 10px;
        border-radius: 5px;
        margin: 10px 0;
    }
    .status-error {
        background-color: #f8d7da;
        color: #721c24;
        padding: 10px;
        border-radius: 5px;
        margin: 10px 0;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# Session state initialization
if 'agents' not in st.session_state:
    st.session_state.agents = {}
if 'active_tasks' not in st.session_state:
    st.session_state.active_tasks = {}
if 'logs' not in st.session_state:
    st.session_state.logs = []

class AgentCreatorDashboard:
    """Main dashboard application"""
    
    def __init__(self):
        self.config_path = Path("config.yaml")
        self.load_config()
        
        # Initialize component managers
        self.env_manager = EnvironmentManager()
        self.db_setup = DatabaseSetup()
        self.agent_builder = AgentBuilder()
        self.tool_library = ToolLibraryUI()
        self.mcp_config = MCPConfigUI()
    
    def load_config(self):
        """Load system configuration"""
        if self.config_path.exists():
            with open(self.config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        else:
            self.config = self._default_config()
    
    def _default_config(self) -> Dict[str, Any]:
        """Return default configuration"""
        return {
            'api': {
                'base_url': 'https://openrouter.ai/api/v1',
                'default_model': 'openai/gpt-4'
            },
            'agent': {
                'max_iterations': 10,
                'timeout': 300,
                'temperature': 0.7
            },
            'orchestrator': {
                'parallel_agents': 4,
                'synthesis_model': 'openai/gpt-4'
            }
        }
    
    def run(self):
        """Run the main dashboard"""
        # Sidebar navigation
        with st.sidebar:
            st.title("🤖 Agent Creator")
            st.markdown("---")
            
            page = st.radio(
                "Navigation",
                [
                    "🏠 Dashboard",
                    "🔧 Environment Setup",
                    "🗄️ Database Config",
                    "🤖 Agent Builder",
                    "🛠️ Tool Library",
                    "🔌 MCP Configuration",
                    "📊 Monitoring",
                    "⚙️ Settings"
                ]
            )
            
            st.markdown("---")
            
            # Quick actions
            st.subheader("Quick Actions")
            if st.button("🚀 Run Single Agent"):
                self._run_single_agent()
            
            if st.button("⚡ Run Multi-Agent"):
                self._run_multi_agent()
            
            if st.button("🧪 Run Tests"):
                self._run_tests()
        
        # Main content area
        if page == "🏠 Dashboard":
            self._render_dashboard()
        elif page == "🔧 Environment Setup":
            self.env_manager.render()
        elif page == "🗄️ Database Config":
            self.db_setup.render()
        elif page == "🤖 Agent Builder":
            self.agent_builder.render()
        elif page == "🛠️ Tool Library":
            self.tool_library.render()
        elif page == "🔌 MCP Configuration":
            self.mcp_config.render()
        elif page == "📊 Monitoring":
            self._render_monitoring()
        elif page == "⚙️ Settings":
            self._render_settings()
    
    def _render_dashboard(self):
        """Render main dashboard"""
        st.title("Agent Creator Dashboard")
        
        # Metrics row
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Active Agents",
                len(st.session_state.agents),
                delta="+2" if len(st.session_state.agents) > 0 else None
            )
        
        with col2:
            st.metric(
                "Running Tasks",
                len(st.session_state.active_tasks),
                delta=None
            )
        
        with col3:
            tool_count = len(ToolDiscovery().discover_tools())
            st.metric(
                "Available Tools",
                tool_count,
                delta=None
            )
        
        with col4:
            st.metric(
                "System Status",
                "Operational",
                delta=None
            )
        
        st.markdown("---")
        
        # Recent activity
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Recent Activity")
            
            if st.session_state.logs:
                for log in st.session_state.logs[-10:]:
                    with st.expander(f"{log['timestamp']} - {log['type']}", expanded=False):
                        st.json(log['data'])
            else:
                st.info("No recent activity")
        
        with col2:
            st.subheader("Quick Stats")
            
            # System health
            st.markdown("### System Health")
            health_checks = self._run_health_checks()
            
            for check, status in health_checks.items():
                if status:
                    st.success(f"✅ {check}")
                else:
                    st.error(f"❌ {check}")
        
        # Agent testing area
        st.markdown("---")
        st.subheader("Test Agent")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            query = st.text_input("Enter a query to test agents:", placeholder="e.g., 'Write a Python function to calculate fibonacci'")
        
        with col2:
            agent_type = st.selectbox("Agent Type", ["Single", "Multi-Agent"])
        
        if st.button("🚀 Execute Query"):
            if query:
                self._execute_query(query, agent_type)
            else:
                st.error("Please enter a query")
    
    def _render_monitoring(self):
        """Render monitoring page"""
        st.title("📊 System Monitoring")
        
        # Real-time metrics
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Agent Performance")
            
            # Mock performance data
            performance_data = {
                "Average Response Time": "2.3s",
                "Success Rate": "94.5%",
                "Tokens Used": "45,231",
                "Cache Hit Rate": "78%"
            }
            
            for metric, value in performance_data.items():
                st.metric(metric, value)
        
        with col2:
            st.subheader("Resource Usage")
            
            # Mock resource data
            resource_data = {
                "CPU Usage": "23%",
                "Memory Usage": "1.2 GB",
                "Active Connections": "5",
                "Queue Length": "0"
            }
            
            for metric, value in resource_data.items():
                st.metric(metric, value)
        
        # Activity log
        st.markdown("---")
        st.subheader("Activity Log")
        
        # Filter options
        col1, col2, col3 = st.columns(3)
        
        with col1:
            log_level = st.selectbox("Log Level", ["All", "Info", "Warning", "Error"])
        
        with col2:
            time_range = st.selectbox("Time Range", ["Last Hour", "Last 24 Hours", "Last Week"])
        
        with col3:
            if st.button("🔄 Refresh Logs"):
                st.rerun()
        
        # Display logs
        if st.session_state.logs:
            for log in reversed(st.session_state.logs):
                log_color = {
                    "info": "blue",
                    "success": "green",
                    "warning": "orange",
                    "error": "red"
                }.get(log.get('level', 'info'), 'gray')
                
                st.markdown(
                    f"<div style='color: {log_color}; margin: 5px 0;'>"
                    f"[{log['timestamp']}] {log['type']}: {log.get('message', '')}"
                    f"</div>",
                    unsafe_allow_html=True
                )
        else:
            st.info("No logs available")
    
    def _render_settings(self):
        """Render settings page"""
        st.title("⚙️ Settings")
        
        # API Configuration
        st.subheader("API Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            api_key = st.text_input(
                "OpenRouter API Key",
                value=os.environ.get("OPENROUTER_API_KEY", ""),
                type="password"
            )
            
            if st.button("Save API Key"):
                os.environ["OPENROUTER_API_KEY"] = api_key
                st.success("API Key saved!")
        
        with col2:
            model = st.selectbox(
                "Default Model",
                ["openai/gpt-4", "openai/gpt-3.5-turbo", "anthropic/claude-2", "google/palm-2"]
            )
        
        # Agent Configuration
        st.markdown("---")
        st.subheader("Agent Configuration")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            max_iterations = st.number_input(
                "Max Iterations",
                min_value=1,
                max_value=50,
                value=self.config['agent']['max_iterations']
            )
        
        with col2:
            timeout = st.number_input(
                "Timeout (seconds)",
                min_value=60,
                max_value=3600,
                value=self.config['agent']['timeout']
            )
        
        with col3:
            temperature = st.slider(
                "Temperature",
                min_value=0.0,
                max_value=2.0,
                value=self.config['agent']['temperature'],
                step=0.1
            )
        
        # Orchestrator Configuration
        st.markdown("---")
        st.subheader("Multi-Agent Configuration")
        
        parallel_agents = st.slider(
            "Parallel Agents",
            min_value=2,
            max_value=10,
            value=self.config['orchestrator']['parallel_agents']
        )
        
        if st.button("💾 Save Configuration"):
            self._save_configuration({
                'api': {
                    'base_url': self.config['api']['base_url'],
                    'default_model': model
                },
                'agent': {
                    'max_iterations': max_iterations,
                    'timeout': timeout,
                    'temperature': temperature
                },
                'orchestrator': {
                    'parallel_agents': parallel_agents,
                    'synthesis_model': model
                }
            })
            st.success("Configuration saved!")
    
    def _run_health_checks(self) -> Dict[str, bool]:
        """Run system health checks"""
        checks = {}
        
        # Check API key
        checks["API Key"] = bool(os.environ.get("OPENROUTER_API_KEY"))
        
        # Check config file
        checks["Config File"] = self.config_path.exists()
        
        # Check tools directory
        checks["Tools Available"] = Path("tools").exists()
        
        # Check Python environment
        checks["Python Environment"] = sys.version_info >= (3, 8)
        
        return checks
    
    def _log_activity(self, activity_type: str, data: Any, level: str = "info", message: str = ""):
        """Log activity to session state"""
        log_entry = {
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'type': activity_type,
            'data': data,
            'level': level,
            'message': message
        }
        st.session_state.logs.append(log_entry)
    
    def _execute_query(self, query: str, agent_type: str):
        """Execute a query using selected agent type"""
        with st.spinner(f"Executing {agent_type} query..."):
            try:
                if agent_type == "Single":
                    # Run single agent
                    result = subprocess.run(
                        [sys.executable, "main.py", query],
                        capture_output=True,
                        text=True,
                        cwd=Path(__file__).parent.parent.parent
                    )
                else:
                    # Run multi-agent
                    result = subprocess.run(
                        [sys.executable, "make_it_heavy.py", query],
                        capture_output=True,
                        text=True,
                        cwd=Path(__file__).parent.parent.parent
                    )
                
                if result.returncode == 0:
                    st.success("Query executed successfully!")
                    st.code(result.stdout)
                    self._log_activity(
                        f"{agent_type} Query",
                        {"query": query, "output": result.stdout},
                        "success"
                    )
                else:
                    st.error("Query execution failed!")
                    st.code(result.stderr)
                    self._log_activity(
                        f"{agent_type} Query Error",
                        {"query": query, "error": result.stderr},
                        "error"
                    )
                    
            except Exception as e:
                st.error(f"Error executing query: {str(e)}")
                self._log_activity(
                    "Query Execution Error",
                    {"error": str(e)},
                    "error",
                    str(e)
                )
    
    def _save_configuration(self, config: Dict[str, Any]):
        """Save configuration to file"""
        with open(self.config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        self.config = config
        self._log_activity("Configuration Update", config, "info", "Configuration saved")
    
    def _run_single_agent(self):
        """Quick action to run single agent"""
        st.session_state['quick_action'] = 'single_agent'
        st.rerun()
    
    def _run_multi_agent(self):
        """Quick action to run multi-agent"""
        st.session_state['quick_action'] = 'multi_agent'
        st.rerun()
    
    def _run_tests(self):
        """Quick action to run tests"""
        with st.spinner("Running tests..."):
            result = subprocess.run(
                ["pytest", "-v"],
                capture_output=True,
                text=True,
                cwd=Path(__file__).parent.parent.parent
            )
            
            if result.returncode == 0:
                st.success("All tests passed!")
                self._log_activity("Test Run", {"status": "passed"}, "success")
            else:
                st.error("Some tests failed!")
                self._log_activity("Test Run", {"status": "failed", "output": result.stdout}, "error")


def main():
    """Main entry point"""
    dashboard = AgentCreatorDashboard()
    dashboard.run()


if __name__ == "__main__":
    main()