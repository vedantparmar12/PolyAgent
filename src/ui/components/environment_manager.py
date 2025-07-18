"""Environment Variable Management UI Component"""

import streamlit as st
import os
import yaml
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
import subprocess
import re


class EnvironmentManager:
    """UI component for managing environment variables and configuration"""
    
    def __init__(self):
        self.env_file = Path(".env")
        self.env_template_file = Path(".env.template")
        self.secrets_file = Path(".streamlit/secrets.toml")
        self.supported_providers = [
            "OpenRouter",
            "OpenAI",
            "Anthropic",
            "Google",
            "AWS Bedrock",
            "Azure OpenAI",
            "Local (Ollama)"
        ]
    
    def render(self):
        """Render the environment management interface"""
        st.title("🔧 Environment Setup")
        st.markdown("Configure API keys, environment variables, and system settings")
        
        # Create tabs for different sections
        tab1, tab2, tab3, tab4 = st.tabs([
            "API Keys",
            "Environment Variables",
            "System Check",
            "Export/Import"
        ])
        
        with tab1:
            self._render_api_keys()
        
        with tab2:
            self._render_env_variables()
        
        with tab3:
            self._render_system_check()
        
        with tab4:
            self._render_export_import()
    
    def _render_api_keys(self):
        """Render API keys configuration"""
        st.subheader("API Provider Configuration")
        
        # Provider selection
        provider = st.selectbox(
            "Select API Provider",
            self.supported_providers,
            help="Choose your AI model provider"
        )
        
        st.markdown("---")
        
        if provider == "OpenRouter":
            self._configure_openrouter()
        elif provider == "OpenAI":
            self._configure_openai()
        elif provider == "Anthropic":
            self._configure_anthropic()
        elif provider == "Google":
            self._configure_google()
        elif provider == "AWS Bedrock":
            self._configure_aws_bedrock()
        elif provider == "Azure OpenAI":
            self._configure_azure_openai()
        elif provider == "Local (Ollama)":
            self._configure_ollama()
        
        # Common configuration
        st.markdown("---")
        st.subheader("Common Settings")
        
        col1, col2 = st.columns(2)
        
        with col1:
            default_model = st.text_input(
                "Default Model",
                value=os.environ.get("DEFAULT_MODEL", "openai/gpt-4"),
                help="Default model to use for agent operations"
            )
        
        with col2:
            request_timeout = st.number_input(
                "Request Timeout (seconds)",
                min_value=10,
                max_value=600,
                value=int(os.environ.get("REQUEST_TIMEOUT", "120")),
                help="Maximum time to wait for API responses"
            )
        
        if st.button("💾 Save Common Settings"):
            self._save_env_var("DEFAULT_MODEL", default_model)
            self._save_env_var("REQUEST_TIMEOUT", str(request_timeout))
            st.success("Common settings saved!")
    
    def _configure_openrouter(self):
        """Configure OpenRouter API settings"""
        st.markdown("### OpenRouter Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            api_key = st.text_input(
                "OpenRouter API Key",
                value=os.environ.get("OPENROUTER_API_KEY", ""),
                type="password",
                help="Get your API key from https://openrouter.ai/keys"
            )
            
            base_url = st.text_input(
                "Base URL",
                value=os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
                help="OpenRouter API base URL"
            )
        
        with col2:
            site_url = st.text_input(
                "Site URL (Optional)",
                value=os.environ.get("OPENROUTER_SITE_URL", ""),
                help="Your site URL for OpenRouter analytics"
            )
            
            site_name = st.text_input(
                "Site Name (Optional)",
                value=os.environ.get("OPENROUTER_SITE_NAME", ""),
                help="Your site name for OpenRouter analytics"
            )
        
        if st.button("💾 Save OpenRouter Settings"):
            self._save_env_var("OPENROUTER_API_KEY", api_key)
            self._save_env_var("OPENROUTER_BASE_URL", base_url)
            if site_url:
                self._save_env_var("OPENROUTER_SITE_URL", site_url)
            if site_name:
                self._save_env_var("OPENROUTER_SITE_NAME", site_name)
            st.success("OpenRouter settings saved!")
            
            # Test connection
            if st.button("🧪 Test Connection"):
                self._test_openrouter_connection(api_key, base_url)
    
    def _configure_openai(self):
        """Configure OpenAI API settings"""
        st.markdown("### OpenAI Configuration")
        
        api_key = st.text_input(
            "OpenAI API Key",
            value=os.environ.get("OPENAI_API_KEY", ""),
            type="password",
            help="Get your API key from https://platform.openai.com/api-keys"
        )
        
        organization = st.text_input(
            "Organization ID (Optional)",
            value=os.environ.get("OPENAI_ORGANIZATION", ""),
            help="Your OpenAI organization ID"
        )
        
        if st.button("💾 Save OpenAI Settings"):
            self._save_env_var("OPENAI_API_KEY", api_key)
            if organization:
                self._save_env_var("OPENAI_ORGANIZATION", organization)
            st.success("OpenAI settings saved!")
    
    def _configure_anthropic(self):
        """Configure Anthropic API settings"""
        st.markdown("### Anthropic Configuration")
        
        api_key = st.text_input(
            "Anthropic API Key",
            value=os.environ.get("ANTHROPIC_API_KEY", ""),
            type="password",
            help="Get your API key from https://console.anthropic.com/settings/keys"
        )
        
        if st.button("💾 Save Anthropic Settings"):
            self._save_env_var("ANTHROPIC_API_KEY", api_key)
            st.success("Anthropic settings saved!")
    
    def _configure_google(self):
        """Configure Google AI settings"""
        st.markdown("### Google AI Configuration")
        
        api_key = st.text_input(
            "Google AI API Key",
            value=os.environ.get("GOOGLE_API_KEY", ""),
            type="password",
            help="Get your API key from Google AI Studio"
        )
        
        if st.button("💾 Save Google Settings"):
            self._save_env_var("GOOGLE_API_KEY", api_key)
            st.success("Google AI settings saved!")
    
    def _configure_aws_bedrock(self):
        """Configure AWS Bedrock settings"""
        st.markdown("### AWS Bedrock Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            access_key = st.text_input(
                "AWS Access Key ID",
                value=os.environ.get("AWS_ACCESS_KEY_ID", ""),
                type="password"
            )
            
            secret_key = st.text_input(
                "AWS Secret Access Key",
                value=os.environ.get("AWS_SECRET_ACCESS_KEY", ""),
                type="password"
            )
        
        with col2:
            region = st.selectbox(
                "AWS Region",
                ["us-east-1", "us-west-2", "eu-west-1", "ap-southeast-1"],
                index=0
            )
            
            session_token = st.text_input(
                "Session Token (Optional)",
                value=os.environ.get("AWS_SESSION_TOKEN", ""),
                type="password"
            )
        
        if st.button("💾 Save AWS Bedrock Settings"):
            self._save_env_var("AWS_ACCESS_KEY_ID", access_key)
            self._save_env_var("AWS_SECRET_ACCESS_KEY", secret_key)
            self._save_env_var("AWS_DEFAULT_REGION", region)
            if session_token:
                self._save_env_var("AWS_SESSION_TOKEN", session_token)
            st.success("AWS Bedrock settings saved!")
    
    def _configure_azure_openai(self):
        """Configure Azure OpenAI settings"""
        st.markdown("### Azure OpenAI Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            api_key = st.text_input(
                "Azure OpenAI API Key",
                value=os.environ.get("AZURE_OPENAI_API_KEY", ""),
                type="password"
            )
            
            endpoint = st.text_input(
                "Azure OpenAI Endpoint",
                value=os.environ.get("AZURE_OPENAI_ENDPOINT", ""),
                placeholder="https://your-resource.openai.azure.com/"
            )
        
        with col2:
            deployment = st.text_input(
                "Deployment Name",
                value=os.environ.get("AZURE_OPENAI_DEPLOYMENT", ""),
                help="Your model deployment name"
            )
            
            api_version = st.text_input(
                "API Version",
                value=os.environ.get("AZURE_OPENAI_API_VERSION", "2024-02-01"),
                help="Azure OpenAI API version"
            )
        
        if st.button("💾 Save Azure Settings"):
            self._save_env_var("AZURE_OPENAI_API_KEY", api_key)
            self._save_env_var("AZURE_OPENAI_ENDPOINT", endpoint)
            self._save_env_var("AZURE_OPENAI_DEPLOYMENT", deployment)
            self._save_env_var("AZURE_OPENAI_API_VERSION", api_version)
            st.success("Azure OpenAI settings saved!")
    
    def _configure_ollama(self):
        """Configure local Ollama settings"""
        st.markdown("### Ollama (Local) Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            base_url = st.text_input(
                "Ollama Base URL",
                value=os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434"),
                help="Local Ollama server URL"
            )
            
            model = st.text_input(
                "Default Model",
                value=os.environ.get("OLLAMA_MODEL", "llama2"),
                help="Default Ollama model to use"
            )
        
        with col2:
            # Check if Ollama is running
            if st.button("🔍 Check Ollama Status"):
                self._check_ollama_status(base_url)
            
            # List available models
            if st.button("📋 List Available Models"):
                self._list_ollama_models(base_url)
        
        if st.button("💾 Save Ollama Settings"):
            self._save_env_var("OLLAMA_BASE_URL", base_url)
            self._save_env_var("OLLAMA_MODEL", model)
            st.success("Ollama settings saved!")
    
    def _render_env_variables(self):
        """Render custom environment variables section"""
        st.subheader("Custom Environment Variables")
        st.markdown("Add custom environment variables for your agents and tools")
        
        # Load existing env vars
        env_vars = self._load_env_vars()
        
        # Display existing variables
        if env_vars:
            st.markdown("### Existing Variables")
            
            # Create a table of variables
            for key, value in env_vars.items():
                col1, col2, col3 = st.columns([2, 3, 1])
                
                with col1:
                    st.text(key)
                
                with col2:
                    # Mask sensitive values
                    if any(sensitive in key.upper() for sensitive in ["KEY", "SECRET", "PASSWORD", "TOKEN"]):
                        st.text("*" * min(len(value), 20))
                    else:
                        st.text(value[:50] + "..." if len(value) > 50 else value)
                
                with col3:
                    if st.button("🗑️", key=f"delete_{key}"):
                        self._delete_env_var(key)
                        st.rerun()
        
        # Add new variable
        st.markdown("### Add New Variable")
        
        col1, col2 = st.columns(2)
        
        with col1:
            new_key = st.text_input("Variable Name", placeholder="MY_VARIABLE")
        
        with col2:
            new_value = st.text_input("Variable Value", placeholder="value")
        
        if st.button("➕ Add Variable"):
            if new_key and new_value:
                self._save_env_var(new_key, new_value)
                st.success(f"Added {new_key}")
                st.rerun()
            else:
                st.error("Both name and value are required")
    
    def _render_system_check(self):
        """Render system requirements check"""
        st.subheader("System Requirements Check")
        st.markdown("Verify that your system meets all requirements")
        
        checks = self._run_system_checks()
        
        # Display results
        for category, category_checks in checks.items():
            st.markdown(f"### {category}")
            
            for check_name, check_result in category_checks.items():
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    if check_result['status']:
                        st.success(f"✅ {check_name}: {check_result['message']}")
                    else:
                        st.error(f"❌ {check_name}: {check_result['message']}")
                
                with col2:
                    if check_result.get('action'):
                        if st.button(check_result['action']['label'], key=f"fix_{check_name}"):
                            self._execute_fix_action(check_result['action'])
    
    def _render_export_import(self):
        """Render export/import functionality"""
        st.subheader("Configuration Export/Import")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Export Configuration")
            st.markdown("Export all environment variables and settings")
            
            export_format = st.selectbox(
                "Export Format",
                ["Environment File (.env)", "JSON", "YAML"]
            )
            
            include_sensitive = st.checkbox(
                "Include sensitive values",
                value=False,
                help="Include API keys and secrets in export"
            )
            
            if st.button("📤 Export Configuration"):
                export_data = self._export_configuration(export_format, include_sensitive)
                
                if export_format == "Environment File (.env)":
                    st.download_button(
                        "Download .env file",
                        export_data,
                        "environment.env",
                        "text/plain"
                    )
                elif export_format == "JSON":
                    st.download_button(
                        "Download JSON",
                        export_data,
                        "configuration.json",
                        "application/json"
                    )
                else:  # YAML
                    st.download_button(
                        "Download YAML",
                        export_data,
                        "configuration.yaml",
                        "text/yaml"
                    )
        
        with col2:
            st.markdown("### Import Configuration")
            st.markdown("Import environment variables from file")
            
            uploaded_file = st.file_uploader(
                "Choose a file",
                type=['env', 'json', 'yaml', 'yml']
            )
            
            if uploaded_file is not None:
                if st.button("📥 Import Configuration"):
                    self._import_configuration(uploaded_file)
                    st.success("Configuration imported successfully!")
                    st.rerun()
    
    def _save_env_var(self, key: str, value: str):
        """Save environment variable to .env file"""
        # Set in current environment
        os.environ[key] = value
        
        # Save to .env file
        env_vars = self._load_env_vars()
        env_vars[key] = value
        
        with open(self.env_file, 'w') as f:
            for k, v in env_vars.items():
                f.write(f"{k}={v}\n")
    
    def _delete_env_var(self, key: str):
        """Delete environment variable"""
        # Remove from current environment
        if key in os.environ:
            del os.environ[key]
        
        # Remove from .env file
        env_vars = self._load_env_vars()
        if key in env_vars:
            del env_vars[key]
        
        with open(self.env_file, 'w') as f:
            for k, v in env_vars.items():
                f.write(f"{k}={v}\n")
    
    def _load_env_vars(self) -> Dict[str, str]:
        """Load environment variables from .env file"""
        env_vars = {}
        
        if self.env_file.exists():
            with open(self.env_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        key, value = line.split('=', 1)
                        env_vars[key.strip()] = value.strip()
        
        return env_vars
    
    def _run_system_checks(self) -> Dict[str, Dict[str, Any]]:
        """Run comprehensive system checks"""
        checks = {
            "Python Environment": {},
            "Dependencies": {},
            "API Access": {},
            "File System": {}
        }
        
        # Python version check
        import sys
        python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        checks["Python Environment"]["Python Version"] = {
            "status": sys.version_info >= (3, 8),
            "message": f"Python {python_version} (requires 3.8+)"
        }
        
        # Required packages check
        required_packages = ["streamlit", "openai", "pydantic", "yaml", "requests"]
        for package in required_packages:
            try:
                __import__(package)
                checks["Dependencies"][package] = {
                    "status": True,
                    "message": "Installed"
                }
            except ImportError:
                checks["Dependencies"][package] = {
                    "status": False,
                    "message": "Not installed",
                    "action": {
                        "label": "Install",
                        "command": f"pip install {package}"
                    }
                }
        
        # API key checks
        api_keys = {
            "OpenRouter API": "OPENROUTER_API_KEY",
            "OpenAI API": "OPENAI_API_KEY",
            "Anthropic API": "ANTHROPIC_API_KEY"
        }
        
        for name, env_var in api_keys.items():
            if os.environ.get(env_var):
                checks["API Access"][name] = {
                    "status": True,
                    "message": "Configured"
                }
            else:
                checks["API Access"][name] = {
                    "status": False,
                    "message": "Not configured"
                }
        
        # File system checks
        important_paths = {
            "Tools Directory": Path("tools"),
            "Context Directory": Path("context"),
            "Config File": Path("config.yaml")
        }
        
        for name, path in important_paths.items():
            if path.exists():
                checks["File System"][name] = {
                    "status": True,
                    "message": f"Found at {path}"
                }
            else:
                checks["File System"][name] = {
                    "status": False,
                    "message": f"Not found at {path}"
                }
        
        return checks
    
    def _test_openrouter_connection(self, api_key: str, base_url: str):
        """Test OpenRouter API connection"""
        import requests
        
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        try:
            response = requests.get(f"{base_url}/models", headers=headers)
            if response.status_code == 200:
                st.success("✅ Connection successful!")
                models = response.json()
                st.info(f"Found {len(models.get('data', []))} available models")
            else:
                st.error(f"❌ Connection failed: {response.status_code}")
        except Exception as e:
            st.error(f"❌ Connection error: {str(e)}")
    
    def _check_ollama_status(self, base_url: str):
        """Check if Ollama is running"""
        import requests
        
        try:
            response = requests.get(f"{base_url}/api/tags")
            if response.status_code == 200:
                st.success("✅ Ollama is running")
            else:
                st.error("❌ Ollama is not responding")
        except Exception as e:
            st.error(f"❌ Cannot connect to Ollama: {str(e)}")
    
    def _list_ollama_models(self, base_url: str):
        """List available Ollama models"""
        import requests
        
        try:
            response = requests.get(f"{base_url}/api/tags")
            if response.status_code == 200:
                models = response.json().get('models', [])
                if models:
                    st.success(f"Found {len(models)} models:")
                    for model in models:
                        st.text(f"• {model['name']}")
                else:
                    st.warning("No models found. Pull a model with: ollama pull llama2")
            else:
                st.error("Failed to get model list")
        except Exception as e:
            st.error(f"Error: {str(e)}")
    
    def _export_configuration(self, format: str, include_sensitive: bool) -> str:
        """Export configuration in specified format"""
        env_vars = self._load_env_vars()
        
        # Filter sensitive values if requested
        if not include_sensitive:
            filtered_vars = {}
            for key, value in env_vars.items():
                if any(sensitive in key.upper() for sensitive in ["KEY", "SECRET", "PASSWORD", "TOKEN"]):
                    filtered_vars[key] = "***REDACTED***"
                else:
                    filtered_vars[key] = value
            env_vars = filtered_vars
        
        if format == "Environment File (.env)":
            return "\n".join([f"{k}={v}" for k, v in env_vars.items()])
        elif format == "JSON":
            return json.dumps(env_vars, indent=2)
        else:  # YAML
            return yaml.dump(env_vars, default_flow_style=False)
    
    def _import_configuration(self, uploaded_file):
        """Import configuration from file"""
        content = uploaded_file.read().decode('utf-8')
        
        if uploaded_file.name.endswith('.json'):
            env_vars = json.loads(content)
        elif uploaded_file.name.endswith(('.yaml', '.yml')):
            env_vars = yaml.safe_load(content)
        else:  # .env format
            env_vars = {}
            for line in content.split('\n'):
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    env_vars[key.strip()] = value.strip()
        
        # Save all variables
        for key, value in env_vars.items():
            if value != "***REDACTED***":  # Skip redacted values
                self._save_env_var(key, value)
    
    def _execute_fix_action(self, action: Dict[str, str]):
        """Execute a fix action"""
        if 'command' in action:
            result = subprocess.run(
                action['command'].split(),
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                st.success(f"Successfully executed: {action['command']}")
            else:
                st.error(f"Failed to execute: {result.stderr}")