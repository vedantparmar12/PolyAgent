"""MCP (Model Context Protocol) configuration interface"""

import streamlit as st
from typing import Dict, Any, List, Optional
import json
import yaml
from pathlib import Path
from datetime import datetime


class MCPConfigUI:
    """MCP configuration interface for VS Code/Cursor integration"""
    
    def __init__(self):
        self.config_dir = Path("config/mcp")
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self.servers_config_path = self.config_dir / "servers.json"
        self.tools_config_path = self.config_dir / "tools.json"
        
    def render(self):
        """Render the MCP configuration interface"""
        st.header("🔌 MCP Configuration")
        st.markdown("""
        Configure Model Context Protocol (MCP) for seamless integration with VS Code, Cursor, and other AI IDEs.
        """)
        
        # Tab selection
        tab1, tab2, tab3, tab4 = st.tabs([
            "🖥️ Server Configuration",
            "🛠️ Tools Registration", 
            "🔐 Authentication",
            "📋 Export Config"
        ])
        
        with tab1:
            self._render_server_config()
            
        with tab2:
            self._render_tools_registration()
            
        with tab3:
            self._render_authentication()
            
        with tab4:
            self._render_export_config()
    
    def _render_server_config(self):
        """Render MCP server configuration"""
        st.subheader("MCP Server Configuration")
        
        # Load existing config
        servers = self._load_servers_config()
        
        # Server settings
        col1, col2 = st.columns(2)
        
        with col1:
            server_name = st.text_input(
                "Server Name",
                value=servers.get("name", "enhanced-agentic-workflow"),
                help="Unique identifier for your MCP server"
            )
            
            server_port = st.number_input(
                "Server Port",
                min_value=1024,
                max_value=65535,
                value=servers.get("port", 8765),
                help="Port for MCP server to listen on"
            )
            
            enable_cors = st.checkbox(
                "Enable CORS",
                value=servers.get("enable_cors", True),
                help="Allow cross-origin requests from IDEs"
            )
        
        with col2:
            server_host = st.text_input(
                "Server Host",
                value=servers.get("host", "localhost"),
                help="Host address for MCP server"
            )
            
            max_connections = st.number_input(
                "Max Connections",
                min_value=1,
                max_value=100,
                value=servers.get("max_connections", 10),
                help="Maximum concurrent IDE connections"
            )
            
            enable_ssl = st.checkbox(
                "Enable SSL/TLS",
                value=servers.get("enable_ssl", False),
                help="Use secure connections"
            )
        
        # Advanced settings
        with st.expander("Advanced Settings"):
            timeout = st.number_input(
                "Request Timeout (seconds)",
                min_value=5,
                max_value=300,
                value=servers.get("timeout", 60),
                help="Maximum time for request processing"
            )
            
            buffer_size = st.number_input(
                "Buffer Size (KB)",
                min_value=64,
                max_value=1024,
                value=servers.get("buffer_size", 256),
                help="Message buffer size in kilobytes"
            )
            
            enable_logging = st.checkbox(
                "Enable Debug Logging",
                value=servers.get("enable_logging", True),
                help="Log all MCP communications"
            )
            
            allowed_origins = st.text_area(
                "Allowed Origins (one per line)",
                value="\n".join(servers.get("allowed_origins", ["*"])),
                help="Allowed CORS origins for IDE connections"
            )
        
        # Save configuration
        if st.button("💾 Save Server Configuration", key="save_server_config"):
            servers_config = {
                "name": server_name,
                "host": server_host,
                "port": server_port,
                "enable_cors": enable_cors,
                "enable_ssl": enable_ssl,
                "max_connections": max_connections,
                "timeout": timeout,
                "buffer_size": buffer_size,
                "enable_logging": enable_logging,
                "allowed_origins": [origin.strip() for origin in allowed_origins.split("\n") if origin.strip()],
                "updated_at": datetime.utcnow().isoformat()
            }
            
            self._save_servers_config(servers_config)
            st.success("✅ Server configuration saved successfully!")
    
    def _render_tools_registration(self):
        """Render tools registration interface"""
        st.subheader("Tools Registration")
        st.markdown("""
        Register tools that will be exposed through MCP to your IDE.
        """)
        
        # Load existing tools
        tools = self._load_tools_config()
        
        # Tool categories
        categories = ["Code Generation", "Testing", "Deployment", "Analysis", "Documentation", "Custom"]
        
        # Add new tool
        st.markdown("### Register New Tool")
        
        col1, col2 = st.columns(2)
        
        with col1:
            tool_name = st.text_input(
                "Tool Name",
                placeholder="e.g., generate_api_endpoint",
                help="Unique identifier for the tool"
            )
            
            tool_category = st.selectbox(
                "Category",
                categories,
                help="Tool category for organization"
            )
            
            tool_description = st.text_area(
                "Description",
                placeholder="Describe what this tool does...",
                help="Clear description for IDE users"
            )
        
        with col2:
            # Tool parameters
            st.markdown("**Parameters**")
            param_count = st.number_input(
                "Number of Parameters",
                min_value=0,
                max_value=10,
                value=1,
                help="Number of input parameters"
            )
            
            parameters = []
            for i in range(param_count):
                with st.container():
                    param_name = st.text_input(
                        f"Parameter {i+1} Name",
                        key=f"param_name_{i}"
                    )
                    param_type = st.selectbox(
                        f"Parameter {i+1} Type",
                        ["string", "number", "boolean", "object", "array"],
                        key=f"param_type_{i}"
                    )
                    param_required = st.checkbox(
                        f"Required",
                        key=f"param_required_{i}"
                    )
                    
                    if param_name:
                        parameters.append({
                            "name": param_name,
                            "type": param_type,
                            "required": param_required
                        })
        
        # Tool implementation
        st.markdown("**Implementation**")
        implementation_type = st.selectbox(
            "Implementation Type",
            ["Python Function", "External API", "Shell Command", "Agent Chain"]
        )
        
        if implementation_type == "Python Function":
            module_path = st.text_input(
                "Module Path",
                placeholder="e.g., src.tools.api_generator",
                help="Python module containing the function"
            )
            function_name = st.text_input(
                "Function Name",
                placeholder="e.g., generate_endpoint",
                help="Function to call"
            )
        elif implementation_type == "External API":
            api_endpoint = st.text_input(
                "API Endpoint",
                placeholder="https://api.example.com/generate",
                help="External API endpoint"
            )
            api_method = st.selectbox(
                "HTTP Method",
                ["GET", "POST", "PUT", "DELETE"]
            )
        elif implementation_type == "Shell Command":
            shell_command = st.text_input(
                "Shell Command",
                placeholder="e.g., python scripts/generate.py",
                help="Shell command to execute"
            )
        else:  # Agent Chain
            agent_chain = st.text_area(
                "Agent Chain Configuration",
                placeholder="Define agent chain in YAML format",
                help="Agent orchestration configuration"
            )
        
        # Register tool
        if st.button("➕ Register Tool", key="register_tool"):
            if tool_name and tool_description:
                new_tool = {
                    "name": tool_name,
                    "category": tool_category,
                    "description": tool_description,
                    "parameters": parameters,
                    "implementation": {
                        "type": implementation_type,
                        "config": self._get_implementation_config(implementation_type, locals())
                    },
                    "enabled": True,
                    "created_at": datetime.utcnow().isoformat()
                }
                
                tools[tool_name] = new_tool
                self._save_tools_config(tools)
                st.success(f"✅ Tool '{tool_name}' registered successfully!")
                st.rerun()
            else:
                st.error("❌ Please provide tool name and description")
        
        # Display existing tools
        st.markdown("### Registered Tools")
        
        if tools:
            # Filter by category
            selected_category = st.selectbox(
                "Filter by Category",
                ["All"] + categories,
                key="filter_category"
            )
            
            filtered_tools = tools
            if selected_category != "All":
                filtered_tools = {
                    name: tool for name, tool in tools.items()
                    if tool.get("category") == selected_category
                }
            
            # Display tools
            for tool_name, tool_info in filtered_tools.items():
                with st.expander(f"🛠️ {tool_name}"):
                    col1, col2, col3 = st.columns([3, 1, 1])
                    
                    with col1:
                        st.markdown(f"**Category:** {tool_info.get('category', 'Unknown')}")
                        st.markdown(f"**Description:** {tool_info.get('description', 'No description')}")
                        
                        # Parameters
                        params = tool_info.get('parameters', [])
                        if params:
                            st.markdown("**Parameters:**")
                            for param in params:
                                req = "✓" if param.get('required') else "○"
                                st.markdown(f"- {req} `{param['name']}` ({param['type']})")
                    
                    with col2:
                        # Enable/Disable toggle
                        enabled = st.checkbox(
                            "Enabled",
                            value=tool_info.get("enabled", True),
                            key=f"enable_{tool_name}"
                        )
                        
                        if enabled != tool_info.get("enabled", True):
                            tool_info["enabled"] = enabled
                            self._save_tools_config(tools)
                    
                    with col3:
                        # Delete button
                        if st.button("🗑️ Delete", key=f"delete_{tool_name}"):
                            del tools[tool_name]
                            self._save_tools_config(tools)
                            st.success(f"Tool '{tool_name}' deleted")
                            st.rerun()
        else:
            st.info("No tools registered yet. Register your first tool above!")
    
    def _render_authentication(self):
        """Render authentication configuration"""
        st.subheader("Authentication Configuration")
        
        auth_config = self._load_auth_config()
        
        # Authentication method
        auth_method = st.selectbox(
            "Authentication Method",
            ["None", "API Key", "OAuth 2.0", "JWT", "Custom"],
            index=["None", "API Key", "OAuth 2.0", "JWT", "Custom"].index(
                auth_config.get("method", "API Key")
            ),
            help="Choose authentication method for MCP connections"
        )
        
        if auth_method == "API Key":
            st.markdown("### API Key Configuration")
            
            api_key_header = st.text_input(
                "API Key Header Name",
                value=auth_config.get("api_key_header", "X-API-Key"),
                help="HTTP header name for API key"
            )
            
            # Generate new API key
            if st.button("🔑 Generate New API Key"):
                import secrets
                new_key = secrets.token_urlsafe(32)
                st.code(new_key)
                st.info("Save this API key securely. It won't be shown again!")
                
                # Save hashed version
                import hashlib
                hashed_key = hashlib.sha256(new_key.encode()).hexdigest()
                auth_config["api_keys"] = auth_config.get("api_keys", [])
                auth_config["api_keys"].append({
                    "hash": hashed_key,
                    "created_at": datetime.utcnow().isoformat(),
                    "active": True
                })
            
            # Manage existing keys
            st.markdown("### Active API Keys")
            api_keys = auth_config.get("api_keys", [])
            
            if api_keys:
                for i, key_info in enumerate(api_keys):
                    if key_info.get("active", True):
                        col1, col2, col3 = st.columns([3, 1, 1])
                        
                        with col1:
                            st.text(f"Key {i+1}: ...{key_info['hash'][-8:]}")
                            st.caption(f"Created: {key_info.get('created_at', 'Unknown')}")
                        
                        with col2:
                            if st.button("🚫 Revoke", key=f"revoke_{i}"):
                                key_info["active"] = False
                                key_info["revoked_at"] = datetime.utcnow().isoformat()
                                self._save_auth_config(auth_config)
                                st.success("Key revoked")
                                st.rerun()
        
        elif auth_method == "OAuth 2.0":
            st.markdown("### OAuth 2.0 Configuration")
            
            oauth_provider = st.selectbox(
                "OAuth Provider",
                ["GitHub", "Google", "Custom"],
                help="OAuth 2.0 provider"
            )
            
            client_id = st.text_input(
                "Client ID",
                value=auth_config.get("oauth_client_id", ""),
                help="OAuth client ID"
            )
            
            client_secret = st.text_input(
                "Client Secret",
                type="password",
                value=auth_config.get("oauth_client_secret", ""),
                help="OAuth client secret"
            )
            
            redirect_uri = st.text_input(
                "Redirect URI",
                value=auth_config.get("oauth_redirect_uri", "http://localhost:8765/callback"),
                help="OAuth redirect URI"
            )
            
            scopes = st.text_input(
                "Scopes (comma-separated)",
                value=", ".join(auth_config.get("oauth_scopes", [])),
                help="Required OAuth scopes"
            )
        
        elif auth_method == "JWT":
            st.markdown("### JWT Configuration")
            
            jwt_secret = st.text_input(
                "JWT Secret",
                type="password",
                value=auth_config.get("jwt_secret", ""),
                help="Secret key for JWT signing"
            )
            
            jwt_algorithm = st.selectbox(
                "Algorithm",
                ["HS256", "HS384", "HS512", "RS256", "RS384", "RS512"],
                index=0,
                help="JWT signing algorithm"
            )
            
            jwt_expiry = st.number_input(
                "Token Expiry (minutes)",
                min_value=5,
                max_value=10080,  # 1 week
                value=auth_config.get("jwt_expiry_minutes", 60),
                help="JWT token expiration time"
            )
        
        # Rate limiting
        st.markdown("### Rate Limiting")
        
        enable_rate_limiting = st.checkbox(
            "Enable Rate Limiting",
            value=auth_config.get("rate_limiting_enabled", True),
            help="Limit requests per client"
        )
        
        if enable_rate_limiting:
            col1, col2 = st.columns(2)
            
            with col1:
                requests_per_minute = st.number_input(
                    "Requests per Minute",
                    min_value=1,
                    max_value=1000,
                    value=auth_config.get("requests_per_minute", 60),
                    help="Maximum requests per minute per client"
                )
            
            with col2:
                burst_size = st.number_input(
                    "Burst Size",
                    min_value=1,
                    max_value=100,
                    value=auth_config.get("burst_size", 10),
                    help="Maximum burst requests allowed"
                )
        
        # Save authentication config
        if st.button("💾 Save Authentication Settings", key="save_auth"):
            auth_config.update({
                "method": auth_method,
                "api_key_header": api_key_header if auth_method == "API Key" else None,
                "oauth_client_id": client_id if auth_method == "OAuth 2.0" else None,
                "oauth_client_secret": client_secret if auth_method == "OAuth 2.0" else None,
                "oauth_redirect_uri": redirect_uri if auth_method == "OAuth 2.0" else None,
                "oauth_scopes": [s.strip() for s in scopes.split(",")] if auth_method == "OAuth 2.0" else [],
                "jwt_secret": jwt_secret if auth_method == "JWT" else None,
                "jwt_algorithm": jwt_algorithm if auth_method == "JWT" else None,
                "jwt_expiry_minutes": jwt_expiry if auth_method == "JWT" else None,
                "rate_limiting_enabled": enable_rate_limiting,
                "requests_per_minute": requests_per_minute if enable_rate_limiting else None,
                "burst_size": burst_size if enable_rate_limiting else None,
                "updated_at": datetime.utcnow().isoformat()
            })
            
            self._save_auth_config(auth_config)
            st.success("✅ Authentication settings saved successfully!")
    
    def _render_export_config(self):
        """Render configuration export interface"""
        st.subheader("Export MCP Configuration")
        st.markdown("""
        Export your MCP configuration for use in VS Code, Cursor, or other compatible IDEs.
        """)
        
        # Export format
        export_format = st.selectbox(
            "Export Format",
            ["VS Code Settings", "Cursor Config", "Raw JSON", "Docker Compose"],
            help="Choose export format for your IDE"
        )
        
        # Generate configuration
        if st.button("📋 Generate Configuration", key="generate_config"):
            servers_config = self._load_servers_config()
            tools_config = self._load_tools_config()
            auth_config = self._load_auth_config()
            
            if export_format == "VS Code Settings":
                config = self._generate_vscode_config(servers_config, tools_config, auth_config)
                st.markdown("### VS Code Settings")
                st.markdown("Add this to your `.vscode/settings.json`:")
                st.code(json.dumps(config, indent=2), language="json")
                
            elif export_format == "Cursor Config":
                config = self._generate_cursor_config(servers_config, tools_config, auth_config)
                st.markdown("### Cursor Configuration")
                st.markdown("Add this to your Cursor settings:")
                st.code(json.dumps(config, indent=2), language="json")
                
            elif export_format == "Raw JSON":
                config = {
                    "servers": servers_config,
                    "tools": tools_config,
                    "auth": auth_config
                }
                st.markdown("### Raw MCP Configuration")
                st.code(json.dumps(config, indent=2), language="json")
                
                # Download button
                st.download_button(
                    label="📥 Download Configuration",
                    data=json.dumps(config, indent=2),
                    file_name="mcp-config.json",
                    mime="application/json"
                )
                
            else:  # Docker Compose
                compose_config = self._generate_docker_compose(servers_config, tools_config, auth_config)
                st.markdown("### Docker Compose Configuration")
                st.code(compose_config, language="yaml")
                
                # Download button
                st.download_button(
                    label="📥 Download docker-compose.yml",
                    data=compose_config,
                    file_name="docker-compose.mcp.yml",
                    mime="text/yaml"
                )
        
        # Connection test
        st.markdown("### Test Connection")
        
        if st.button("🔍 Test MCP Connection", key="test_connection"):
            with st.spinner("Testing connection..."):
                test_result = self._test_mcp_connection()
                
                if test_result["success"]:
                    st.success("✅ MCP server is accessible!")
                    st.json(test_result["details"])
                else:
                    st.error(f"❌ Connection failed: {test_result['error']}")
    
    def _get_implementation_config(self, impl_type: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Get implementation configuration based on type"""
        if impl_type == "Python Function":
            return {
                "module": context.get("module_path", ""),
                "function": context.get("function_name", "")
            }
        elif impl_type == "External API":
            return {
                "endpoint": context.get("api_endpoint", ""),
                "method": context.get("api_method", "GET")
            }
        elif impl_type == "Shell Command":
            return {
                "command": context.get("shell_command", "")
            }
        else:  # Agent Chain
            return {
                "chain": context.get("agent_chain", "")
            }
    
    def _load_servers_config(self) -> Dict[str, Any]:
        """Load server configuration"""
        if self.servers_config_path.exists():
            with open(self.servers_config_path, 'r') as f:
                return json.load(f)
        return {}
    
    def _save_servers_config(self, config: Dict[str, Any]):
        """Save server configuration"""
        with open(self.servers_config_path, 'w') as f:
            json.dump(config, f, indent=2)
    
    def _load_tools_config(self) -> Dict[str, Any]:
        """Load tools configuration"""
        if self.tools_config_path.exists():
            with open(self.tools_config_path, 'r') as f:
                return json.load(f)
        return {}
    
    def _save_tools_config(self, config: Dict[str, Any]):
        """Save tools configuration"""
        with open(self.tools_config_path, 'w') as f:
            json.dump(config, f, indent=2)
    
    def _load_auth_config(self) -> Dict[str, Any]:
        """Load authentication configuration"""
        auth_path = self.config_dir / "auth.json"
        if auth_path.exists():
            with open(auth_path, 'r') as f:
                return json.load(f)
        return {"method": "API Key"}
    
    def _save_auth_config(self, config: Dict[str, Any]):
        """Save authentication configuration"""
        auth_path = self.config_dir / "auth.json"
        with open(auth_path, 'w') as f:
            json.dump(config, f, indent=2)
    
    def _generate_vscode_config(self, servers: Dict, tools: Dict, auth: Dict) -> Dict[str, Any]:
        """Generate VS Code compatible configuration"""
        return {
            "mcp.servers": {
                servers.get("name", "enhanced-agentic-workflow"): {
                    "host": servers.get("host", "localhost"),
                    "port": servers.get("port", 8765),
                    "ssl": servers.get("enable_ssl", False),
                    "auth": {
                        "type": auth.get("method", "API Key").lower().replace(" ", "_"),
                        "config": self._get_auth_config_for_ide(auth)
                    }
                }
            },
            "mcp.defaultServer": servers.get("name", "enhanced-agentic-workflow")
        }
    
    def _generate_cursor_config(self, servers: Dict, tools: Dict, auth: Dict) -> Dict[str, Any]:
        """Generate Cursor IDE compatible configuration"""
        return {
            "ai.mcp.enabled": True,
            "ai.mcp.server": {
                "url": f"{'https' if servers.get('enable_ssl') else 'http'}://{servers.get('host', 'localhost')}:{servers.get('port', 8765)}",
                "auth": self._get_auth_config_for_ide(auth)
            },
            "ai.mcp.tools": {
                name: {
                    "enabled": tool.get("enabled", True),
                    "category": tool.get("category", "Custom")
                }
                for name, tool in tools.items()
            }
        }
    
    def _get_auth_config_for_ide(self, auth: Dict) -> Dict[str, Any]:
        """Get authentication config for IDE"""
        method = auth.get("method", "None")
        
        if method == "API Key":
            return {
                "header": auth.get("api_key_header", "X-API-Key"),
                "value": "${MCP_API_KEY}"  # Environment variable reference
            }
        elif method == "OAuth 2.0":
            return {
                "clientId": auth.get("oauth_client_id", ""),
                "scopes": auth.get("oauth_scopes", [])
            }
        elif method == "JWT":
            return {
                "algorithm": auth.get("jwt_algorithm", "HS256")
            }
        else:
            return {}
    
    def _generate_docker_compose(self, servers: Dict, tools: Dict, auth: Dict) -> str:
        """Generate Docker Compose configuration"""
        return f"""version: '3.8'

services:
  mcp-server:
    build: .
    container_name: {servers.get('name', 'enhanced-agentic-workflow')}
    ports:
      - "{servers.get('port', 8765)}:{servers.get('port', 8765)}"
    environment:
      - MCP_HOST={servers.get('host', '0.0.0.0')}
      - MCP_PORT={servers.get('port', 8765)}
      - MCP_AUTH_METHOD={auth.get('method', 'API Key')}
      - MCP_ENABLE_SSL={str(servers.get('enable_ssl', False)).lower()}
      - MCP_MAX_CONNECTIONS={servers.get('max_connections', 10)}
      - MCP_ENABLE_LOGGING={str(servers.get('enable_logging', True)).lower()}
    volumes:
      - ./config/mcp:/app/config/mcp
      - ./src:/app/src
    restart: unless-stopped
    networks:
      - mcp-network

networks:
  mcp-network:
    driver: bridge
"""
    
    def _test_mcp_connection(self) -> Dict[str, Any]:
        """Test MCP server connection"""
        import requests
        
        servers = self._load_servers_config()
        
        try:
            protocol = "https" if servers.get("enable_ssl") else "http"
            url = f"{protocol}://{servers.get('host', 'localhost')}:{servers.get('port', 8765)}/health"
            
            response = requests.get(url, timeout=5)
            
            if response.status_code == 200:
                return {
                    "success": True,
                    "details": response.json() if response.headers.get('content-type') == 'application/json' else {
                        "status": "healthy",
                        "message": response.text
                    }
                }
            else:
                return {
                    "success": False,
                    "error": f"Server returned status code: {response.status_code}"
                }
                
        except requests.exceptions.ConnectionError:
            return {
                "success": False,
                "error": "Cannot connect to MCP server. Make sure it's running."
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }