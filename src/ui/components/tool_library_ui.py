"""Tool Library Browser and Manager UI Component"""

import streamlit as st
import yaml
import json
import os
from pathlib import Path
from typing import Dict, List, Any, Optional
import inspect
import ast
import tempfile
import subprocess
import shutil


class ToolLibraryUI:
    """UI component for tool library management"""
    
    def __init__(self):
        self.tools_dir = Path("tools")
        self.tool_templates_dir = Path("templates/tools")
        self.tool_categories = {
            "File Operations": {
                "icon": "📁",
                "description": "Read, write, and manipulate files"
            },
            "Web & API": {
                "icon": "🌐",
                "description": "Web scraping, API calls, and HTTP requests"
            },
            "Data Processing": {
                "icon": "📊",
                "description": "Data transformation and analysis"
            },
            "Development": {
                "icon": "💻",
                "description": "Code generation, testing, and deployment"
            },
            "Communication": {
                "icon": "💬",
                "description": "Email, messaging, and notifications"
            },
            "AI & ML": {
                "icon": "🤖",
                "description": "Machine learning and AI integrations"
            },
            "System": {
                "icon": "⚙️",
                "description": "System operations and monitoring"
            },
            "Custom": {
                "icon": "🔧",
                "description": "User-created custom tools"
            }
        }
    
    def render(self):
        """Render the tool library interface"""
        st.title("🛠️ Tool Library")
        st.markdown("Browse, create, and manage tools for your agents")
        
        # Create tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "Browse Tools",
            "Create Tool",
            "Tool Templates",
            "Tool Testing",
            "Import/Export"
        ])
        
        with tab1:
            self._render_browse_tools()
        
        with tab2:
            self._render_create_tool()
        
        with tab3:
            self._render_tool_templates()
        
        with tab4:
            self._render_tool_testing()
        
        with tab5:
            self._render_import_export()
    
    def _render_browse_tools(self):
        """Render tool browsing interface"""
        st.subheader("Tool Library Browser")
        
        # Search and filter
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            search = st.text_input("Search tools...", placeholder="e.g., file, api, data")
        
        with col2:
            category_filter = st.selectbox(
                "Category",
                ["All"] + list(self.tool_categories.keys())
            )
        
        with col3:
            sort_by = st.selectbox("Sort by", ["Name", "Category", "Last Modified"])
        
        # Load and display tools
        tools = self._load_tools()
        
        # Filter tools
        if search:
            tools = [t for t in tools if search.lower() in t['name'].lower() or 
                    search.lower() in t.get('description', '').lower()]
        
        if category_filter != "All":
            tools = [t for t in tools if t.get('category') == category_filter]
        
        # Sort tools
        if sort_by == "Name":
            tools.sort(key=lambda x: x['name'])
        elif sort_by == "Category":
            tools.sort(key=lambda x: x.get('category', ''))
        
        # Display tools
        if tools:
            for tool in tools:
                with st.expander(f"{tool['name']} - {tool.get('category', 'Uncategorized')}"):
                    self._display_tool_details(tool)
        else:
            st.info("No tools found matching your criteria")
        
        # Statistics
        st.markdown("---")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Tools", len(self._load_tools()))
        
        with col2:
            active_tools = len([t for t in tools if t.get('active', True)])
            st.metric("Active Tools", active_tools)
        
        with col3:
            categories = len(set(t.get('category', 'Uncategorized') for t in tools))
            st.metric("Categories", categories)
        
        with col4:
            custom_tools = len([t for t in tools if t.get('category') == 'Custom'])
            st.metric("Custom Tools", custom_tools)
    
    def _render_create_tool(self):
        """Render tool creation interface"""
        st.subheader("Create New Tool")
        
        # Tool creation method
        creation_method = st.radio(
            "Creation Method",
            ["Visual Builder", "Code Editor", "From Template", "AI Generator"]
        )
        
        st.markdown("---")
        
        if creation_method == "Visual Builder":
            self._render_visual_builder()
        elif creation_method == "Code Editor":
            self._render_code_editor()
        elif creation_method == "From Template":
            self._render_template_selection()
        elif creation_method == "AI Generator":
            self._render_ai_generator()
    
    def _render_visual_builder(self):
        """Render visual tool builder"""
        st.markdown("### Visual Tool Builder")
        
        # Basic information
        col1, col2 = st.columns(2)
        
        with col1:
            tool_name = st.text_input(
                "Tool Name",
                placeholder="my_custom_tool",
                help="Lowercase, underscore-separated"
            )
            
            category = st.selectbox(
                "Category",
                list(self.tool_categories.keys())
            )
        
        with col2:
            display_name = st.text_input(
                "Display Name",
                placeholder="My Custom Tool"
            )
            
            icon = st.text_input(
                "Icon (emoji)",
                placeholder="🔧",
                max_chars=2
            )
        
        # Description
        description = st.text_area(
            "Description",
            placeholder="Describe what this tool does...",
            height=100
        )
        
        # Parameters
        st.markdown("### Tool Parameters")
        
        if 'tool_params' not in st.session_state:
            st.session_state.tool_params = []
        
        # Add parameter interface
        with st.expander("Add Parameter"):
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                param_name = st.text_input("Parameter Name")
            
            with col2:
                param_type = st.selectbox(
                    "Type",
                    ["string", "integer", "float", "boolean", "list", "dict"]
                )
            
            with col3:
                param_required = st.checkbox("Required")
            
            with col4:
                if st.button("Add Parameter"):
                    if param_name:
                        st.session_state.tool_params.append({
                            "name": param_name,
                            "type": param_type,
                            "required": param_required
                        })
        
        # Display parameters
        if st.session_state.tool_params:
            st.markdown("#### Parameters")
            for i, param in enumerate(st.session_state.tool_params):
                col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
                
                with col1:
                    st.text(param['name'])
                
                with col2:
                    st.text(param['type'])
                
                with col3:
                    st.text("Required" if param['required'] else "Optional")
                
                with col4:
                    if st.button("🗑️", key=f"del_param_{i}"):
                        st.session_state.tool_params.pop(i)
                        st.rerun()
        
        # Implementation
        st.markdown("### Tool Implementation")
        
        implementation_type = st.selectbox(
            "Implementation Type",
            ["Python Function", "API Call", "Shell Command", "Composite"]
        )
        
        if implementation_type == "Python Function":
            code = st.text_area(
                "Python Code",
                value="""async def execute(self, **kwargs):
    # Your tool implementation here
    result = "Tool executed successfully"
    return result""",
                height=200
            )
        
        elif implementation_type == "API Call":
            col1, col2 = st.columns(2)
            
            with col1:
                api_url = st.text_input("API URL")
                api_method = st.selectbox("Method", ["GET", "POST", "PUT", "DELETE"])
            
            with col2:
                api_headers = st.text_area("Headers (JSON)", value="{}")
                api_auth = st.selectbox("Authentication", ["None", "API Key", "Bearer Token", "Basic Auth"])
        
        elif implementation_type == "Shell Command":
            command = st.text_input("Shell Command")
            working_dir = st.text_input("Working Directory", value=".")
            capture_output = st.checkbox("Capture Output", value=True)
        
        # Dependencies
        st.markdown("### Dependencies")
        dependencies = st.text_area(
            "Required Libraries (one per line)",
            placeholder="requests\npandas\nbeautifulsoup4",
            height=100
        )
        
        # Create tool button
        if st.button("🚀 Create Tool", type="primary"):
            if tool_name and description:
                tool_config = {
                    "name": tool_name,
                    "display_name": display_name,
                    "category": category,
                    "icon": icon,
                    "description": description,
                    "parameters": st.session_state.tool_params,
                    "implementation_type": implementation_type,
                    "dependencies": dependencies.split('\n') if dependencies else []
                }
                
                self._create_tool(tool_config)
                st.session_state.tool_params = []
            else:
                st.error("Please provide tool name and description")
    
    def _render_code_editor(self):
        """Render code editor for tool creation"""
        st.markdown("### Code Editor")
        
        # Template selection
        template = st.selectbox(
            "Start from template",
            ["Blank", "Basic Tool", "API Tool", "File Tool", "Data Tool"]
        )
        
        # Load template code
        template_code = self._get_tool_template(template)
        
        # Code editor
        code = st.text_area(
            "Tool Code",
            value=template_code,
            height=400,
            help="Write your tool implementation"
        )
        
        # Validation
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🔍 Validate Code"):
                validation_result = self._validate_tool_code(code)
                if validation_result['valid']:
                    st.success("✅ Code is valid!")
                else:
                    st.error(f"❌ Validation failed: {validation_result['error']}")
        
        with col2:
            if st.button("💾 Save Tool"):
                tool_name = st.text_input("Tool filename (without .py)")
                if tool_name:
                    self._save_tool_code(tool_name, code)
    
    def _render_ai_generator(self):
        """Render AI-powered tool generator"""
        st.markdown("### AI Tool Generator")
        st.info("Describe what you want the tool to do, and AI will generate it for you!")
        
        # Tool requirements
        tool_purpose = st.text_area(
            "What should this tool do?",
            placeholder="I need a tool that can extract data from PDF files and convert it to structured JSON format...",
            height=150
        )
        
        # Additional specifications
        with st.expander("Additional Specifications"):
            col1, col2 = st.columns(2)
            
            with col1:
                include_error_handling = st.checkbox("Include error handling", value=True)
                include_logging = st.checkbox("Include logging", value=True)
                include_validation = st.checkbox("Include input validation", value=True)
            
            with col2:
                async_implementation = st.checkbox("Async implementation", value=True)
                include_tests = st.checkbox("Generate tests", value=False)
                include_documentation = st.checkbox("Generate documentation", value=True)
        
        # Generate button
        if st.button("🤖 Generate Tool", type="primary"):
            if tool_purpose:
                with st.spinner("AI is generating your tool..."):
                    # Simulate AI generation
                    generated_code = self._generate_tool_with_ai(tool_purpose)
                    
                    st.markdown("### Generated Tool")
                    st.code(generated_code, language="python")
                    
                    # Save options
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if st.button("💾 Save Generated Tool"):
                            tool_name = st.text_input("Tool name")
                            if tool_name:
                                self._save_tool_code(tool_name, generated_code)
                    
                    with col2:
                        if st.button("🔄 Regenerate"):
                            st.rerun()
            else:
                st.error("Please describe what the tool should do")
    
    def _render_tool_templates(self):
        """Render tool templates"""
        st.subheader("Tool Templates")
        st.markdown("Start with pre-built templates for common tool patterns")
        
        # Template categories
        template_categories = {
            "API Integration": [
                {"name": "REST API Client", "description": "Generic REST API integration"},
                {"name": "GraphQL Client", "description": "GraphQL API queries"},
                {"name": "Webhook Handler", "description": "Handle incoming webhooks"}
            ],
            "Data Processing": [
                {"name": "CSV Processor", "description": "Read and process CSV files"},
                {"name": "JSON Transformer", "description": "Transform JSON data"},
                {"name": "Data Validator", "description": "Validate data against schemas"}
            ],
            "File Operations": [
                {"name": "File Search", "description": "Search files by pattern"},
                {"name": "File Archiver", "description": "Create and extract archives"},
                {"name": "File Converter", "description": "Convert between file formats"}
            ],
            "Web Scraping": [
                {"name": "HTML Parser", "description": "Extract data from HTML"},
                {"name": "Dynamic Scraper", "description": "Scrape JavaScript sites"},
                {"name": "API Scraper", "description": "Extract data from APIs"}
            ],
            "Development": [
                {"name": "Code Formatter", "description": "Format code files"},
                {"name": "Test Runner", "description": "Execute test suites"},
                {"name": "Dependency Checker", "description": "Check project dependencies"}
            ]
        }
        
        # Display templates by category
        for category, templates in template_categories.items():
            st.markdown(f"### {category}")
            
            cols = st.columns(3)
            for i, template in enumerate(templates):
                with cols[i % 3]:
                    with st.container():
                        st.markdown(f"**{template['name']}**")
                        st.text(template['description'])
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            if st.button("Use", key=f"use_{template['name']}"):
                                self._load_tool_template(template['name'])
                        
                        with col2:
                            if st.button("Preview", key=f"preview_{template['name']}"):
                                self._preview_tool_template(template['name'])
    
    def _render_tool_testing(self):
        """Render tool testing interface"""
        st.subheader("Tool Testing Playground")
        
        # Tool selection
        available_tools = self._get_available_tool_names()
        
        if not available_tools:
            st.warning("No tools available for testing")
            return
        
        selected_tool = st.selectbox("Select Tool to Test", available_tools)
        
        if selected_tool:
            tool_info = self._load_tool_info(selected_tool)
            
            # Display tool information
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown(f"### {tool_info.get('display_name', selected_tool)}")
                st.text(tool_info.get('description', 'No description available'))
            
            with col2:
                st.metric("Category", tool_info.get('category', 'Uncategorized'))
                st.metric("Version", tool_info.get('version', '1.0.0'))
            
            st.markdown("---")
            
            # Parameter inputs
            st.markdown("### Parameters")
            
            params = tool_info.get('parameters', [])
            param_values = {}
            
            if params:
                for param in params:
                    param_name = param['name']
                    param_type = param['type']
                    required = param.get('required', False)
                    
                    label = f"{param_name} ({param_type})"
                    if required:
                        label += " *"
                    
                    if param_type == "string":
                        param_values[param_name] = st.text_input(label)
                    elif param_type == "integer":
                        param_values[param_name] = st.number_input(label, value=0)
                    elif param_type == "float":
                        param_values[param_name] = st.number_input(label, value=0.0)
                    elif param_type == "boolean":
                        param_values[param_name] = st.checkbox(label)
                    elif param_type == "list":
                        param_values[param_name] = st.text_area(label, help="One item per line").split('\n')
                    elif param_type == "dict":
                        param_values[param_name] = st.text_area(label, help="JSON format")
            else:
                st.info("This tool has no parameters")
            
            # Test execution
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("🚀 Run Test", type="primary"):
                    with st.spinner("Running tool..."):
                        result = self._run_tool_test(selected_tool, param_values)
                        
                        st.markdown("### Test Results")
                        
                        if result['success']:
                            st.success("✅ Tool executed successfully!")
                            
                            # Display output
                            st.markdown("#### Output")
                            if isinstance(result['output'], dict):
                                st.json(result['output'])
                            else:
                                st.code(str(result['output']))
                            
                            # Display metrics
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                st.metric("Execution Time", f"{result['execution_time']:.2f}s")
                            
                            with col2:
                                st.metric("Memory Used", f"{result['memory_used']:.1f} MB")
                            
                            with col3:
                                st.metric("Status", "Success")
                        else:
                            st.error("❌ Tool execution failed!")
                            st.code(result['error'])
            
            with col2:
                if st.button("📊 Run Benchmark"):
                    self._run_tool_benchmark(selected_tool)
            
            # Test history
            with st.expander("Test History"):
                st.info("Previous test runs would be displayed here")
    
    def _render_import_export(self):
        """Render import/export functionality"""
        st.subheader("Import/Export Tools")
        
        tab1, tab2 = st.tabs(["Export Tools", "Import Tools"])
        
        with tab1:
            st.markdown("### Export Tools")
            
            # Select tools to export
            available_tools = self._get_available_tool_names()
            selected_tools = st.multiselect(
                "Select tools to export",
                available_tools,
                default=available_tools
            )
            
            # Export format
            export_format = st.selectbox(
                "Export Format",
                ["Tool Package (.zip)", "Individual Files", "JSON Configuration"]
            )
            
            # Export options
            include_dependencies = st.checkbox("Include dependencies", value=True)
            include_tests = st.checkbox("Include tests", value=False)
            include_docs = st.checkbox("Include documentation", value=True)
            
            if st.button("📤 Export Tools"):
                if selected_tools:
                    export_data = self._export_tools(
                        selected_tools,
                        export_format,
                        include_dependencies,
                        include_tests,
                        include_docs
                    )
                    
                    if export_format == "Tool Package (.zip)":
                        st.download_button(
                            "Download Tool Package",
                            export_data,
                            "tools_export.zip",
                            "application/zip"
                        )
                    else:
                        st.success("Tools exported successfully!")
                else:
                    st.error("Please select tools to export")
        
        with tab2:
            st.markdown("### Import Tools")
            
            # Import method
            import_method = st.radio(
                "Import Method",
                ["Upload Package", "From URL", "From Git Repository"]
            )
            
            if import_method == "Upload Package":
                uploaded_file = st.file_uploader(
                    "Upload tool package",
                    type=['zip', 'py', 'json']
                )
                
                if uploaded_file:
                    if st.button("📥 Import Tools"):
                        self._import_tools_from_file(uploaded_file)
            
            elif import_method == "From URL":
                tool_url = st.text_input("Tool URL")
                
                if st.button("📥 Import from URL"):
                    if tool_url:
                        self._import_tools_from_url(tool_url)
                    else:
                        st.error("Please provide a URL")
            
            elif import_method == "From Git Repository":
                repo_url = st.text_input("Repository URL")
                branch = st.text_input("Branch", value="main")
                path = st.text_input("Tools path", value="tools/")
                
                if st.button("📥 Import from Git"):
                    if repo_url:
                        self._import_tools_from_git(repo_url, branch, path)
                    else:
                        st.error("Please provide a repository URL")
    
    def _display_tool_details(self, tool: Dict[str, Any]):
        """Display detailed tool information"""
        # Basic info
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.markdown(f"**Description:** {tool.get('description', 'No description')}")
            st.markdown(f"**Category:** {tool.get('category', 'Uncategorized')}")
            st.markdown(f"**Version:** {tool.get('version', '1.0.0')}")
        
        with col2:
            if st.button("Edit", key=f"edit_{tool['name']}"):
                st.session_state['editing_tool'] = tool['name']
                st.rerun()
            
            if st.button("Delete", key=f"delete_{tool['name']}"):
                if st.confirm(f"Delete tool '{tool['name']}'?"):
                    self._delete_tool(tool['name'])
                    st.rerun()
        
        # Parameters
        if tool.get('parameters'):
            st.markdown("**Parameters:**")
            for param in tool['parameters']:
                st.text(f"• {param['name']} ({param['type']}) - {'Required' if param.get('required') else 'Optional'}")
        
        # Dependencies
        if tool.get('dependencies'):
            st.markdown("**Dependencies:**")
            st.text(", ".join(tool['dependencies']))
        
        # Usage example
        if tool.get('example'):
            st.markdown("**Usage Example:**")
            st.code(tool['example'], language="python")
    
    def _load_tools(self) -> List[Dict[str, Any]]:
        """Load all available tools"""
        tools = []
        
        if self.tools_dir.exists():
            for tool_file in self.tools_dir.glob("*_tool.py"):
                if tool_file.stem != "base_tool":
                    tool_info = self._extract_tool_info(tool_file)
                    if tool_info:
                        tools.append(tool_info)
        
        return tools
    
    def _extract_tool_info(self, tool_file: Path) -> Optional[Dict[str, Any]]:
        """Extract tool information from file"""
        try:
            with open(tool_file, 'r') as f:
                content = f.read()
            
            # Parse the file to extract info
            tree = ast.parse(content)
            
            tool_info = {
                "name": tool_file.stem,
                "file": str(tool_file),
                "category": "Custom"
            }
            
            # Extract docstring and other metadata
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    if node.name.endswith("Tool"):
                        # Get docstring
                        docstring = ast.get_docstring(node)
                        if docstring:
                            tool_info["description"] = docstring.split('\n')[0]
                        
                        # Look for metadata
                        for item in node.body:
                            if isinstance(item, ast.Assign):
                                for target in item.targets:
                                    if isinstance(target, ast.Name):
                                        if target.id == "category":
                                            if isinstance(item.value, ast.Str):
                                                tool_info["category"] = item.value.s
            
            return tool_info
            
        except Exception as e:
            st.error(f"Error loading tool {tool_file}: {str(e)}")
            return None
    
    def _get_tool_template(self, template_name: str) -> str:
        """Get tool template code"""
        templates = {
            "Blank": """from tools.base_tool import BaseTool
from typing import Dict, Any


class MyTool(BaseTool):
    \"\"\"Your tool description here\"\"\"
    
    name = "my_tool"
    description = "What this tool does"
    
    async def execute(self, **kwargs) -> Dict[str, Any]:
        \"\"\"Execute the tool\"\"\"
        # Your implementation here
        return {"result": "success"}
""",
            "Basic Tool": """from tools.base_tool import BaseTool
from typing import Dict, Any


class BasicTool(BaseTool):
    \"\"\"A basic tool template\"\"\"
    
    name = "basic_tool"
    description = "A basic tool that demonstrates the structure"
    category = "Custom"
    version = "1.0.0"
    
    async def execute(self, input_text: str, **kwargs) -> Dict[str, Any]:
        \"\"\"
        Execute the tool
        
        Args:
            input_text: The input text to process
            
        Returns:
            Dict containing the result
        \"\"\"
        try:
            # Process the input
            result = f"Processed: {input_text}"
            
            return {
                "success": True,
                "result": result
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
""",
            "API Tool": """from tools.base_tool import BaseTool
from typing import Dict, Any
import aiohttp
import json


class APITool(BaseTool):
    \"\"\"Tool for making API requests\"\"\"
    
    name = "api_tool"
    description = "Make HTTP API requests"
    category = "Web & API"
    version = "1.0.0"
    
    async def execute(
        self, 
        url: str, 
        method: str = "GET", 
        headers: Dict[str, str] = None,
        data: Dict[str, Any] = None,
        **kwargs
    ) -> Dict[str, Any]:
        \"\"\"
        Make an API request
        
        Args:
            url: The API endpoint URL
            method: HTTP method (GET, POST, PUT, DELETE)
            headers: Request headers
            data: Request data (for POST/PUT)
            
        Returns:
            API response
        \"\"\"
        try:
            async with aiohttp.ClientSession() as session:
                async with session.request(
                    method=method,
                    url=url,
                    headers=headers,
                    json=data
                ) as response:
                    result = await response.json()
                    
                    return {
                        "success": True,
                        "status_code": response.status,
                        "data": result
                    }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
"""
        }
        
        return templates.get(template_name, templates["Blank"])
    
    def _create_tool(self, config: Dict[str, Any]):
        """Create a new tool"""
        try:
            tool_file = self.tools_dir / f"{config['name']}_tool.py"
            
            # Generate tool code
            tool_code = self._generate_tool_code(config)
            
            # Save tool file
            with open(tool_file, 'w') as f:
                f.write(tool_code)
            
            st.success(f"✅ Tool '{config['name']}' created successfully!")
            st.info(f"Location: {tool_file}")
            
        except Exception as e:
            st.error(f"Error creating tool: {str(e)}")
    
    def _generate_tool_code(self, config: Dict[str, Any]) -> str:
        """Generate tool code from configuration"""
        # Generate parameter string
        params = []
        for param in config.get('parameters', []):
            param_str = f"{param['name']}: {param['type']}"
            if not param.get('required'):
                param_str += " = None"
            params.append(param_str)
        
        params_str = ", ".join(params) if params else ""
        
        code = f'''"""
{config['display_name']} - {config['description']}
Generated by Tool Builder
"""

from tools.base_tool import BaseTool
from typing import Dict, Any
{self._generate_imports(config)}


class {config['name'].title().replace('_', '')}Tool(BaseTool):
    """{config['description']}"""
    
    name = "{config['name']}"
    description = "{config['description']}"
    category = "{config['category']}"
    version = "1.0.0"
    
    async def execute(self, {params_str}{"," if params_str else ""} **kwargs) -> Dict[str, Any]:
        """
        Execute the tool
        
        Args:
{self._generate_docstring_params(config['parameters'])}
            
        Returns:
            Dict containing the result
        """
        try:
{self._generate_implementation(config)}
            
            return {{
                "success": True,
                "result": result
            }}
        except Exception as e:
            return {{
                "success": False,
                "error": str(e)
            }}
'''
        
        return code
    
    def _generate_imports(self, config: Dict[str, Any]) -> str:
        """Generate import statements based on tool configuration"""
        imports = []
        
        if config.get('implementation_type') == 'API Call':
            imports.append("import aiohttp")
            imports.append("import json")
        elif config.get('implementation_type') == 'Shell Command':
            imports.append("import subprocess")
            imports.append("import asyncio")
        
        # Add dependency imports
        for dep in config.get('dependencies', []):
            imports.append(f"import {dep}")
        
        return "\n".join(imports)
    
    def _generate_docstring_params(self, parameters: List[Dict[str, Any]]) -> str:
        """Generate docstring parameters"""
        if not parameters:
            return "            No parameters"
        
        lines = []
        for param in parameters:
            lines.append(f"            {param['name']}: Description of {param['name']}")
        
        return "\n".join(lines)
    
    def _generate_implementation(self, config: Dict[str, Any]) -> str:
        """Generate tool implementation based on type"""
        impl_type = config.get('implementation_type', 'Python Function')
        
        if impl_type == 'Python Function':
            return """            # Your implementation here
            result = "Tool executed successfully\""""
        
        elif impl_type == 'API Call':
            return """            # Make API request
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    result = await response.json()"""
        
        elif impl_type == 'Shell Command':
            return """            # Execute shell command
            proc = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await proc.communicate()
            result = stdout.decode()"""
        
        else:
            return """            # Composite implementation
            result = "Implement your logic here\""""
    
    def _get_available_tool_names(self) -> List[str]:
        """Get list of available tool names"""
        tools = []
        
        if self.tools_dir.exists():
            for tool_file in self.tools_dir.glob("*_tool.py"):
                if tool_file.stem != "base_tool":
                    tools.append(tool_file.stem)
        
        return tools
    
    def _load_tool_info(self, tool_name: str) -> Dict[str, Any]:
        """Load information about a specific tool"""
        tool_file = self.tools_dir / f"{tool_name}.py"
        
        if tool_file.exists():
            return self._extract_tool_info(tool_file) or {}
        
        return {}
    
    def _run_tool_test(self, tool_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Run a test of the selected tool"""
        # Simulate tool execution
        import time
        import random
        
        start_time = time.time()
        
        # Simulate processing
        time.sleep(random.uniform(0.5, 2.0))
        
        execution_time = time.time() - start_time
        
        return {
            "success": random.choice([True, True, True, False]),  # 75% success rate
            "output": {
                "message": f"Tool {tool_name} executed successfully",
                "params_received": params,
                "timestamp": time.time()
            },
            "execution_time": execution_time,
            "memory_used": random.uniform(10, 100),
            "error": "Simulated error" if random.random() < 0.25 else None
        }
    
    def _generate_tool_with_ai(self, purpose: str) -> str:
        """Generate tool code using AI"""
        # This would call an AI model to generate the tool
        # For now, return a template
        return f'''"""
AI-Generated Tool
Purpose: {purpose}
"""

from tools.base_tool import BaseTool
from typing import Dict, Any


class GeneratedTool(BaseTool):
    """{purpose}"""
    
    name = "generated_tool"
    description = "{purpose}"
    category = "Custom"
    version = "1.0.0"
    
    async def execute(self, input_data: Any, **kwargs) -> Dict[str, Any]:
        """
        Execute the AI-generated tool
        
        Args:
            input_data: The input data to process
            
        Returns:
            Dict containing the result
        """
        try:
            # AI-generated implementation would go here
            # This is a placeholder
            result = f"Processed: {{input_data}}"
            
            return {{
                "success": True,
                "result": result
            }}
        except Exception as e:
            return {{
                "success": False,
                "error": str(e)
            }}
'''