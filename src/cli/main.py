"""Command-line interface for Enhanced Agentic Workflow"""

import click
import asyncio
from typing import Optional, List
import json
import yaml
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

# Import core components
from ..core.orchestrator import AgentOrchestrator
from ..core.workflow_state import WorkflowState
from ..library.tool_library import ToolLibrary
from ..library.agent_templates import AgentTemplateLibrary
from ..progress.progress_tracker import ProgressTracker
from ..progress.streaming import StreamingProgress, ConsoleStreamHandler
from ..knowledge.knowledge_base import KnowledgeBase
from ..knowledge.vector_store import VectorStore

console = Console()


@click.group()
@click.version_option(version="1.0.0", prog_name="Enhanced Agentic Workflow")
def cli():
    """Enhanced Agentic Workflow - Production-ready AI agent orchestration"""
    pass


@cli.command()
@click.argument('query')
@click.option('--mode', '-m', type=click.Choice(['simple', 'complex', 'grok']), default='auto',
              help='Execution mode: simple (single agent), complex (multi-agent), grok (deep analysis), auto (automatic)')
@click.option('--output', '-o', type=click.Choice(['text', 'json', 'markdown']), default='text',
              help='Output format')
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose output')
@click.option('--config', '-c', type=click.Path(exists=True), help='Configuration file path')
@click.option('--no-progress', is_flag=True, help='Disable progress display')
async def run(query: str, mode: str, output: str, verbose: bool, config: Optional[str], no_progress: bool):
    """Execute a query using the agentic workflow"""
    
    # Display banner
    if not no_progress:
        banner = Panel(
            f"[bold cyan]Enhanced Agentic Workflow[/bold cyan]\n"
            f"[dim]Query: {query}[/dim]\n"
            f"[dim]Mode: {mode}[/dim]",
            border_style="cyan",
            padding=(1, 2)
        )
        console.print(banner)
        console.print()
    
    # Load configuration
    config_data = {}
    if config:
        config_path = Path(config)
        if config_path.suffix == '.yaml':
            with open(config_path, 'r') as f:
                config_data = yaml.safe_load(f)
        else:
            with open(config_path, 'r') as f:
                config_data = json.load(f)
    
    # Initialize components
    orchestrator = AgentOrchestrator(config=config_data)
    progress_tracker = ProgressTracker() if not no_progress else None
    
    # Set up streaming if enabled
    streaming = None
    if progress_tracker and not no_progress:
        streaming = StreamingProgress(progress_tracker)
        console_handler = ConsoleStreamHandler(verbose=verbose)
        streaming.add_handler(console_handler)
        await streaming.start_streaming()
    
    try:
        # Execute query
        if mode == 'auto':
            # Let the system decide based on complexity
            mode = await orchestrator.determine_mode(query)
            if verbose:
                console.print(f"[dim]Auto-detected mode: {mode}[/dim]")
        
        # Create main task
        if progress_tracker:
            main_task = await progress_tracker.create_task(
                task_id="main",
                name=f"Processing query: {query[:50]}...",
                metadata={"mode": mode, "query": query}
            )
        
        # Execute based on mode
        if mode == 'simple':
            result = await orchestrator.execute_simple(query)
        elif mode == 'complex':
            result = await orchestrator.execute_complex(query)
        elif mode == 'grok':
            # Import grok mode
            from ..progress.grok_mode import GrokHeavyMode
            grok = GrokHeavyMode(
                agent=orchestrator.get_base_agent(),
                knowledge_base=orchestrator.knowledge_base,
                validation_gate=orchestrator.validation_gate,
                progress_tracker=progress_tracker
            )
            result = await grok.analyze(query)
        else:
            result = await orchestrator.execute(query)
        
        # Complete main task
        if progress_tracker:
            await progress_tracker.complete_task("main", "Query processing complete")
        
        # Format and display output
        if output == 'json':
            console.print_json(data=result if isinstance(result, dict) else {"result": str(result)})
        elif output == 'markdown':
            # Convert to markdown
            md_output = _format_as_markdown(result)
            console.print(md_output)
        else:
            # Text output
            if isinstance(result, dict):
                _display_result_dict(result)
            else:
                console.print(result)
        
    except Exception as e:
        if progress_tracker:
            await progress_tracker.fail_task("main", str(e))
        
        console.print(f"[red]Error: {str(e)}[/red]")
        if verbose:
            import traceback
            console.print(traceback.format_exc())
        
    finally:
        if streaming:
            await streaming.stop_streaming()


@cli.command()
@click.option('--category', '-c', help='Filter by category')
@click.option('--search', '-s', help='Search tools by name or description')
@click.option('--format', '-f', type=click.Choice(['table', 'json', 'list']), default='table',
              help='Output format')
def tools(category: Optional[str], search: Optional[str], format: str):
    """List available tools"""
    asyncio.run(_list_tools(category, search, format))


async def _list_tools(category: Optional[str], search: Optional[str], format: str):
    """List available tools implementation"""
    tool_library = ToolLibrary()
    
    if search:
        tools = await tool_library.search_tools(search)
    else:
        tools = await tool_library.list_tools(category=category)
    
    if format == 'json':
        console.print_json(data=[tool.dict() for tool in tools])
    elif format == 'list':
        for tool in tools:
            console.print(f"• {tool.name} - {tool.description}")
    else:  # table
        table = Table(title="Available Tools")
        table.add_column("Name", style="cyan")
        table.add_column("Category", style="magenta")
        table.add_column("Description", style="white")
        table.add_column("MCP", style="green")
        
        for tool in tools:
            table.add_row(
                tool.name,
                tool.category,
                tool.description[:50] + "..." if len(tool.description) > 50 else tool.description,
                "✓" if tool.mcp_compatible else "✗"
            )
        
        console.print(table)


@cli.command()
@click.option('--category', '-c', help='Filter by category')
@click.option('--format', '-f', type=click.Choice(['table', 'json', 'list']), default='table',
              help='Output format')
def templates(category: Optional[str], format: str):
    """List available agent templates"""
    asyncio.run(_list_templates(category, format))


async def _list_templates(category: Optional[str], format: str):
    """List available templates implementation"""
    template_library = AgentTemplateLibrary()
    templates = await template_library.list_templates(category=category)
    
    if format == 'json':
        console.print_json(data=[template.dict() for template in templates])
    elif format == 'list':
        for template in templates:
            console.print(f"• {template.name} ({template.id}) - {template.description}")
    else:  # table
        table = Table(title="Agent Templates")
        table.add_column("ID", style="cyan")
        table.add_column("Name", style="magenta")
        table.add_column("Category", style="yellow")
        table.add_column("Description", style="white")
        
        for template in templates:
            table.add_row(
                template.id,
                template.name,
                template.category,
                template.description[:50] + "..." if len(template.description) > 50 else template.description
            )
        
        console.print(table)


@cli.command()
@click.argument('agent_type')
@click.argument('requirements')
@click.option('--template', '-t', help='Template ID to use')
@click.option('--tools', '-T', multiple=True, help='Tools to include (can be specified multiple times)')
@click.option('--output-dir', '-o', type=click.Path(), default='./generated',
              help='Output directory for generated agent')
@click.option('--no-validation', is_flag=True, help='Skip validation')
async def generate(agent_type: str, requirements: str, template: Optional[str], 
                  tools: List[str], output_dir: str, no_validation: bool):
    """Generate a new agent based on requirements"""
    
    console.print(f"[bold cyan]Generating {agent_type} agent...[/bold cyan]")
    
    # Initialize components
    orchestrator = AgentOrchestrator()
    progress_tracker = ProgressTracker()
    streaming = StreamingProgress(progress_tracker)
    console_handler = ConsoleStreamHandler(verbose=True)
    streaming.add_handler(console_handler)
    await streaming.start_streaming()
    
    try:
        # Create generation task
        task = await progress_tracker.create_task(
            task_id="generate",
            name=f"Generating {agent_type} agent",
            metadata={"type": agent_type, "requirements": requirements}
        )
        
        # Generate agent
        result = await orchestrator.generate_agent(
            agent_type=agent_type,
            requirements=requirements,
            template_id=template,
            tools=list(tools),
            validate=not no_validation
        )
        
        # Save generated files
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        for file_name, content in result.get('files', {}).items():
            file_path = output_path / file_name
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(file_path, 'w') as f:
                f.write(content)
            
            console.print(f"[green]✓[/green] Generated: {file_path}")
        
        # Complete task
        await progress_tracker.complete_task("generate", "Agent generation complete")
        
        # Display summary
        console.print()
        console.print(Panel(
            f"[bold green]Agent generated successfully![/bold green]\n\n"
            f"Type: {agent_type}\n"
            f"Files: {len(result.get('files', {}))}\n"
            f"Location: {output_path}",
            title="Generation Complete",
            border_style="green"
        ))
        
        # Display setup instructions if available
        if 'setup_instructions' in result:
            console.print("\n[bold yellow]Setup Instructions:[/bold yellow]")
            for instruction in result['setup_instructions']:
                console.print(f"  • {instruction}")
        
    except Exception as e:
        await progress_tracker.fail_task("generate", str(e))
        console.print(f"[red]Generation failed: {str(e)}[/red]")
    
    finally:
        await streaming.stop_streaming()


@cli.command()
@click.argument('file_path', type=click.Path(exists=True))
@click.option('--chunk-size', '-c', type=int, default=1000, help='Chunk size for splitting')
@click.option('--category', '-C', required=True, help='Knowledge category')
@click.option('--tags', '-t', multiple=True, help='Tags for the knowledge')
async def index(file_path: str, chunk_size: int, category: str, tags: List[str]):
    """Index a file into the knowledge base"""
    
    console.print(f"[bold cyan]Indexing {file_path}...[/bold cyan]")
    
    # Initialize knowledge base
    # Note: In production, these would come from config
    vector_store = VectorStore(
        supabase_url="YOUR_SUPABASE_URL",
        supabase_key="YOUR_SUPABASE_KEY",
        openai_api_key="YOUR_OPENAI_KEY"
    )
    knowledge_base = KnowledgeBase(vector_store)
    
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Indexing file...", total=None)
            
            # Index the file
            doc_ids = await knowledge_base.add_file_knowledge(
                file_path=Path(file_path),
                category=category,
                tags=list(tags),
                chunk_size=chunk_size
            )
            
            progress.update(task, completed=True)
        
        console.print(f"[green]✓[/green] Indexed {len(doc_ids)} chunks successfully!")
        
    except Exception as e:
        console.print(f"[red]Indexing failed: {str(e)}[/red]")


@cli.command()
@click.option('--host', '-h', default='localhost', help='Server host')
@click.option('--port', '-p', type=int, default=8000, help='Server port')
@click.option('--reload', is_flag=True, help='Enable auto-reload')
def serve(host: str, port: int, reload: bool):
    """Start the FastAPI service"""
    import uvicorn
    
    console.print(f"[bold cyan]Starting Enhanced Agentic Workflow API...[/bold cyan]")
    console.print(f"Host: {host}")
    console.print(f"Port: {port}")
    console.print(f"Reload: {'enabled' if reload else 'disabled'}")
    console.print()
    
    uvicorn.run(
        "src.api.main:app",
        host=host,
        port=port,
        reload=reload
    )


@cli.command()
def ui():
    """Launch the Streamlit UI"""
    import subprocess
    import sys
    
    console.print("[bold cyan]Launching Streamlit UI...[/bold cyan]")
    
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run",
            "src/ui/streamlit_dashboard.py"
        ])
    except KeyboardInterrupt:
        console.print("\n[yellow]UI closed[/yellow]")


@cli.command()
@click.option('--format', '-f', type=click.Choice(['yaml', 'json', 'env']), default='yaml',
              help='Configuration format')
@click.option('--output', '-o', type=click.Path(), help='Output file path')
def init(format: str, output: Optional[str]):
    """Initialize configuration file"""
    
    config = {
        "workflow": {
            "max_parallel_agents": 4,
            "max_refinement_cycles": 3,
            "validation_timeout": 300,
            "enable_self_correction": True
        },
        "agents": {
            "model_provider": "openai:gpt-4",
            "temperature": 0.7,
            "max_tokens": 2000
        },
        "knowledge": {
            "supabase_url": "${SUPABASE_URL}",
            "supabase_key": "${SUPABASE_KEY}",
            "openai_api_key": "${OPENAI_API_KEY}"
        },
        "validation": {
            "enable_pytest": True,
            "enable_ruff": True,
            "enable_mypy": True
        },
        "mcp": {
            "enabled": True,
            "port": 8765,
            "auth_method": "api_key"
        }
    }
    
    # Format configuration
    if format == 'yaml':
        import yaml
        content = yaml.dump(config, default_flow_style=False)
        default_file = 'config.yaml'
    elif format == 'json':
        content = json.dumps(config, indent=2)
        default_file = 'config.json'
    else:  # env
        lines = []
        lines.append("# Enhanced Agentic Workflow Configuration")
        lines.append("WORKFLOW_MAX_PARALLEL_AGENTS=4")
        lines.append("WORKFLOW_MAX_REFINEMENT_CYCLES=3")
        lines.append("WORKFLOW_VALIDATION_TIMEOUT=300")
        lines.append("AGENTS_MODEL_PROVIDER=openai:gpt-4")
        lines.append("SUPABASE_URL=your_supabase_url")
        lines.append("SUPABASE_KEY=your_supabase_key")
        lines.append("OPENAI_API_KEY=your_openai_key")
        content = "\n".join(lines)
        default_file = '.env'
    
    # Write configuration
    output_path = output or default_file
    with open(output_path, 'w') as f:
        f.write(content)
    
    console.print(f"[green]✓[/green] Configuration initialized: {output_path}")
    console.print("\n[yellow]Next steps:[/yellow]")
    console.print("1. Edit the configuration file with your API keys")
    console.print("2. Run 'enhanced-workflow run <query>' to execute queries")
    console.print("3. Run 'enhanced-workflow ui' to launch the web interface")


def _format_as_markdown(result: Any) -> str:
    """Format result as markdown"""
    if isinstance(result, dict):
        lines = []
        
        # Title
        if 'title' in result:
            lines.append(f"# {result['title']}")
            lines.append("")
        
        # Summary
        if 'summary' in result:
            lines.append(f"## Summary")
            lines.append(result['summary'])
            lines.append("")
        
        # Main content
        if 'content' in result:
            lines.append(f"## Content")
            lines.append(result['content'])
            lines.append("")
        
        # Code blocks
        if 'code' in result:
            lines.append(f"## Code")
            lines.append("```python")
            lines.append(result['code'])
            lines.append("```")
            lines.append("")
        
        # Other fields
        for key, value in result.items():
            if key not in ['title', 'summary', 'content', 'code']:
                lines.append(f"## {key.replace('_', ' ').title()}")
                if isinstance(value, list):
                    for item in value:
                        lines.append(f"- {item}")
                else:
                    lines.append(str(value))
                lines.append("")
        
        return "\n".join(lines)
    else:
        return str(result)


def _display_result_dict(result: Dict[str, Any]):
    """Display a result dictionary in a formatted way"""
    # Create panels for different sections
    for key, value in result.items():
        if isinstance(value, list) and value:
            # Display lists as bullet points
            content = "\n".join(f"• {item}" for item in value)
            panel = Panel(content, title=key.replace('_', ' ').title(), border_style="blue")
            console.print(panel)
        elif isinstance(value, dict):
            # Display nested dicts as JSON
            panel = Panel(
                json.dumps(value, indent=2),
                title=key.replace('_', ' ').title(),
                border_style="blue"
            )
            console.print(panel)
        elif isinstance(value, str) and len(value) > 50:
            # Display long strings in panels
            panel = Panel(value, title=key.replace('_', ' ').title(), border_style="blue")
            console.print(panel)
        else:
            # Display simple values inline
            console.print(f"[bold]{key.replace('_', ' ').title()}:[/bold] {value}")
        
        console.print()  # Add spacing


if __name__ == '__main__':
    cli()