"""Context loader for project files and documentation."""

from pathlib import Path
from typing import Dict, Optional, Any
from .models import ProjectContext

class ContextLoader:
    """Loads and manages project context for agents."""
    
    def __init__(self, project_root: Optional[Path] = None):
        """Initialize context loader.
        
        Args:
            project_root: Root directory of the project. Defaults to current directory.
        """
        self.root = Path(project_root) if project_root else Path.cwd()
        self._cache: Dict[str, Any] = {}
        
    def _load_file(self, filename: str, required: bool = False) -> Optional[str]:
        """Load a file from the project root.
        
        Args:
            filename: Name of the file to load
            required: Whether the file is required to exist
            
        Returns:
            File contents or None if not found and not required
            
        Raises:
            FileNotFoundError: If required file is not found
        """
        file_path = self.root / filename
        
        if not file_path.exists():
            if required:
                raise FileNotFoundError(f"Required file {filename} not found at {file_path}")
            return None
            
        try:
            return file_path.read_text(encoding='utf-8')
        except Exception as e:
            if required:
                raise
            print(f"Warning: Could not read {filename}: {e}")
            return None
    
    def load_project_context(self) -> ProjectContext:
        """Load complete project context.
        
        Returns:
            ProjectContext with all available context files
        """
        # Check cache first
        if "project_context" in self._cache:
            return self._cache["project_context"]
        
        # Load CLAUDE.md (required)
        claude_rules = self._load_file("CLAUDE.md", required=False)
        if not claude_rules:
            # Use default rules if not present
            claude_rules = self._get_default_claude_rules()
            
        # Load optional files
        planning = self._load_file("PLANNING.md", required=False)
        tasks = self._load_file("TASK.md", required=False)
        
        # Load examples
        examples = self._load_examples()
        
        # Load PRPs
        prps = self._load_prps()
        
        # Create and cache context
        context = ProjectContext(
            claude_rules=claude_rules,
            planning=planning,
            tasks=tasks,
            examples=examples,
            prps=prps
        )
        
        self._cache["project_context"] = context
        return context
    
    def _load_examples(self) -> Dict[str, str]:
        """Load example files from examples directory.
        
        Returns:
            Dictionary mapping filename to content
        """
        examples: Dict[str, str] = {}
        examples_dir = self.root / "examples"
        
        if not examples_dir.exists():
            return examples
            
        # Load Python files
        for py_file in examples_dir.glob("*.py"):
            try:
                examples[py_file.name] = py_file.read_text(encoding='utf-8')
            except Exception as e:
                print(f"Warning: Could not read example {py_file.name}: {e}")
                
        return examples
    
    def _load_prps(self) -> Dict[str, str]:
        """Load PRP files from PRPs directory.
        
        Returns:
            Dictionary mapping PRP name to content
        """
        prps: Dict[str, str] = {}
        prps_dir = self.root / "PRPs"
        
        if not prps_dir.exists():
            return prps
            
        # Load markdown files (excluding templates)
        for md_file in prps_dir.glob("*.md"):
            if md_file.parent.name != "templates":
                try:
                    prps[md_file.stem] = md_file.read_text(encoding='utf-8')
                except Exception as e:
                    print(f"Warning: Could not read PRP {md_file.name}: {e}")
                    
        return prps
    
    def _get_default_claude_rules(self) -> str:
        """Get default CLAUDE.md rules for the project."""
        return """### 🔄 Project Awareness & Context
- **Always check available tools** before implementing new functionality
- **Use the tool discovery system** to find and load appropriate tools
- **Follow existing patterns** in the codebase for consistency

### 🧱 Code Structure & Modularity
- **Never create a file longer than 500 lines**. Split into modules if needed.
- **Organize code into clearly separated modules** by feature or responsibility
- **Use clear, consistent imports** and prefer relative imports within packages
- **Follow the existing tool pattern** when creating new tools

### 🧪 Testing & Reliability
- **Always create unit tests for new features** (tools, agents, etc)
- **Tests should live in /tests** mirroring the main structure
- **Include edge cases and error scenarios** in tests

### ✅ Task Completion
- **Mark tasks complete** only when fully implemented and tested
- **Use the task_done_tool** to signal completion with summary

### 📎 Style & Conventions
- **Use Python** as the primary language
- **Follow PEP8** and use type hints
- **Format with black** or similar formatter
- **Write docstrings** for all functions and classes

### 📚 Documentation
- **Update README.md** when adding new features
- **Comment non-obvious code** with explanations
- **Keep configuration in config.yaml** not hardcoded

### 🧠 AI Behavior Rules
- **Never assume missing context** - ask if uncertain
- **Never hallucinate libraries** - only use verified packages
- **Always validate file paths** before operations
- **Follow the multi-agent patterns** from orchestrator.py"""
    
    def clear_cache(self):
        """Clear the context cache."""
        self._cache.clear()