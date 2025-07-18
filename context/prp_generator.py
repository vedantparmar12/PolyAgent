"""PRP (Product Requirements Prompt) generator for context-aware implementations."""

from typing import Optional
import json
from pathlib import Path
from .models import PRPRequest
from agent import OpenRouterAgent

class PRPGenerator:
    """Generates comprehensive PRPs from feature requests."""
    
    def __init__(self, agent: Optional[OpenRouterAgent] = None):
        """Initialize PRP generator.
        
        Args:
            agent: OpenRouterAgent instance for AI generation
        """
        self.agent = agent or OpenRouterAgent(silent=True)
        self.template = self._load_template()
        
    def _load_template(self) -> str:
        """Load the PRP template."""
        template_path = Path("PRPs/templates/prp_base.md")
        if not template_path.exists():
            # Use default template if file doesn't exist
            return self._get_default_template()
        
        return template_path.read_text(encoding='utf-8')
    
    def generate_prp(self, request: PRPRequest) -> str:
        """Generate a comprehensive PRP from a feature request.
        
        Args:
            request: PRPRequest with feature details
            
        Returns:
            Generated PRP content
        """
        # Research codebase for patterns
        research_prompt = f"""
Research the codebase to understand patterns and conventions for implementing:
{request.feature_description}

Look for:
1. Similar features or patterns
2. Code structure and organization
3. Testing patterns
4. Configuration patterns
5. Tool implementation patterns

Use the search_web tool if needed for external documentation.
Provide a comprehensive analysis.
"""
        
        research_results = self.agent.run(research_prompt)
        
        # Build comprehensive prompt for PRP generation
        generation_prompt = f"""
Generate a comprehensive PRP (Product Requirements Prompt) for the following feature:

FEATURE DESCRIPTION:
{request.feature_description}

EXAMPLES TO REFERENCE:
{json.dumps(request.examples, indent=2)}

DOCUMENTATION URLS:
{json.dumps(request.documentation_urls, indent=2)}

SPECIAL CONSIDERATIONS:
{request.considerations or "None"}

RESEARCH RESULTS:
{research_results}

TEMPLATE TO FOLLOW:
{self.template}

Generate a complete PRP following the template structure exactly. Be specific about:
1. Implementation details and pseudocode
2. File paths and modifications
3. Testing requirements
4. Validation steps
5. Integration points

The PRP should be comprehensive enough that an AI agent can implement the feature successfully.
"""
        
        # Generate the PRP
        prp_content = self.agent.run(generation_prompt)
        
        # Validate PRP structure
        if not self._validate_prp_structure(prp_content):
            # Try to fix structure
            fix_prompt = f"""
The generated PRP is missing required sections. Please fix it to include all sections from the template:

CURRENT PRP:
{prp_content}

REQUIRED TEMPLATE:
{self.template}

Generate a corrected version with all required sections.
"""
            prp_content = self.agent.run(fix_prompt)
        
        return prp_content
    
    def _validate_prp_structure(self, prp_content: str) -> bool:
        """Validate that PRP has all required sections."""
        required_sections = [
            "## Goal",
            "## Why", 
            "## What",
            "### Success Criteria",
            "## All Needed Context",
            "## Implementation Blueprint",
            "## Validation Loop",
            "## Final Validation Checklist"
        ]
        
        return all(section in prp_content for section in required_sections)
    
    def save_prp(self, prp_content: str, filename: str) -> Path:
        """Save PRP to file.
        
        Args:
            prp_content: PRP content to save
            filename: Name for the PRP file (without extension)
            
        Returns:
            Path to saved file
        """
        prps_dir = Path("PRPs")
        prps_dir.mkdir(exist_ok=True)
        
        # Ensure .md extension
        if not filename.endswith('.md'):
            filename += '.md'
            
        filepath = prps_dir / filename
        filepath.write_text(prp_content, encoding='utf-8')
        
        return filepath
    
    def _get_default_template(self) -> str:
        """Get default PRP template."""
        return """name: "Feature PRP"
description: |

## Purpose
[Purpose of this PRP]

## Goal
[What needs to be built]

## Why
- [Business value]
- [User impact]

## What
[Technical requirements]

### Success Criteria
- [ ] [Measurable outcome]

## All Needed Context

### Documentation & References
```yaml
- file: [path/to/file]
  why: [reason]
```

### Current Codebase Structure
```bash
[Structure]
```

### Implementation Blueprint

### List of Tasks
```yaml
Task 1: [Name]
```

## Validation Loop

### Level 1: Syntax & Style
```bash
ruff check . --fix
```

### Level 2: Unit Tests
```bash
pytest tests/ -v
```

## Final Validation Checklist
- [ ] All tests pass
- [ ] No linting errors

## Anti-Patterns to Avoid
- ❌ [Pattern to avoid]
"""