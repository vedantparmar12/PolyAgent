"""Data models for context engineering system."""

from pydantic import BaseModel, Field
from typing import List, Dict, Optional
from enum import Enum

class ContextType(str, Enum):
    """Types of context files in the system."""
    CLAUDE_MD = "claude_md"
    PLANNING = "planning"
    TASK = "task"
    EXAMPLE = "example"
    PRP = "prp"

class ProjectContext(BaseModel):
    """Complete project context for agents."""
    claude_rules: str = Field(description="Content of CLAUDE.md")
    planning: Optional[str] = Field(default=None, description="Content of PLANNING.md")
    tasks: Optional[str] = Field(default=None, description="Content of TASK.md")
    examples: Dict[str, str] = Field(default_factory=dict, description="Example code files")
    prps: Dict[str, str] = Field(default_factory=dict, description="Available PRPs")
    
    def get_formatted_context(self) -> str:
        """Format context for inclusion in agent prompts."""
        sections = []
        
        sections.append("## Project Rules (CLAUDE.md)")
        sections.append(self.claude_rules)
        
        if self.planning:
            sections.append("\n## Architecture & Planning")
            sections.append(self.planning)
            
        if self.tasks:
            sections.append("\n## Current Tasks")
            sections.append(self.tasks)
            
        if self.examples:
            sections.append("\n## Available Examples")
            for name in self.examples.keys():
                sections.append(f"- {name}")
                
        return "\n".join(sections)

class PRPRequest(BaseModel):
    """Request to generate a PRP."""
    feature_description: str = Field(description="What to build")
    examples: List[str] = Field(default_factory=list, description="Example files to reference")
    documentation_urls: List[str] = Field(default_factory=list, description="Documentation URLs")
    considerations: Optional[str] = Field(default=None, description="Special considerations")

class ValidationResult(BaseModel):
    """Result of code validation."""
    success: bool
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    fixed_code: Optional[str] = Field(default=None)
    
    def get_summary(self) -> str:
        """Get a summary of validation results."""
        if self.success:
            return "✅ Validation passed"
        else:
            error_count = len(self.errors)
            warning_count = len(self.warnings)
            return f"❌ Validation failed: {error_count} errors, {warning_count} warnings"