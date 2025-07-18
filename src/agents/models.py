"""Structured response models for Pydantic AI agents"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from enum import Enum


class TaskComplexity(str, Enum):
    """Task complexity levels"""
    SIMPLE = "simple"
    COMPLEX = "complex"
    RESEARCH = "research"


class ContextOutput(BaseModel):
    """Structured output from advisor agent"""
    relevant_examples: List[str] = Field(description="Relevant code examples found")
    context_summary: str = Field(description="Summary of relevant context")
    recommendations: List[str] = Field(description="Specific recommendations")
    confidence_score: float = Field(ge=0, le=1, description="Confidence in recommendations")


class ScopeOutput(BaseModel):
    """Structured output from scope reasoner agent"""
    task_breakdown: List[str] = Field(description="Breakdown of task into subtasks")
    complexity_assessment: TaskComplexity = Field(description="Assessed complexity level")
    estimated_effort: str = Field(description="Estimated time/effort required")
    dependencies: List[str] = Field(description="Task dependencies identified")
    risk_factors: List[str] = Field(description="Potential risk factors")


class CodeOutput(BaseModel):
    """Structured output from coder agent"""
    generated_code: str = Field(description="Generated code implementation")
    file_changes: List[Dict[str, str]] = Field(description="Files that need to be modified")
    test_cases: Optional[str] = Field(description="Generated test cases")
    documentation: Optional[str] = Field(description="Generated documentation")
    next_steps: List[str] = Field(description="Next steps for implementation")


class ValidationResult(BaseModel):
    """Structured validation result"""
    test_passed: bool = Field(description="Whether tests passed")
    lint_passed: bool = Field(description="Whether linting passed")
    type_check_passed: bool = Field(description="Whether type checking passed")
    errors: List[str] = Field(default_factory=list, description="List of errors found")
    warnings: List[str] = Field(default_factory=list, description="List of warnings")


class RefineOutput(BaseModel):
    """Structured output from refiner agent"""
    refined_code: str = Field(description="Improved code after refinement")
    validation_results: ValidationResult = Field(description="Validation results")
    improvements_made: List[str] = Field(description="List of improvements applied")
    remaining_issues: List[str] = Field(description="Issues that still need attention")
    refinement_complete: bool = Field(description="Whether refinement is complete")


class FinalOutput(BaseModel):
    """Final synthesized output"""
    solution: str = Field(description="Complete solution description")
    implementation_plan: List[str] = Field(description="Step-by-step implementation plan")
    code_artifacts: Dict[str, str] = Field(description="Generated code files")
    validation_summary: ValidationResult = Field(description="Overall validation status")
    confidence_score: float = Field(ge=0, le=1, description="Overall confidence score")
    success_metrics: Dict[str, Any] = Field(description="Success metrics and KPIs")


# Grok Heavy Mode Models
class GrokQuestionSet(BaseModel):
    """Set of 4 specialized research questions for Grok heavy mode"""
    research_question: str = Field(description="Research-focused question for background information")
    analysis_question: str = Field(description="Analysis-focused question for achievements/contributions")
    perspective_question: str = Field(description="Perspective-focused question for alternative viewpoints")
    verification_question: str = Field(description="Verification-focused question for fact-checking")


class ResearchOutput(BaseModel):
    """Output from research-focused agent"""
    factual_findings: List[str] = Field(description="Key factual findings")
    background_information: str = Field(description="Comprehensive background information")
    sources: List[str] = Field(description="Sources consulted")
    confidence_level: float = Field(ge=0, le=1, description="Confidence in findings")


class AnalysisOutput(BaseModel):
    """Output from analysis-focused agent"""
    achievements: List[str] = Field(description="Key achievements identified")
    contributions: List[str] = Field(description="Significant contributions")
    impact_assessment: str = Field(description="Overall impact analysis")
    quantitative_metrics: Dict[str, Any] = Field(description="Quantitative measures where available")


class PerspectiveOutput(BaseModel):
    """Output from perspective-focused agent"""
    alternative_viewpoints: List[str] = Field(description="Different perspectives identified")
    broader_context: str = Field(description="Broader contextual analysis")
    potential_criticisms: List[str] = Field(description="Potential criticisms or limitations")
    stakeholder_views: Dict[str, str] = Field(description="Different stakeholder perspectives")


class VerificationOutput(BaseModel):
    """Output from verification-focused agent"""
    verified_facts: List[str] = Field(description="Facts that have been verified")
    questionable_claims: List[str] = Field(description="Claims requiring further verification")
    current_status: str = Field(description="Current status validation")
    credibility_assessment: Dict[str, float] = Field(description="Credibility scores for different claims")


class GrokSynthesisOutput(BaseModel):
    """Final comprehensive synthesis output for Grok heavy mode"""
    comprehensive_analysis: str = Field(description="Complete synthesized analysis")
    key_insights: List[str] = Field(description="Top insights from all perspectives")
    confidence_score: float = Field(ge=0, le=1, description="Overall confidence in analysis")
    sources: List[str] = Field(description="All sources used across agents")
    agent_contributions: Dict[str, str] = Field(description="Summary of each agent's contribution")
    methodology: str = Field(description="Description of analysis methodology used")


# Refiner Agent Models
class PromptRefineOutput(BaseModel):
    """Output from prompt refiner agent"""
    optimized_prompt: str = Field(description="Optimized system prompt")
    improvements_made: List[str] = Field(description="List of improvements applied")
    effectiveness_score: float = Field(ge=0, le=1, description="Predicted effectiveness")
    reasoning: str = Field(description="Reasoning behind optimizations")


class ToolsRefineOutput(BaseModel):
    """Output from tools refiner agent"""
    optimized_tools: List[Dict[str, Any]] = Field(description="Optimized tool configurations")
    new_tools_added: List[str] = Field(description="New tools added")
    tools_removed: List[str] = Field(description="Tools removed or deprecated")
    mcp_configurations: Dict[str, Any] = Field(description="MCP server configurations")
    validation_results: Dict[str, bool] = Field(description="Tool validation results")


class AgentRefineOutput(BaseModel):
    """Output from agent configuration refiner"""
    optimized_config: Dict[str, Any] = Field(description="Optimized agent configuration")
    performance_improvements: Dict[str, float] = Field(description="Performance metric improvements")
    configuration_changes: List[str] = Field(description="List of configuration changes made")
    recommendations: List[str] = Field(description="Further optimization recommendations")