"""Type-safe dependency definitions for Pydantic AI agents"""

from dataclasses import dataclass
from typing import Optional, Dict, Any, List
from supabase import Client as SupabaseClient
import httpx


@dataclass
class BaseDependencies:
    """Base dependencies for all agents"""
    user_id: str
    session_id: str
    api_keys: Dict[str, str]
    http_client: httpx.AsyncClient


@dataclass
class AdvisorDependencies(BaseDependencies):
    """Dependencies for advisor agent"""
    vector_client: SupabaseClient
    examples_path: str
    context_limit: int = 5


@dataclass
class ScopeDependencies(BaseDependencies):
    """Dependencies for scope reasoner agent"""
    project_context: Dict[str, Any]
    task_history: List[Dict[str, Any]]
    reasoning_depth: int = 3


@dataclass
class CoderDependencies(BaseDependencies):
    """Dependencies for coder agent"""
    workspace_path: str
    git_repo: Optional[str] = None
    tool_configs: Dict[str, Any] = None
    language_configs: Dict[str, Any] = None


@dataclass
class RefinerDependencies(BaseDependencies):
    """Dependencies for refiner agent"""
    validation_config: Dict[str, bool]
    max_retry_attempts: int = 3
    improvement_patterns: List[str] = None


@dataclass
class PromptRefinerDependencies(BaseDependencies):
    """Dependencies for prompt refiner agent"""
    prompt_patterns: List[Dict[str, str]]
    evaluation_metrics: List[str]
    test_cases: Optional[List[Dict[str, Any]]] = None


@dataclass
class ToolsRefinerDependencies(BaseDependencies):
    """Dependencies for tools refiner agent"""
    mcp_servers: List[Dict[str, Any]]
    tool_library: Dict[str, Any]
    validation_suite: Optional[Dict[str, Any]] = None


@dataclass
class AgentRefinerDependencies(BaseDependencies):
    """Dependencies for agent configuration refiner"""
    agent_templates: List[Dict[str, Any]]
    performance_metrics: Dict[str, float]
    optimization_goals: List[str]


@dataclass
class SynthesisDependencies(BaseDependencies):
    """Dependencies for synthesis agent"""
    agent_outputs: Dict[str, Any]
    synthesis_strategy: str = "comprehensive"
    output_format: str = "structured"


@dataclass
class ResearchDependencies(BaseDependencies):
    """Dependencies for research-focused agent (Grok heavy mode)"""
    search_engines: List[str]
    fact_check_sources: List[str]
    credibility_threshold: float = 0.8


@dataclass
class AnalysisDependencies(BaseDependencies):
    """Dependencies for analysis-focused agent (Grok heavy mode)"""
    analysis_frameworks: List[str]
    quantitative_tools: Dict[str, Any]
    visualization_configs: Optional[Dict[str, Any]] = None


@dataclass
class PerspectiveDependencies(BaseDependencies):
    """Dependencies for perspective-focused agent (Grok heavy mode)"""
    viewpoint_sources: List[str]
    bias_detection_config: Dict[str, Any]
    stakeholder_profiles: Optional[List[Dict[str, Any]]] = None


@dataclass
class VerificationDependencies(BaseDependencies):
    """Dependencies for verification-focused agent (Grok heavy mode)"""
    fact_check_apis: List[str]
    verification_databases: List[str]
    confidence_thresholds: Dict[str, float]