"""Agent configuration optimization specialist"""

from typing import List, Dict, Any, Optional
from pydantic_ai import Agent, RunContext
from .base_agent import BaseAgent
from .dependencies import AgentRefinerDependencies
from .models import AgentRefineOutput
import logfire
import json
import yaml


class AgentRefinerAgent(BaseAgent[AgentRefinerDependencies, AgentRefineOutput]):
    """Specialized agent for optimizing other agent configurations"""
    
    def __init__(self):
        """Initialize the agent refiner agent"""
        super().__init__(
            model='openai:gpt-4',
            deps_type=AgentRefinerDependencies,
            result_type=AgentRefineOutput,
            enable_logfire=True
        )
    
    def get_system_prompt(self) -> str:
        """Get the system prompt for agent refiner"""
        return """You are an agent configuration expert that optimizes AI agent performance and efficiency.
        
        Your expertise includes:
        1. Analyzing agent performance metrics
        2. Optimizing agent prompts and configurations
        3. Identifying agent capability gaps
        4. Recommending agent improvements
        5. Balancing agent autonomy and control
        6. Ensuring agent reliability and consistency
        
        Focus on:
        - Performance optimization
        - Cost efficiency
        - Response quality
        - Agent specialization
        - Error recovery
        - Collaboration patterns
        
        Ensure all agent configurations are optimized for their specific roles."""
    
    def _register_tools(self):
        """Register tools for the agent refiner"""
        self.agent.tool(self.analyze_agent_performance)
        self.agent.tool(self.optimize_agent_config)
        self.agent.tool(self.recommend_agent_improvements)
        self.agent.tool(self.balance_agent_autonomy)
        self.agent.tool(self.create_agent_templates)
        self.agent.tool(self.validate_agent_behavior)
    
    async def analyze_agent_performance(
        self,
        ctx: RunContext[AgentRefinerDependencies],
        agent_id: str,
        metrics: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Analyze agent performance metrics
        
        Args:
            ctx: Run context
            agent_id: Agent identifier
            metrics: Performance metrics
            
        Returns:
            Performance analysis
        """
        logfire.info("analyzing_agent_performance", agent_id=agent_id)
        
        # Get metrics from context if not provided
        if not metrics and ctx.deps.performance_metrics:
            metrics = ctx.deps.performance_metrics.get(agent_id, {})
        
        analysis = {
            "agent_id": agent_id,
            "performance_score": 0.0,
            "efficiency_score": 0.0,
            "quality_score": 0.0,
            "issues": [],
            "strengths": [],
            "recommendations": []
        }
        
        if metrics:
            # Analyze response times
            avg_response_time = metrics.get("avg_response_time", 0)
            if avg_response_time > 5000:  # > 5 seconds
                analysis["issues"].append("Slow response times")
                analysis["recommendations"].append("Consider using a faster model or optimizing prompts")
            elif avg_response_time < 1000:  # < 1 second
                analysis["strengths"].append("Fast response times")
            
            # Analyze success rate
            success_rate = metrics.get("success_rate", 0)
            if success_rate < 0.8:
                analysis["issues"].append("Low success rate")
                analysis["recommendations"].append("Review error patterns and improve error handling")
            elif success_rate > 0.95:
                analysis["strengths"].append("High success rate")
            
            # Analyze token usage
            avg_tokens = metrics.get("avg_tokens_used", 0)
            if avg_tokens > 3000:
                analysis["issues"].append("High token usage")
                analysis["recommendations"].append("Optimize prompts to be more concise")
            
            # Calculate scores
            analysis["performance_score"] = min(1.0, 1000 / max(avg_response_time, 1))
            analysis["efficiency_score"] = min(1.0, 1000 / max(avg_tokens, 1))
            analysis["quality_score"] = success_rate
        
        # Overall score
        scores = [
            analysis["performance_score"],
            analysis["efficiency_score"],
            analysis["quality_score"]
        ]
        analysis["overall_score"] = sum(scores) / len(scores) if scores else 0.0
        
        return analysis
    
    async def optimize_agent_config(
        self,
        ctx: RunContext[AgentRefinerDependencies],
        current_config: Dict[str, Any],
        performance_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Optimize agent configuration
        
        Args:
            ctx: Run context
            current_config: Current agent configuration
            performance_analysis: Performance analysis results
            
        Returns:
            Optimized configuration
        """
        logfire.info("optimizing_agent_config", agent_id=current_config.get("agent_id"))
        
        optimized_config = current_config.copy()
        
        # Optimize model selection
        if performance_analysis.get("performance_score", 0) < 0.5:
            # Switch to faster model
            current_model = current_config.get("model", "gpt-4")
            if "gpt-4" in current_model:
                optimized_config["model"] = "gpt-3.5-turbo"
                optimized_config["optimization_notes"] = "Switched to faster model for better performance"
        
        # Optimize temperature
        if "Low success rate" in performance_analysis.get("issues", []):
            # Lower temperature for more consistent results
            optimized_config["temperature"] = max(0.0, current_config.get("temperature", 0.7) - 0.2)
        
        # Optimize max tokens
        if "High token usage" in performance_analysis.get("issues", []):
            current_max = current_config.get("max_tokens", 4000)
            optimized_config["max_tokens"] = int(current_max * 0.75)
        
        # Optimize retry strategy
        if performance_analysis.get("success_rate", 1.0) < 0.9:
            optimized_config["retry_config"] = {
                "max_retries": 3,
                "retry_delay": 1000,
                "exponential_backoff": True
            }
        
        # Optimize prompt
        if "system_prompt" in current_config:
            optimized_prompt = await self._optimize_prompt(
                current_config["system_prompt"],
                performance_analysis
            )
            optimized_config["system_prompt"] = optimized_prompt
        
        # Add performance guards
        optimized_config["performance_guards"] = {
            "max_response_time": 10000,  # 10 seconds
            "max_tokens_per_request": 3000,
            "enable_caching": True,
            "cache_ttl": 3600  # 1 hour
        }
        
        return optimized_config
    
    async def recommend_agent_improvements(
        self,
        ctx: RunContext[AgentRefinerDependencies],
        agent_config: Dict[str, Any],
        use_cases: List[str]
    ) -> List[Dict[str, Any]]:
        """Recommend improvements for agent
        
        Args:
            ctx: Run context
            agent_config: Agent configuration
            use_cases: Agent use cases
            
        Returns:
            List of improvement recommendations
        """
        logfire.info("recommending_agent_improvements", agent_id=agent_config.get("agent_id"))
        
        recommendations = []
        
        # Analyze current capabilities
        current_tools = agent_config.get("tools", [])
        current_skills = agent_config.get("skills", [])
        
        # Recommend additional tools
        for use_case in use_cases:
            required_tools = self._get_required_tools_for_use_case(use_case)
            missing_tools = [
                tool for tool in required_tools
                if tool not in current_tools
            ]
            
            if missing_tools:
                recommendations.append({
                    "type": "add_tools",
                    "priority": "high",
                    "description": f"Add tools for {use_case}",
                    "tools": missing_tools,
                    "expected_impact": "Improved capability coverage"
                })
        
        # Recommend skill enhancements
        if len(current_skills) < 3:
            recommendations.append({
                "type": "enhance_skills",
                "priority": "medium",
                "description": "Expand agent skill set",
                "suggested_skills": self._suggest_complementary_skills(current_skills),
                "expected_impact": "Broader problem-solving capability"
            })
        
        # Recommend prompt improvements
        if not agent_config.get("examples_in_prompt", False):
            recommendations.append({
                "type": "add_examples",
                "priority": "medium",
                "description": "Add few-shot examples to prompt",
                "rationale": "Examples improve task understanding",
                "expected_impact": "Better task execution"
            })
        
        # Recommend configuration optimizations
        if not agent_config.get("performance_guards"):
            recommendations.append({
                "type": "add_performance_guards",
                "priority": "high",
                "description": "Add performance monitoring and guards",
                "config_additions": {
                    "timeout": 30000,
                    "max_retries": 3,
                    "circuit_breaker": True
                },
                "expected_impact": "Improved reliability"
            })
        
        # Recommend collaboration patterns
        if len(use_cases) > 2:
            recommendations.append({
                "type": "enable_collaboration",
                "priority": "low",
                "description": "Enable multi-agent collaboration",
                "suggested_partners": self._suggest_collaborating_agents(use_cases),
                "expected_impact": "Better handling of complex tasks"
            })
        
        # Sort by priority
        priority_order = {"high": 0, "medium": 1, "low": 2}
        recommendations.sort(key=lambda x: priority_order.get(x["priority"], 3))
        
        return recommendations
    
    async def balance_agent_autonomy(
        self,
        ctx: RunContext[AgentRefinerDependencies],
        agent_config: Dict[str, Any],
        risk_tolerance: str = "medium"
    ) -> Dict[str, Any]:
        """Balance agent autonomy with control
        
        Args:
            ctx: Run context
            agent_config: Agent configuration
            risk_tolerance: Risk tolerance level
            
        Returns:
            Balanced autonomy configuration
        """
        logfire.info("balancing_agent_autonomy", risk_tolerance=risk_tolerance)
        
        autonomy_config = {
            "autonomy_level": "balanced",
            "decision_boundaries": {},
            "approval_requirements": [],
            "monitoring_level": "standard"
        }
        
        # Set autonomy based on risk tolerance
        if risk_tolerance == "low":
            autonomy_config["autonomy_level"] = "restricted"
            autonomy_config["decision_boundaries"] = {
                "max_cost_per_action": 10,
                "requires_approval_for": ["delete", "modify", "deploy"],
                "allowed_actions": ["read", "analyze", "suggest"]
            }
            autonomy_config["monitoring_level"] = "detailed"
            
        elif risk_tolerance == "high":
            autonomy_config["autonomy_level"] = "high"
            autonomy_config["decision_boundaries"] = {
                "max_cost_per_action": 100,
                "requires_approval_for": ["production_deploy"],
                "allowed_actions": ["all"]
            }
            autonomy_config["monitoring_level"] = "minimal"
            
        else:  # medium
            autonomy_config["autonomy_level"] = "balanced"
            autonomy_config["decision_boundaries"] = {
                "max_cost_per_action": 50,
                "requires_approval_for": ["delete", "production_deploy"],
                "allowed_actions": ["read", "analyze", "suggest", "create", "modify"]
            }
        
        # Add safety mechanisms
        autonomy_config["safety_mechanisms"] = {
            "rollback_enabled": True,
            "dry_run_mode": risk_tolerance == "low",
            "audit_logging": True,
            "rate_limiting": {
                "actions_per_minute": 10 if risk_tolerance == "low" else 60,
                "cost_per_hour": 100 if risk_tolerance == "low" else 1000
            }
        }
        
        # Add human-in-the-loop configurations
        if risk_tolerance != "high":
            autonomy_config["human_in_loop"] = {
                "confirmation_required": True,
                "confirmation_timeout": 300,  # 5 minutes
                "fallback_action": "abort"
            }
        
        return autonomy_config
    
    async def create_agent_templates(
        self,
        ctx: RunContext[AgentRefinerDependencies],
        agent_type: str,
        specialization: Optional[str] = None
    ) -> Dict[str, Any]:
        """Create optimized agent templates
        
        Args:
            ctx: Run context
            agent_type: Type of agent
            specialization: Optional specialization
            
        Returns:
            Agent template configuration
        """
        logfire.info("creating_agent_template", type=agent_type, spec=specialization)
        
        # Base template
        template = {
            "name": f"{agent_type}_agent",
            "type": agent_type,
            "version": "1.0.0",
            "model": "gpt-4",
            "temperature": 0.7,
            "max_tokens": 2000,
            "tools": [],
            "skills": [],
            "system_prompt": "",
            "performance_config": {},
            "autonomy_config": {}
        }
        
        # Customize based on agent type
        if agent_type == "analyzer":
            template.update({
                "model": "gpt-4",
                "temperature": 0.3,  # Lower for more consistent analysis
                "tools": ["search", "read_file", "analyze_data"],
                "skills": ["pattern_recognition", "data_analysis", "reporting"],
                "system_prompt": self._get_analyzer_prompt(specialization),
                "performance_config": {
                    "cache_analysis_results": True,
                    "parallel_analysis": True
                }
            })
            
        elif agent_type == "executor":
            template.update({
                "model": "gpt-3.5-turbo",  # Faster for execution
                "temperature": 0.5,
                "tools": ["execute_code", "file_operations", "api_calls"],
                "skills": ["task_execution", "error_handling", "validation"],
                "system_prompt": self._get_executor_prompt(specialization),
                "performance_config": {
                    "timeout": 30000,
                    "retry_on_failure": True
                }
            })
            
        elif agent_type == "coordinator":
            template.update({
                "model": "gpt-4",
                "temperature": 0.7,
                "tools": ["delegate_task", "monitor_progress", "aggregate_results"],
                "skills": ["planning", "delegation", "coordination"],
                "system_prompt": self._get_coordinator_prompt(specialization),
                "performance_config": {
                    "enable_parallel_coordination": True,
                    "progress_tracking": True
                }
            })
            
        elif agent_type == "validator":
            template.update({
                "model": "gpt-4",
                "temperature": 0.2,  # Very low for consistent validation
                "tools": ["run_tests", "check_compliance", "validate_output"],
                "skills": ["testing", "quality_assurance", "compliance"],
                "system_prompt": self._get_validator_prompt(specialization),
                "performance_config": {
                    "strict_mode": True,
                    "detailed_reporting": True
                }
            })
        
        # Add specialization-specific configurations
        if specialization:
            specialized_config = self._get_specialization_config(agent_type, specialization)
            template = self._merge_configs(template, specialized_config)
        
        # Add metadata
        template["metadata"] = {
            "created_by": "agent_refiner",
            "optimized_for": [agent_type, specialization] if specialization else [agent_type],
            "recommended_use_cases": self._get_recommended_use_cases(agent_type, specialization)
        }
        
        return template
    
    async def validate_agent_behavior(
        self,
        ctx: RunContext[AgentRefinerDependencies],
        agent_config: Dict[str, Any],
        test_scenarios: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Validate agent behavior against test scenarios
        
        Args:
            ctx: Run context
            agent_config: Agent configuration
            test_scenarios: Test scenarios
            
        Returns:
            Validation results
        """
        logfire.info("validating_agent_behavior", scenario_count=len(test_scenarios))
        
        validation_results = {
            "agent_id": agent_config.get("agent_id", "unknown"),
            "scenarios_tested": len(test_scenarios),
            "scenarios_passed": 0,
            "scenarios_failed": 0,
            "issues": [],
            "recommendations": [],
            "overall_score": 0.0
        }
        
        # Run each test scenario
        for scenario in test_scenarios:
            result = await self._run_scenario_test(agent_config, scenario)
            
            if result["passed"]:
                validation_results["scenarios_passed"] += 1
            else:
                validation_results["scenarios_failed"] += 1
                validation_results["issues"].append({
                    "scenario": scenario["name"],
                    "issue": result["issue"],
                    "severity": result["severity"]
                })
        
        # Calculate overall score
        if test_scenarios:
            validation_results["overall_score"] = (
                validation_results["scenarios_passed"] / len(test_scenarios)
            )
        
        # Generate recommendations based on failures
        if validation_results["scenarios_failed"] > 0:
            failure_patterns = self._analyze_failure_patterns(validation_results["issues"])
            
            for pattern in failure_patterns:
                if pattern["type"] == "consistency":
                    validation_results["recommendations"].append(
                        "Lower temperature for more consistent responses"
                    )
                elif pattern["type"] == "understanding":
                    validation_results["recommendations"].append(
                        "Improve prompt clarity and add examples"
                    )
                elif pattern["type"] == "capability":
                    validation_results["recommendations"].append(
                        "Add missing tools or enhance agent skills"
                    )
        
        return validation_results
    
    async def _optimize_prompt(
        self,
        current_prompt: str,
        performance_analysis: Dict[str, Any]
    ) -> str:
        """Optimize agent prompt based on performance"""
        optimized = current_prompt
        
        # Make prompt more concise if token usage is high
        if "High token usage" in performance_analysis.get("issues", []):
            # Remove redundant phrases
            redundant_phrases = [
                "please make sure to",
                "it is important that you",
                "you should always remember to"
            ]
            for phrase in redundant_phrases:
                optimized = optimized.replace(phrase, "")
            
            # Consolidate instructions
            optimized = self._consolidate_instructions(optimized)
        
        # Add clarity if success rate is low
        if performance_analysis.get("success_rate", 1.0) < 0.8:
            # Add structure markers
            if ":" not in optimized:
                sections = optimized.split('\n\n')
                if len(sections) > 1:
                    optimized = "INSTRUCTIONS:\n" + sections[0]
                    if len(sections) > 1:
                        optimized += "\n\nGUIDELINES:\n" + '\n\n'.join(sections[1:])
        
        return optimized.strip()
    
    def _consolidate_instructions(self, prompt: str) -> str:
        """Consolidate similar instructions in prompt"""
        lines = prompt.split('\n')
        consolidated = []
        seen_instructions = set()
        
        for line in lines:
            # Simple deduplication based on key words
            key_words = set(line.lower().split()[:5])
            if not any(kw in seen_instructions for kw in key_words):
                consolidated.append(line)
                seen_instructions.update(key_words)
        
        return '\n'.join(consolidated)
    
    def _get_required_tools_for_use_case(self, use_case: str) -> List[str]:
        """Get required tools for a use case"""
        use_case_tools = {
            "code_generation": ["file_operations", "code_executor", "syntax_checker"],
            "data_analysis": ["data_reader", "statistics", "visualization"],
            "api_integration": ["http_client", "json_parser", "auth_handler"],
            "testing": ["test_runner", "assertion_checker", "coverage_analyzer"],
            "deployment": ["build_tool", "deploy_tool", "monitor_tool"],
            "documentation": ["markdown_generator", "diagram_creator", "api_doc_gen"]
        }
        
        return use_case_tools.get(use_case, ["general_tool"])
    
    def _suggest_complementary_skills(self, current_skills: List[str]) -> List[str]:
        """Suggest complementary skills"""
        skill_complements = {
            "analysis": ["synthesis", "reporting"],
            "coding": ["debugging", "optimization"],
            "planning": ["execution", "monitoring"],
            "testing": ["debugging", "documentation"]
        }
        
        suggestions = []
        for skill in current_skills:
            suggestions.extend(skill_complements.get(skill, []))
        
        # Add general useful skills
        general_skills = ["error_handling", "logging", "validation"]
        suggestions.extend([s for s in general_skills if s not in current_skills])
        
        return list(set(suggestions))[:3]  # Return top 3 unique suggestions
    
    def _suggest_collaborating_agents(self, use_cases: List[str]) -> List[str]:
        """Suggest agents for collaboration"""
        collaborators = []
        
        if "code_generation" in use_cases:
            collaborators.append("code_reviewer_agent")
        
        if "data_analysis" in use_cases:
            collaborators.append("data_validator_agent")
        
        if "testing" in use_cases:
            collaborators.append("test_generator_agent")
        
        if len(use_cases) > 3:
            collaborators.append("coordinator_agent")
        
        return collaborators
    
    def _get_analyzer_prompt(self, specialization: Optional[str]) -> str:
        """Get analyzer agent prompt"""
        base = "You are an expert analyzer that examines data, code, and systems to provide insights."
        
        if specialization == "security":
            return f"{base}\nFocus on security vulnerabilities, best practices, and threat analysis."
        elif specialization == "performance":
            return f"{base}\nFocus on performance bottlenecks, optimization opportunities, and efficiency."
        
        return base
    
    def _get_executor_prompt(self, specialization: Optional[str]) -> str:
        """Get executor agent prompt"""
        base = "You are a reliable executor that performs tasks accurately and efficiently."
        
        if specialization == "deployment":
            return f"{base}\nSpecialize in deployment operations, rollbacks, and production safety."
        elif specialization == "automation":
            return f"{base}\nSpecialize in automating repetitive tasks and creating efficient workflows."
        
        return base
    
    def _get_coordinator_prompt(self, specialization: Optional[str]) -> str:
        """Get coordinator agent prompt"""
        base = "You are an expert coordinator that manages complex multi-agent workflows."
        
        if specialization == "project":
            return f"{base}\nFocus on project management, timeline coordination, and resource allocation."
        
        return base
    
    def _get_validator_prompt(self, specialization: Optional[str]) -> str:
        """Get validator agent prompt"""
        base = "You are a meticulous validator that ensures quality and correctness."
        
        if specialization == "compliance":
            return f"{base}\nFocus on regulatory compliance, standards adherence, and audit trails."
        
        return base
    
    def _get_specialization_config(
        self,
        agent_type: str,
        specialization: str
    ) -> Dict[str, Any]:
        """Get specialization-specific configuration"""
        configs = {
            ("analyzer", "security"): {
                "tools": ["security_scanner", "vulnerability_db"],
                "skills": ["threat_modeling", "penetration_testing"]
            },
            ("executor", "deployment"): {
                "tools": ["kubernetes_client", "docker_api"],
                "skills": ["rollback_management", "health_checking"]
            }
        }
        
        return configs.get((agent_type, specialization), {})
    
    def _merge_configs(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """Merge configurations"""
        merged = base.copy()
        
        for key, value in override.items():
            if key in merged and isinstance(merged[key], list):
                merged[key].extend(value)
                merged[key] = list(set(merged[key]))  # Remove duplicates
            elif key in merged and isinstance(merged[key], dict):
                merged[key].update(value)
            else:
                merged[key] = value
        
        return merged
    
    def _get_recommended_use_cases(
        self,
        agent_type: str,
        specialization: Optional[str]
    ) -> List[str]:
        """Get recommended use cases for agent type"""
        use_cases = {
            "analyzer": ["code_review", "data_analysis", "system_audit"],
            "executor": ["task_automation", "deployment", "data_processing"],
            "coordinator": ["workflow_management", "team_coordination", "pipeline_orchestration"],
            "validator": ["testing", "quality_assurance", "compliance_checking"]
        }
        
        base_cases = use_cases.get(agent_type, ["general_purpose"])
        
        if specialization:
            base_cases.append(f"{specialization}_specialized")
        
        return base_cases
    
    async def _run_scenario_test(
        self,
        agent_config: Dict[str, Any],
        scenario: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Run a single scenario test"""
        # This would actually run the agent with the scenario
        # For now, return simulated results
        
        result = {
            "passed": True,
            "issue": None,
            "severity": None
        }
        
        # Simulate some failures based on config
        if agent_config.get("temperature", 0.7) > 0.8 and scenario.get("requires_consistency"):
            result["passed"] = False
            result["issue"] = "Inconsistent responses"
            result["severity"] = "medium"
        
        return result
    
    def _analyze_failure_patterns(self, issues: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """Analyze patterns in test failures"""
        patterns = []
        
        # Count issue types
        issue_counts = {}
        for issue in issues:
            issue_type = issue["issue"]
            issue_counts[issue_type] = issue_counts.get(issue_type, 0) + 1
        
        # Identify patterns
        for issue_type, count in issue_counts.items():
            if count > 1:
                if "inconsistent" in issue_type.lower():
                    patterns.append({"type": "consistency", "count": count})
                elif "understand" in issue_type.lower():
                    patterns.append({"type": "understanding", "count": count})
                else:
                    patterns.append({"type": "capability", "count": count})
        
        return patterns
    
    async def refine(self, agent_data: Dict[str, Any]) -> Dict[str, Any]:
        """Refine agent configurations"""
        agent_id = agent_data.get('agent_id', 'unknown')
        
        # Analyze current performance
        performance = await self.analyze_agent_performance(None, agent_id)
        
        if performance['overall_score'] < 0.8:
            # Optimize configuration
            optimized_config = await self.optimize_agent_config(
                None,
                agent_data.get('config', {}),
                performance
            )
            
            # Get improvement recommendations
            recommendations = await self.recommend_agent_improvements(
                None,
                optimized_config,
                agent_data.get('use_cases', [])
            )
            
            # Update agent data
            agent_data['config'] = optimized_config
            agent_data['performance_analysis'] = performance
            agent_data['improvement_recommendations'] = recommendations
            
            # Balance autonomy based on performance
            if performance['overall_score'] < 0.6:
                autonomy_config = await self.balance_agent_autonomy(
                    None,
                    optimized_config,
                    "low"  # Restrict autonomy for poor performers
                )
                agent_data['autonomy_config'] = autonomy_config
        
        return agent_data