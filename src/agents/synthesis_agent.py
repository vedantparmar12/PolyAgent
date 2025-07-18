"""Synthesis agent for combining and integrating results"""

from typing import List, Dict, Any, Optional, Union
from pydantic_ai import Agent, RunContext
from .base_agent import BaseAgent
from .dependencies import SynthesisDependencies
from .models import SynthesisOutput
import logfire
import json
from datetime import datetime


class SynthesisAgent(BaseAgent[SynthesisDependencies, SynthesisOutput]):
    """Agent that synthesizes results from multiple sources into cohesive outputs"""
    
    def __init__(self):
        """Initialize the synthesis agent"""
        super().__init__(
            model='openai:gpt-4',
            deps_type=SynthesisDependencies,
            result_type=SynthesisOutput,
            enable_logfire=True
        )
    
    def get_system_prompt(self) -> str:
        """Get the system prompt for synthesis agent"""
        return """You are an expert at synthesizing information from multiple sources into coherent, actionable outputs.
        
        Your expertise includes:
        1. Combining results from multiple agents
        2. Resolving conflicts and contradictions
        3. Creating comprehensive summaries
        4. Generating unified outputs
        5. Identifying patterns and insights
        6. Producing actionable recommendations
        
        Focus on:
        - Clarity and coherence
        - Identifying key insights
        - Resolving ambiguities
        - Creating actionable outputs
        - Maintaining context consistency
        - Highlighting important patterns
        
        Always produce well-structured, comprehensive syntheses that provide clear value."""
    
    def _register_tools(self):
        """Register tools for the synthesis agent"""
        self.agent.tool(self.combine_agent_results)
        self.agent.tool(self.resolve_conflicts)
        self.agent.tool(self.create_summary)
        self.agent.tool(self.identify_patterns)
        self.agent.tool(self.generate_recommendations)
        self.agent.tool(self.create_unified_output)
    
    async def combine_agent_results(
        self,
        ctx: RunContext[SynthesisDependencies],
        results: List[Dict[str, Any]],
        merge_strategy: str = "intelligent"
    ) -> Dict[str, Any]:
        """Combine results from multiple agents
        
        Args:
            ctx: Run context
            results: List of agent results
            merge_strategy: Strategy for merging
            
        Returns:
            Combined results
        """
        logfire.info("combining_agent_results", count=len(results), strategy=merge_strategy)
        
        combined = {
            "merged_at": datetime.utcnow().isoformat(),
            "source_count": len(results),
            "merge_strategy": merge_strategy,
            "data": {},
            "metadata": {
                "sources": [],
                "conflicts_resolved": 0,
                "confidence": 1.0
            }
        }
        
        if not results:
            return combined
        
        # Extract source information
        for i, result in enumerate(results):
            source_info = {
                "index": i,
                "agent": result.get("agent_name", f"agent_{i}"),
                "timestamp": result.get("timestamp", datetime.utcnow().isoformat()),
                "success": result.get("success", True)
            }
            combined["metadata"]["sources"].append(source_info)
        
        # Apply merge strategy
        if merge_strategy == "first":
            combined["data"] = results[0].get("data", {})
            
        elif merge_strategy == "last":
            combined["data"] = results[-1].get("data", {})
            
        elif merge_strategy == "vote":
            combined["data"] = self._merge_by_voting(results)
            
        elif merge_strategy == "intelligent":
            # Intelligent merging with conflict resolution
            merged_data = {}
            conflicts = []
            
            # Collect all unique keys
            all_keys = set()
            for result in results:
                if isinstance(result.get("data"), dict):
                    all_keys.update(result["data"].keys())
            
            # Merge each key
            for key in all_keys:
                values = []
                sources = []
                
                for i, result in enumerate(results):
                    if key in result.get("data", {}):
                        values.append(result["data"][key])
                        sources.append(i)
                
                if len(set(str(v) for v in values)) == 1:
                    # All values agree
                    merged_data[key] = values[0]
                else:
                    # Conflict detected
                    conflicts.append({
                        "key": key,
                        "values": values,
                        "sources": sources
                    })
                    
                    # Resolve conflict
                    resolved_value = await self._resolve_single_conflict(
                        key, values, sources, results
                    )
                    merged_data[key] = resolved_value
                    combined["metadata"]["conflicts_resolved"] += 1
            
            combined["data"] = merged_data
            
            # Adjust confidence based on conflicts
            if conflicts:
                combined["metadata"]["confidence"] = max(
                    0.5,
                    1.0 - (len(conflicts) / len(all_keys))
                )
        
        # Add synthesis metadata
        combined["metadata"]["synthesis_method"] = self._determine_synthesis_method(results)
        combined["metadata"]["quality_score"] = self._assess_result_quality(combined)
        
        return combined
    
    async def resolve_conflicts(
        self,
        ctx: RunContext[SynthesisDependencies],
        conflicts: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Resolve conflicts between different results
        
        Args:
            ctx: Run context
            conflicts: List of conflicts to resolve
            
        Returns:
            Resolution results
        """
        logfire.info("resolving_conflicts", count=len(conflicts))
        
        resolutions = {
            "resolved_count": 0,
            "failed_count": 0,
            "resolutions": [],
            "unresolved": []
        }
        
        for conflict in conflicts:
            resolution = await self._resolve_conflict(conflict, ctx)
            
            if resolution["resolved"]:
                resolutions["resolved_count"] += 1
                resolutions["resolutions"].append(resolution)
            else:
                resolutions["failed_count"] += 1
                resolutions["unresolved"].append(conflict)
        
        # Generate resolution summary
        resolutions["summary"] = self._create_resolution_summary(resolutions)
        
        # Suggest manual intervention if needed
        if resolutions["failed_count"] > 0:
            resolutions["manual_intervention_needed"] = True
            resolutions["intervention_suggestions"] = self._suggest_interventions(
                resolutions["unresolved"]
            )
        
        return resolutions
    
    async def create_summary(
        self,
        ctx: RunContext[SynthesisDependencies],
        data: Dict[str, Any],
        summary_type: str = "executive"
    ) -> str:
        """Create summary of synthesized data
        
        Args:
            ctx: Run context
            data: Data to summarize
            summary_type: Type of summary
            
        Returns:
            Summary text
        """
        logfire.info("creating_summary", type=summary_type)
        
        if summary_type == "executive":
            return self._create_executive_summary(data)
        elif summary_type == "technical":
            return self._create_technical_summary(data)
        elif summary_type == "detailed":
            return self._create_detailed_summary(data)
        else:
            return self._create_general_summary(data)
    
    async def identify_patterns(
        self,
        ctx: RunContext[SynthesisDependencies],
        data_points: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Identify patterns in synthesized data
        
        Args:
            ctx: Run context
            data_points: Data points to analyze
            
        Returns:
            Identified patterns
        """
        logfire.info("identifying_patterns", data_count=len(data_points))
        
        patterns = []
        
        # Temporal patterns
        temporal_patterns = self._identify_temporal_patterns(data_points)
        patterns.extend(temporal_patterns)
        
        # Value patterns
        value_patterns = self._identify_value_patterns(data_points)
        patterns.extend(value_patterns)
        
        # Correlation patterns
        if len(data_points) > 10:
            correlation_patterns = self._identify_correlations(data_points)
            patterns.extend(correlation_patterns)
        
        # Anomaly patterns
        anomalies = self._identify_anomalies(data_points)
        if anomalies:
            patterns.append({
                "type": "anomaly",
                "description": "Unusual data points detected",
                "instances": anomalies,
                "severity": self._assess_anomaly_severity(anomalies)
            })
        
        # Rank patterns by significance
        for pattern in patterns:
            pattern["significance"] = self._calculate_pattern_significance(pattern)
        
        patterns.sort(key=lambda x: x["significance"], reverse=True)
        
        return patterns
    
    async def generate_recommendations(
        self,
        ctx: RunContext[SynthesisDependencies],
        synthesis_results: Dict[str, Any],
        patterns: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Generate actionable recommendations
        
        Args:
            ctx: Run context
            synthesis_results: Synthesized results
            patterns: Identified patterns
            
        Returns:
            List of recommendations
        """
        logfire.info("generating_recommendations")
        
        recommendations = []
        
        # Pattern-based recommendations
        for pattern in patterns:
            if pattern["significance"] > 0.7:
                rec = self._create_pattern_recommendation(pattern)
                if rec:
                    recommendations.append(rec)
        
        # Quality-based recommendations
        quality_score = synthesis_results.get("metadata", {}).get("quality_score", 1.0)
        if quality_score < 0.8:
            recommendations.append({
                "type": "quality_improvement",
                "priority": "high",
                "title": "Improve Data Quality",
                "description": "The synthesis quality score is below optimal",
                "actions": [
                    "Review source data for completeness",
                    "Resolve remaining conflicts",
                    "Gather additional data points"
                ],
                "impact": "high",
                "effort": "medium"
            })
        
        # Conflict-based recommendations
        conflicts_resolved = synthesis_results.get("metadata", {}).get("conflicts_resolved", 0)
        if conflicts_resolved > 5:
            recommendations.append({
                "type": "process_improvement",
                "priority": "medium",
                "title": "Reduce Source Conflicts",
                "description": f"{conflicts_resolved} conflicts were resolved during synthesis",
                "actions": [
                    "Standardize data collection methods",
                    "Align agent outputs",
                    "Implement validation at source"
                ],
                "impact": "medium",
                "effort": "high"
            })
        
        # Coverage recommendations
        coverage_gaps = self._identify_coverage_gaps(synthesis_results)
        if coverage_gaps:
            recommendations.append({
                "type": "coverage_expansion",
                "priority": "low",
                "title": "Expand Coverage",
                "description": "Gaps identified in data coverage",
                "actions": [
                    f"Add coverage for: {', '.join(coverage_gaps)}",
                    "Deploy additional agents",
                    "Expand data sources"
                ],
                "impact": "medium",
                "effort": "medium"
            })
        
        # Priority-based ordering
        priority_order = {"high": 0, "medium": 1, "low": 2}
        recommendations.sort(key=lambda x: priority_order.get(x["priority"], 3))
        
        # Add implementation guidance
        for rec in recommendations:
            rec["implementation_guide"] = self._create_implementation_guide(rec)
        
        return recommendations
    
    async def create_unified_output(
        self,
        ctx: RunContext[SynthesisDependencies],
        synthesis_data: Dict[str, Any],
        output_format: str = "structured"
    ) -> Union[Dict[str, Any], str]:
        """Create unified output from synthesis
        
        Args:
            ctx: Run context
            synthesis_data: Synthesized data
            output_format: Output format
            
        Returns:
            Unified output
        """
        logfire.info("creating_unified_output", format=output_format)
        
        if output_format == "structured":
            return self._create_structured_output(synthesis_data)
        elif output_format == "narrative":
            return self._create_narrative_output(synthesis_data)
        elif output_format == "report":
            return self._create_report_output(synthesis_data)
        elif output_format == "json":
            return json.dumps(synthesis_data, indent=2, default=str)
        else:
            return synthesis_data
    
    def _merge_by_voting(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Merge results by voting on values"""
        merged = {}
        
        # Collect all keys
        all_keys = set()
        for result in results:
            if isinstance(result.get("data"), dict):
                all_keys.update(result["data"].keys())
        
        # Vote on each key
        for key in all_keys:
            votes = {}
            for result in results:
                if key in result.get("data", {}):
                    value = str(result["data"][key])
                    votes[value] = votes.get(value, 0) + 1
            
            # Select value with most votes
            if votes:
                winner = max(votes.items(), key=lambda x: x[1])
                # Try to convert back to original type
                try:
                    merged[key] = json.loads(winner[0])
                except:
                    merged[key] = winner[0]
        
        return merged
    
    async def _resolve_single_conflict(
        self,
        key: str,
        values: List[Any],
        sources: List[int],
        results: List[Dict[str, Any]]
    ) -> Any:
        """Resolve a single conflict"""
        # Strategy 1: Use most recent value
        timestamps = []
        for source in sources:
            ts = results[source].get("timestamp", "")
            timestamps.append((ts, values[sources.index(source)]))
        
        if timestamps and all(t[0] for t in timestamps):
            # Sort by timestamp and return most recent
            timestamps.sort(key=lambda x: x[0], reverse=True)
            return timestamps[0][1]
        
        # Strategy 2: Use value from highest confidence source
        confidences = []
        for i, source in enumerate(sources):
            conf = results[source].get("confidence", 0.5)
            confidences.append((conf, values[i]))
        
        if confidences:
            # Return value from highest confidence source
            confidences.sort(key=lambda x: x[0], reverse=True)
            return confidences[0][1]
        
        # Strategy 3: Use first non-null value
        for value in values:
            if value is not None:
                return value
        
        return None
    
    def _determine_synthesis_method(self, results: List[Dict[str, Any]]) -> str:
        """Determine the synthesis method used"""
        if len(results) == 1:
            return "single_source"
        elif len(results) == 2:
            return "dual_source_merge"
        elif all(r.get("agent_name", "").startswith("validator") for r in results):
            return "validation_consensus"
        elif any(r.get("agent_name", "").startswith("analyzer") for r in results):
            return "analytical_synthesis"
        else:
            return "multi_agent_synthesis"
    
    def _assess_result_quality(self, combined: Dict[str, Any]) -> float:
        """Assess quality of combined results"""
        score = 1.0
        
        # Reduce score for conflicts
        conflicts = combined["metadata"].get("conflicts_resolved", 0)
        if conflicts > 0:
            score -= min(0.3, conflicts * 0.05)
        
        # Reduce score for missing data
        if isinstance(combined.get("data"), dict):
            null_count = sum(1 for v in combined["data"].values() if v is None)
            if null_count > 0:
                score -= min(0.2, null_count * 0.02)
        
        # Boost score for multiple agreeing sources
        source_count = combined["metadata"].get("source_count", 1)
        if source_count > 3:
            score = min(1.0, score + 0.1)
        
        return max(0.0, score)
    
    async def _resolve_conflict(
        self,
        conflict: Dict[str, Any],
        ctx: RunContext[SynthesisDependencies]
    ) -> Dict[str, Any]:
        """Resolve a specific conflict"""
        resolution = {
            "conflict_id": conflict.get("id", "unknown"),
            "resolved": False,
            "method": None,
            "resolved_value": None,
            "confidence": 0.0
        }
        
        # Try different resolution strategies
        strategies = [
            ("timestamp", self._resolve_by_timestamp),
            ("confidence", self._resolve_by_confidence),
            ("consensus", self._resolve_by_consensus),
            ("rules", self._resolve_by_rules)
        ]
        
        for strategy_name, strategy_func in strategies:
            try:
                result = strategy_func(conflict)
                if result["success"]:
                    resolution["resolved"] = True
                    resolution["method"] = strategy_name
                    resolution["resolved_value"] = result["value"]
                    resolution["confidence"] = result["confidence"]
                    break
            except Exception as e:
                logfire.error(f"Resolution strategy {strategy_name} failed", error=str(e))
        
        return resolution
    
    def _resolve_by_timestamp(self, conflict: Dict[str, Any]) -> Dict[str, Any]:
        """Resolve by most recent timestamp"""
        if "timestamps" not in conflict:
            return {"success": False}
        
        # Find most recent
        latest_idx = max(
            range(len(conflict["timestamps"])),
            key=lambda i: conflict["timestamps"][i]
        )
        
        return {
            "success": True,
            "value": conflict["values"][latest_idx],
            "confidence": 0.8
        }
    
    def _resolve_by_confidence(self, conflict: Dict[str, Any]) -> Dict[str, Any]:
        """Resolve by highest confidence"""
        if "confidences" not in conflict:
            return {"success": False}
        
        # Find highest confidence
        best_idx = max(
            range(len(conflict["confidences"])),
            key=lambda i: conflict["confidences"][i]
        )
        
        return {
            "success": True,
            "value": conflict["values"][best_idx],
            "confidence": conflict["confidences"][best_idx]
        }
    
    def _resolve_by_consensus(self, conflict: Dict[str, Any]) -> Dict[str, Any]:
        """Resolve by consensus/voting"""
        values = conflict.get("values", [])
        if len(values) < 3:
            return {"success": False}
        
        # Count occurrences
        value_counts = {}
        for value in values:
            value_str = str(value)
            value_counts[value_str] = value_counts.get(value_str, 0) + 1
        
        # Find majority
        max_count = max(value_counts.values())
        if max_count > len(values) / 2:
            winner = [k for k, v in value_counts.items() if v == max_count][0]
            return {
                "success": True,
                "value": winner,
                "confidence": max_count / len(values)
            }
        
        return {"success": False}
    
    def _resolve_by_rules(self, conflict: Dict[str, Any]) -> Dict[str, Any]:
        """Resolve using predefined rules"""
        # This would use domain-specific rules
        # For now, return failure
        return {"success": False}
    
    def _create_resolution_summary(self, resolutions: Dict[str, Any]) -> str:
        """Create summary of resolutions"""
        total = resolutions["resolved_count"] + resolutions["failed_count"]
        if total == 0:
            return "No conflicts to resolve"
        
        success_rate = resolutions["resolved_count"] / total
        
        summary = f"Resolved {resolutions['resolved_count']} of {total} conflicts ({success_rate:.1%} success rate). "
        
        if resolutions["resolved_count"] > 0:
            methods = {}
            for res in resolutions["resolutions"]:
                method = res["method"]
                methods[method] = methods.get(method, 0) + 1
            
            summary += f"Methods used: {', '.join(f'{m} ({c})' for m, c in methods.items())}. "
        
        if resolutions["failed_count"] > 0:
            summary += f"{resolutions['failed_count']} conflicts require manual intervention."
        
        return summary
    
    def _suggest_interventions(self, unresolved: List[Dict[str, Any]]) -> List[str]:
        """Suggest manual interventions for unresolved conflicts"""
        suggestions = []
        
        # Analyze unresolved conflicts
        conflict_types = {}
        for conflict in unresolved:
            c_type = conflict.get("type", "unknown")
            conflict_types[c_type] = conflict_types.get(c_type, 0) + 1
        
        # Generate suggestions
        for c_type, count in conflict_types.items():
            if c_type == "value_mismatch":
                suggestions.append(f"Review {count} value mismatches and establish source of truth")
            elif c_type == "missing_data":
                suggestions.append(f"Gather missing data for {count} incomplete records")
            else:
                suggestions.append(f"Manually review {count} {c_type} conflicts")
        
        return suggestions
    
    def _create_executive_summary(self, data: Dict[str, Any]) -> str:
        """Create executive summary"""
        summary = "## Executive Summary\n\n"
        
        # Key metrics
        if "metrics" in data:
            summary += "**Key Metrics:**\n"
            for key, value in list(data["metrics"].items())[:5]:
                summary += f"- {key}: {value}\n"
            summary += "\n"
        
        # Main findings
        if "findings" in data:
            summary += "**Main Findings:**\n"
            for finding in data["findings"][:3]:
                summary += f"- {finding}\n"
            summary += "\n"
        
        # Recommendations
        if "recommendations" in data:
            summary += "**Top Recommendations:**\n"
            for rec in data["recommendations"][:3]:
                summary += f"- {rec.get('title', 'Recommendation')}: {rec.get('description', '')}\n"
        
        return summary
    
    def _create_technical_summary(self, data: Dict[str, Any]) -> str:
        """Create technical summary"""
        summary = "## Technical Summary\n\n"
        
        # Add technical details
        summary += f"**Data Sources:** {data.get('metadata', {}).get('source_count', 'N/A')}\n"
        summary += f"**Synthesis Method:** {data.get('metadata', {}).get('synthesis_method', 'N/A')}\n"
        summary += f"**Quality Score:** {data.get('metadata', {}).get('quality_score', 'N/A'):.2f}\n"
        summary += f"**Conflicts Resolved:** {data.get('metadata', {}).get('conflicts_resolved', 0)}\n\n"
        
        # Technical details
        if "technical_details" in data:
            summary += "**Technical Details:**\n"
            summary += json.dumps(data["technical_details"], indent=2)
        
        return summary
    
    def _create_detailed_summary(self, data: Dict[str, Any]) -> str:
        """Create detailed summary"""
        # This would create a comprehensive summary
        # For now, combine executive and technical
        return self._create_executive_summary(data) + "\n" + self._create_technical_summary(data)
    
    def _create_general_summary(self, data: Dict[str, Any]) -> str:
        """Create general summary"""
        summary = "## Summary\n\n"
        
        # Simple key-value summary
        for key, value in data.items():
            if key != "metadata" and not key.startswith("_"):
                if isinstance(value, (str, int, float, bool)):
                    summary += f"**{key}:** {value}\n"
                elif isinstance(value, list) and len(value) > 0:
                    summary += f"**{key}:** {len(value)} items\n"
                elif isinstance(value, dict):
                    summary += f"**{key}:** {len(value)} entries\n"
        
        return summary
    
    def _identify_temporal_patterns(self, data_points: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Identify temporal patterns"""
        patterns = []
        
        # Check if data has timestamps
        timestamps = []
        for point in data_points:
            if "timestamp" in point:
                timestamps.append(point["timestamp"])
        
        if len(timestamps) > 5:
            # Simple trend detection
            if all(timestamps[i] <= timestamps[i+1] for i in range(len(timestamps)-1)):
                patterns.append({
                    "type": "temporal",
                    "subtype": "monotonic_increase",
                    "description": "Timestamps show consistent progression"
                })
        
        return patterns
    
    def _identify_value_patterns(self, data_points: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Identify patterns in values"""
        patterns = []
        
        # Collect numeric values
        numeric_values = []
        for point in data_points:
            for key, value in point.items():
                if isinstance(value, (int, float)):
                    numeric_values.append(value)
        
        if numeric_values:
            avg = sum(numeric_values) / len(numeric_values)
            if all(abs(v - avg) < avg * 0.1 for v in numeric_values):
                patterns.append({
                    "type": "value",
                    "subtype": "clustering",
                    "description": f"Values cluster around {avg:.2f}"
                })
        
        return patterns
    
    def _identify_correlations(self, data_points: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Identify correlations between values"""
        # This would implement correlation analysis
        # For now, return empty
        return []
    
    def _identify_anomalies(self, data_points: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Identify anomalous data points"""
        anomalies = []
        
        # Simple anomaly detection based on missing fields
        if data_points:
            common_keys = set(data_points[0].keys())
            for i, point in enumerate(data_points[1:], 1):
                missing_keys = common_keys - set(point.keys())
                if missing_keys:
                    anomalies.append({
                        "index": i,
                        "type": "missing_fields",
                        "fields": list(missing_keys)
                    })
        
        return anomalies
    
    def _assess_anomaly_severity(self, anomalies: List[Dict[str, Any]]) -> str:
        """Assess severity of anomalies"""
        if not anomalies:
            return "none"
        elif len(anomalies) < 3:
            return "low"
        elif len(anomalies) < 10:
            return "medium"
        else:
            return "high"
    
    def _calculate_pattern_significance(self, pattern: Dict[str, Any]) -> float:
        """Calculate significance of a pattern"""
        # Base significance on pattern type
        type_scores = {
            "anomaly": 0.9,
            "correlation": 0.8,
            "temporal": 0.6,
            "value": 0.5
        }
        
        base_score = type_scores.get(pattern.get("type"), 0.5)
        
        # Adjust based on severity/strength
        if "severity" in pattern:
            severity_multipliers = {"high": 1.5, "medium": 1.0, "low": 0.5}
            base_score *= severity_multipliers.get(pattern["severity"], 1.0)
        
        return min(1.0, base_score)
    
    def _create_pattern_recommendation(self, pattern: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Create recommendation based on pattern"""
        if pattern["type"] == "anomaly" and pattern.get("severity") in ["medium", "high"]:
            return {
                "type": "investigation",
                "priority": "high" if pattern["severity"] == "high" else "medium",
                "title": "Investigate Anomalies",
                "description": f"Multiple anomalies detected: {pattern['description']}",
                "actions": [
                    "Review anomalous data points",
                    "Identify root cause",
                    "Implement validation to prevent future occurrences"
                ],
                "impact": "high",
                "effort": "medium"
            }
        
        elif pattern["type"] == "correlation":
            return {
                "type": "optimization",
                "priority": "medium",
                "title": "Leverage Correlations",
                "description": f"Strong correlations identified: {pattern['description']}",
                "actions": [
                    "Use correlations for prediction",
                    "Optimize based on relationships",
                    "Monitor correlation stability"
                ],
                "impact": "medium",
                "effort": "low"
            }
        
        return None
    
    def _identify_coverage_gaps(self, synthesis_results: Dict[str, Any]) -> List[str]:
        """Identify gaps in data coverage"""
        gaps = []
        
        # Check for common expected fields
        expected_fields = ["status", "results", "errors", "metrics", "metadata"]
        data = synthesis_results.get("data", {})
        
        for field in expected_fields:
            if field not in data:
                gaps.append(field)
        
        return gaps
    
    def _create_implementation_guide(self, recommendation: Dict[str, Any]) -> Dict[str, Any]:
        """Create implementation guide for recommendation"""
        guide = {
            "steps": [],
            "estimated_time": "",
            "required_resources": [],
            "success_criteria": []
        }
        
        # Create steps based on recommendation type
        if recommendation["type"] == "quality_improvement":
            guide["steps"] = [
                "Audit current data sources",
                "Identify quality issues",
                "Implement validation rules",
                "Monitor improvements"
            ]
            guide["estimated_time"] = "1-2 weeks"
            guide["required_resources"] = ["Data analyst", "Quality tools"]
            
        elif recommendation["type"] == "process_improvement":
            guide["steps"] = [
                "Document current process",
                "Identify improvement areas",
                "Design new process",
                "Implement gradually",
                "Measure impact"
            ]
            guide["estimated_time"] = "2-4 weeks"
            guide["required_resources"] = ["Process engineer", "Stakeholder time"]
        
        return guide
    
    def _create_structured_output(self, synthesis_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create structured output"""
        return {
            "synthesis_id": synthesis_data.get("id", "unknown"),
            "timestamp": datetime.utcnow().isoformat(),
            "summary": {
                "source_count": synthesis_data.get("metadata", {}).get("source_count", 0),
                "quality_score": synthesis_data.get("metadata", {}).get("quality_score", 0),
                "conflicts_resolved": synthesis_data.get("metadata", {}).get("conflicts_resolved", 0)
            },
            "data": synthesis_data.get("data", {}),
            "patterns": synthesis_data.get("patterns", []),
            "recommendations": synthesis_data.get("recommendations", []),
            "metadata": synthesis_data.get("metadata", {})
        }
    
    def _create_narrative_output(self, synthesis_data: Dict[str, Any]) -> str:
        """Create narrative output"""
        narrative = "Based on the synthesis of multiple data sources, "
        
        source_count = synthesis_data.get("metadata", {}).get("source_count", 0)
        narrative += f"we analyzed {source_count} different inputs. "
        
        conflicts = synthesis_data.get("metadata", {}).get("conflicts_resolved", 0)
        if conflicts > 0:
            narrative += f"We successfully resolved {conflicts} conflicts between sources. "
        
        patterns = synthesis_data.get("patterns", [])
        if patterns:
            narrative += f"We identified {len(patterns)} significant patterns, "
            narrative += f"with the most important being {patterns[0].get('description', 'various insights')}. "
        
        recommendations = synthesis_data.get("recommendations", [])
        if recommendations:
            narrative += f"Based on our analysis, we recommend {len(recommendations)} actions, "
            narrative += f"starting with {recommendations[0].get('title', 'key improvements')}."
        
        return narrative
    
    def _create_report_output(self, synthesis_data: Dict[str, Any]) -> str:
        """Create report format output"""
        report = "# Synthesis Report\n\n"
        report += f"**Generated:** {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}\n\n"
        
        # Executive Summary
        report += self._create_executive_summary(synthesis_data)
        report += "\n"
        
        # Detailed Findings
        report += "## Detailed Findings\n\n"
        data = synthesis_data.get("data", {})
        for key, value in data.items():
            report += f"### {key}\n"
            if isinstance(value, dict):
                for k, v in value.items():
                    report += f"- **{k}:** {v}\n"
            else:
                report += f"{value}\n"
            report += "\n"
        
        # Patterns
        patterns = synthesis_data.get("patterns", [])
        if patterns:
            report += "## Identified Patterns\n\n"
            for i, pattern in enumerate(patterns, 1):
                report += f"{i}. **{pattern.get('type', 'Unknown')}:** {pattern.get('description', '')}\n"
                report += f"   - Significance: {pattern.get('significance', 0):.2f}\n"
            report += "\n"
        
        # Recommendations
        recommendations = synthesis_data.get("recommendations", [])
        if recommendations:
            report += "## Recommendations\n\n"
            for i, rec in enumerate(recommendations, 1):
                report += f"{i}. **{rec.get('title', 'Recommendation')}**\n"
                report += f"   - Priority: {rec.get('priority', 'medium')}\n"
                report += f"   - Description: {rec.get('description', '')}\n"
                report += f"   - Impact: {rec.get('impact', 'medium')}\n"
                report += f"   - Effort: {rec.get('effort', 'medium')}\n"
                report += "\n"
        
        # Technical Details
        report += "## Technical Details\n\n"
        metadata = synthesis_data.get("metadata", {})
        report += f"- **Synthesis Method:** {metadata.get('synthesis_method', 'N/A')}\n"
        report += f"- **Quality Score:** {metadata.get('quality_score', 0):.2f}\n"
        report += f"- **Confidence Level:** {metadata.get('confidence', 0):.2f}\n"
        
        return report