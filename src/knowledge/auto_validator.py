"""Automated validation system with learning capabilities"""

from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from collections import defaultdict
import json
from pathlib import Path
from .validation_gate import ValidationGate, ValidationRule, ValidationLevel, ValidationResult
from .knowledge_base import KnowledgeBase
import logfire
import asyncio


class ValidationPattern:
    """Pattern learned from validation history"""
    
    def __init__(self, pattern_type: str, condition: Dict[str, Any], outcome: str):
        self.pattern_type = pattern_type
        self.condition = condition
        self.outcome = outcome
        self.confidence = 0.5
        self.occurrences = 1
        self.last_seen = datetime.utcnow()
    
    def update(self, success: bool):
        """Update pattern based on new occurrence"""
        self.occurrences += 1
        self.last_seen = datetime.utcnow()
        
        # Update confidence using exponential moving average
        alpha = 0.1  # Learning rate
        self.confidence = alpha * (1.0 if success else 0.0) + (1 - alpha) * self.confidence
    
    def matches(self, data: Dict[str, Any]) -> bool:
        """Check if data matches this pattern"""
        for key, value in self.condition.items():
            if key not in data:
                return False
            
            if isinstance(value, dict) and "$regex" in value:
                # Regex matching
                import re
                if not re.search(value["$regex"], str(data[key])):
                    return False
            elif isinstance(value, dict) and "$contains" in value:
                # Contains matching
                if value["$contains"] not in str(data[key]):
                    return False
            elif data[key] != value:
                return False
        
        return True


class AutoValidator:
    """Automated validation system with pattern learning"""
    
    def __init__(
        self,
        validation_gate: ValidationGate,
        knowledge_base: Optional[KnowledgeBase] = None
    ):
        """Initialize auto validator
        
        Args:
            validation_gate: Validation gate instance
            knowledge_base: Optional knowledge base for learning
        """
        self.validation_gate = validation_gate
        self.knowledge_base = knowledge_base
        self._logger = logfire.span("auto_validator")
        
        # Validation history
        self._history: List[Dict[str, Any]] = []
        self._patterns: List[ValidationPattern] = []
        
        # Statistics
        self._stats = defaultdict(lambda: {
            "total": 0,
            "passed": 0,
            "failed": 0,
            "errors": 0,
            "warnings": 0
        })
        
        # Configuration
        self.learning_enabled = True
        self.pattern_threshold = 0.7  # Confidence threshold for applying patterns
        self.history_limit = 1000
        self.pattern_limit = 100
    
    async def validate_auto(
        self,
        data: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Automatically validate data with learning
        
        Args:
            data: Data to validate
            context: Optional context information
            
        Returns:
            Validation results with recommendations
        """
        start_time = datetime.utcnow()
        
        # Prepare validation context
        validation_context = {
            "data": data,
            "context": context or {},
            "timestamp": start_time
        }
        
        # Get recommendations from patterns
        recommendations = self._get_pattern_recommendations(data)
        
        # Determine which rules to run
        tags = self._determine_validation_tags(data, recommendations)
        exclude_rules = self._determine_excluded_rules(data, recommendations)
        
        # Run validation
        results = await self.validation_gate.validate(
            data=data,
            tags=tags,
            exclude_rules=exclude_rules
        )
        
        # Add recommendations
        results["recommendations"] = recommendations
        
        # Learn from results if enabled
        if self.learning_enabled:
            await self._learn_from_validation(validation_context, results)
        
        # Update statistics
        self._update_statistics(data.get("type", "unknown"), results)
        
        # Add metadata
        results["auto_validation"] = {
            "patterns_applied": len(recommendations),
            "execution_time_ms": (datetime.utcnow() - start_time).total_seconds() * 1000,
            "confidence": self._calculate_validation_confidence(results)
        }
        
        self._logger.info(
            "Auto validation complete",
            data_type=data.get("type"),
            passed=results["passed"],
            patterns_applied=len(recommendations)
        )
        
        return results
    
    async def train_from_history(
        self,
        history_file: Optional[Path] = None
    ) -> Dict[str, Any]:
        """Train validator from historical data
        
        Args:
            history_file: Optional file containing historical validations
            
        Returns:
            Training results
        """
        self._logger.info("Training from history")
        
        # Load history if file provided
        if history_file and history_file.exists():
            with open(history_file, 'r') as f:
                historical_data = json.load(f)
                self._history.extend(historical_data.get("validations", []))
        
        # Extract patterns
        patterns_found = 0
        
        # Group validations by outcome
        passed_validations = []
        failed_validations = []
        
        for validation in self._history:
            if validation["results"]["passed"]:
                passed_validations.append(validation)
            else:
                failed_validations.append(validation)
        
        # Find patterns in failures
        failure_patterns = self._extract_patterns(failed_validations, "failure")
        patterns_found += len(failure_patterns)
        self._patterns.extend(failure_patterns)
        
        # Find patterns in successes
        success_patterns = self._extract_patterns(passed_validations, "success")
        patterns_found += len(success_patterns)
        self._patterns.extend(success_patterns)
        
        # Prune low-confidence patterns
        self._prune_patterns()
        
        # Store patterns in knowledge base if available
        if self.knowledge_base:
            await self._store_patterns_in_kb()
        
        results = {
            "patterns_found": patterns_found,
            "patterns_kept": len(self._patterns),
            "history_size": len(self._history),
            "training_complete": True
        }
        
        self._logger.info("Training complete", **results)
        
        return results
    
    def _get_pattern_recommendations(
        self,
        data: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Get recommendations based on learned patterns"""
        recommendations = []
        
        for pattern in self._patterns:
            if pattern.confidence >= self.pattern_threshold and pattern.matches(data):
                recommendation = {
                    "pattern_type": pattern.pattern_type,
                    "confidence": pattern.confidence,
                    "outcome": pattern.outcome,
                    "action": self._get_pattern_action(pattern)
                }
                recommendations.append(recommendation)
        
        # Sort by confidence
        recommendations.sort(key=lambda x: x["confidence"], reverse=True)
        
        return recommendations
    
    def _get_pattern_action(self, pattern: ValidationPattern) -> Dict[str, Any]:
        """Get recommended action for a pattern"""
        if pattern.outcome == "failure":
            return {
                "type": "prevent",
                "message": f"Similar data has failed validation {pattern.occurrences} times",
                "suggestions": self._get_failure_suggestions(pattern)
            }
        else:
            return {
                "type": "optimize",
                "message": f"Similar data typically passes validation",
                "suggestions": self._get_success_suggestions(pattern)
            }
    
    def _get_failure_suggestions(self, pattern: ValidationPattern) -> List[str]:
        """Get suggestions for failure patterns"""
        suggestions = []
        
        if "syntax" in pattern.pattern_type:
            suggestions.append("Review code syntax before submission")
            suggestions.append("Use a linter or syntax checker")
        
        if "security" in pattern.pattern_type:
            suggestions.append("Review security best practices")
            suggestions.append("Scan for common vulnerabilities")
        
        if "completeness" in pattern.pattern_type:
            suggestions.append("Ensure all required fields are present")
            suggestions.append("Check for TODO markers")
        
        return suggestions
    
    def _get_success_suggestions(self, pattern: ValidationPattern) -> List[str]:
        """Get suggestions for success patterns"""
        suggestions = []
        
        if pattern.confidence > 0.9:
            suggestions.append("Consider skipping some validation checks for efficiency")
        
        suggestions.append("Continue following current patterns")
        
        return suggestions
    
    def _determine_validation_tags(
        self,
        data: Dict[str, Any],
        recommendations: List[Dict[str, Any]]
    ) -> Optional[List[str]]:
        """Determine which validation tags to use"""
        data_type = data.get("type", "unknown")
        
        # Base tags
        tags = []
        
        if data_type == "code":
            tags.extend(["code", "syntax", "security"])
        elif data_type == "documentation":
            tags.extend(["documentation", "format"])
        elif data_type == "test":
            tags.extend(["test", "quality"])
        
        # Add tags from recommendations
        for rec in recommendations:
            if rec["outcome"] == "failure" and rec["confidence"] > 0.8:
                # Focus on problem areas
                if "security" in rec["pattern_type"]:
                    tags.append("security")
                if "performance" in rec["pattern_type"]:
                    tags.append("performance")
        
        return tags if tags else None
    
    def _determine_excluded_rules(
        self,
        data: Dict[str, Any],
        recommendations: List[Dict[str, Any]]
    ) -> Optional[List[str]]:
        """Determine which rules to exclude"""
        excluded = []
        
        # Skip rules that consistently pass for this type of data
        for rec in recommendations:
            if rec["outcome"] == "success" and rec["confidence"] > 0.95:
                if rec["pattern_type"] == "always_passes_docstring":
                    excluded.append("docstring_check")
                elif rec["pattern_type"] == "always_passes_complexity":
                    excluded.append("complexity_check")
        
        return excluded if excluded else None
    
    async def _learn_from_validation(
        self,
        context: Dict[str, Any],
        results: Dict[str, Any]
    ) -> None:
        """Learn from validation results"""
        # Add to history
        validation_record = {
            "timestamp": context["timestamp"].isoformat(),
            "data_type": context["data"].get("type", "unknown"),
            "context": context["context"],
            "results": results
        }
        
        self._history.append(validation_record)
        
        # Limit history size
        if len(self._history) > self.history_limit:
            self._history = self._history[-self.history_limit:]
        
        # Update existing patterns
        data = context["data"]
        for pattern in self._patterns:
            if pattern.matches(data):
                pattern.update(results["passed"])
        
        # Look for new patterns if this was unexpected
        if self._is_unexpected_result(data, results):
            new_patterns = self._extract_patterns([validation_record], 
                                                 "failure" if not results["passed"] else "success")
            self._patterns.extend(new_patterns)
            self._prune_patterns()
    
    def _is_unexpected_result(
        self,
        data: Dict[str, Any],
        results: Dict[str, Any]
    ) -> bool:
        """Check if result was unexpected based on patterns"""
        predictions = self._get_pattern_recommendations(data)
        
        if not predictions:
            return True
        
        # Check if outcome matches predictions
        expected_outcome = predictions[0]["outcome"]
        actual_outcome = "success" if results["passed"] else "failure"
        
        return expected_outcome != actual_outcome
    
    def _extract_patterns(
        self,
        validations: List[Dict[str, Any]],
        outcome: str
    ) -> List[ValidationPattern]:
        """Extract patterns from validations"""
        patterns = []
        
        # Group by data type
        by_type = defaultdict(list)
        for validation in validations:
            data_type = validation.get("data_type", "unknown")
            by_type[data_type].append(validation)
        
        # Extract patterns for each type
        for data_type, type_validations in by_type.items():
            if len(type_validations) >= 3:  # Need minimum occurrences
                # Find common attributes
                common_attrs = self._find_common_attributes(type_validations)
                
                if common_attrs:
                    pattern = ValidationPattern(
                        pattern_type=f"{data_type}_{outcome}",
                        condition=common_attrs,
                        outcome=outcome
                    )
                    patterns.append(pattern)
        
        return patterns
    
    def _find_common_attributes(
        self,
        validations: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Find common attributes in validations"""
        if not validations:
            return {}
        
        # Simple implementation: find exact matches
        # In production, this would use more sophisticated pattern matching
        common = {}
        
        # Check for common data attributes
        first_data = validations[0].get("context", {}).get("data", {})
        
        for key, value in first_data.items():
            if all(v.get("context", {}).get("data", {}).get(key) == value 
                   for v in validations[1:]):
                common[key] = value
        
        return common
    
    def _prune_patterns(self):
        """Remove low-confidence or old patterns"""
        cutoff_date = datetime.utcnow() - timedelta(days=30)
        
        self._patterns = [
            p for p in self._patterns
            if p.confidence >= 0.3 and p.last_seen > cutoff_date
        ]
        
        # Limit number of patterns
        if len(self._patterns) > self.pattern_limit:
            # Keep highest confidence patterns
            self._patterns.sort(key=lambda p: p.confidence, reverse=True)
            self._patterns = self._patterns[:self.pattern_limit]
    
    async def _store_patterns_in_kb(self):
        """Store patterns in knowledge base"""
        if not self.knowledge_base:
            return
        
        for pattern in self._patterns:
            content = f"""Validation Pattern: {pattern.pattern_type}
Condition: {json.dumps(pattern.condition)}
Outcome: {pattern.outcome}
Confidence: {pattern.confidence:.2f}
Occurrences: {pattern.occurrences}
Last seen: {pattern.last_seen.isoformat()}
"""
            
            await self.knowledge_base.add_knowledge(
                content=content,
                category="validation_patterns",
                tags=["pattern", pattern.outcome, pattern.pattern_type],
                metadata={
                    "pattern_type": pattern.pattern_type,
                    "confidence": pattern.confidence,
                    "outcome": pattern.outcome
                }
            )
    
    def _update_statistics(self, data_type: str, results: Dict[str, Any]):
        """Update validation statistics"""
        stats = self._stats[data_type]
        stats["total"] += 1
        
        if results["passed"]:
            stats["passed"] += 1
        else:
            stats["failed"] += 1
        
        stats["errors"] += len(results.get("errors", []))
        stats["warnings"] += len(results.get("warnings", []))
    
    def _calculate_validation_confidence(self, results: Dict[str, Any]) -> float:
        """Calculate confidence in validation results"""
        if not results["passed"]:
            # High confidence in failures
            return 0.9
        
        # Calculate based on warnings and patterns
        confidence = 1.0
        
        # Reduce for warnings
        warning_count = len(results.get("warnings", []))
        confidence -= warning_count * 0.05
        
        # Boost for pattern matches
        pattern_count = len(results.get("recommendations", []))
        confidence += pattern_count * 0.02
        
        return max(0.0, min(1.0, confidence))
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get validation statistics"""
        total_stats = {
            "total": sum(s["total"] for s in self._stats.values()),
            "passed": sum(s["passed"] for s in self._stats.values()),
            "failed": sum(s["failed"] for s in self._stats.values()),
            "errors": sum(s["errors"] for s in self._stats.values()),
            "warnings": sum(s["warnings"] for s in self._stats.values())
        }
        
        return {
            "overall": total_stats,
            "by_type": dict(self._stats),
            "patterns": {
                "total": len(self._patterns),
                "high_confidence": len([p for p in self._patterns if p.confidence > 0.8]),
                "active": len([p for p in self._patterns 
                             if (datetime.utcnow() - p.last_seen).days < 7])
            },
            "history_size": len(self._history)
        }
    
    def export_patterns(self, file_path: Path) -> None:
        """Export learned patterns to file"""
        patterns_data = []
        
        for pattern in self._patterns:
            patterns_data.append({
                "pattern_type": pattern.pattern_type,
                "condition": pattern.condition,
                "outcome": pattern.outcome,
                "confidence": pattern.confidence,
                "occurrences": pattern.occurrences,
                "last_seen": pattern.last_seen.isoformat()
            })
        
        export_data = {
            "exported_at": datetime.utcnow().isoformat(),
            "patterns": patterns_data,
            "statistics": self.get_statistics()
        }
        
        with open(file_path, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        self._logger.info("Patterns exported", file=str(file_path), count=len(patterns_data))
    
    def import_patterns(self, file_path: Path) -> None:
        """Import patterns from file"""
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        imported = 0
        for pattern_data in data.get("patterns", []):
            pattern = ValidationPattern(
                pattern_type=pattern_data["pattern_type"],
                condition=pattern_data["condition"],
                outcome=pattern_data["outcome"]
            )
            pattern.confidence = pattern_data["confidence"]
            pattern.occurrences = pattern_data["occurrences"]
            pattern.last_seen = datetime.fromisoformat(pattern_data["last_seen"])
            
            self._patterns.append(pattern)
            imported += 1
        
        self._prune_patterns()
        
        self._logger.info("Patterns imported", file=str(file_path), count=imported)