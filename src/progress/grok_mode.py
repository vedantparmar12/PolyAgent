"""Grok Heavy Mode implementation for deep understanding and analysis"""

from typing import Dict, Any, List, Optional, Tuple, Set
from datetime import datetime
from pydantic import BaseModel, Field
from enum import Enum
import asyncio
from collections import defaultdict
import networkx as nx
from ..core.base_agent import BaseAgent
from ..knowledge.knowledge_base import KnowledgeBase
from ..knowledge.validation_gate import ValidationGate
from .progress_tracker import ProgressTracker
import logfire


class GrokLevel(str, Enum):
    """Grok analysis levels"""
    SURFACE = "surface"      # Basic understanding
    STRUCTURAL = "structural"  # Code structure analysis
    SEMANTIC = "semantic"     # Meaning and intent
    CONTEXTUAL = "contextual" # Full context understanding
    DEEP = "deep"            # Deep insights and patterns


class GrokContext(BaseModel):
    """Context for Grok analysis"""
    target: str = Field(description="Analysis target (file, function, module)")
    level: GrokLevel = Field(default=GrokLevel.STRUCTURAL)
    include_dependencies: bool = Field(default=True)
    include_patterns: bool = Field(default=True)
    include_suggestions: bool = Field(default=True)
    max_depth: int = Field(default=3, description="Maximum analysis depth")
    focus_areas: List[str] = Field(default_factory=list, description="Specific areas to focus on")


class CodeEntity(BaseModel):
    """Represents a code entity (class, function, module)"""
    name: str
    type: str  # module, class, function, variable
    file_path: str
    line_start: int
    line_end: int
    docstring: Optional[str] = None
    signature: Optional[str] = None
    dependencies: List[str] = Field(default_factory=list)
    metrics: Dict[str, Any] = Field(default_factory=dict)


class GrokInsight(BaseModel):
    """Insight from Grok analysis"""
    type: str  # pattern, issue, optimization, suggestion
    severity: str  # info, warning, critical
    title: str
    description: str
    location: Optional[str] = None
    evidence: List[str] = Field(default_factory=list)
    suggestions: List[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)


class GrokResult(BaseModel):
    """Result of Grok analysis"""
    context: GrokContext
    entities: List[CodeEntity] = Field(default_factory=list)
    insights: List[GrokInsight] = Field(default_factory=list)
    patterns: Dict[str, List[str]] = Field(default_factory=dict)
    dependencies: Dict[str, List[str]] = Field(default_factory=dict)
    metrics: Dict[str, Any] = Field(default_factory=dict)
    summary: str = ""
    analysis_time: float = 0.0
    confidence_score: float = 0.0


class GrokHeavyMode:
    """Grok Heavy Mode for deep code understanding"""
    
    def __init__(
        self,
        agent: BaseAgent,
        knowledge_base: Optional[KnowledgeBase] = None,
        validation_gate: Optional[ValidationGate] = None,
        progress_tracker: Optional[ProgressTracker] = None
    ):
        """Initialize Grok Heavy Mode
        
        Args:
            agent: Base agent for analysis
            knowledge_base: Optional knowledge base
            validation_gate: Optional validation gate
            progress_tracker: Optional progress tracker
        """
        self.agent = agent
        self.knowledge_base = knowledge_base
        self.validation_gate = validation_gate
        self.progress_tracker = progress_tracker
        self._logger = logfire.span("grok_heavy_mode")
        
        # Analysis cache
        self._cache: Dict[str, GrokResult] = {}
        
        # Code graph for dependency analysis
        self._code_graph = nx.DiGraph()
    
    async def analyze(
        self,
        context: GrokContext,
        code_content: Optional[str] = None
    ) -> GrokResult:
        """Perform Grok analysis
        
        Args:
            context: Analysis context
            code_content: Optional code content (if not file-based)
            
        Returns:
            Grok analysis result
        """
        start_time = datetime.utcnow()
        
        # Check cache
        cache_key = f"{context.target}:{context.level}"
        if cache_key in self._cache:
            cached = self._cache[cache_key]
            if (datetime.utcnow() - datetime.fromisoformat(
                cached.metrics.get("timestamp", "1970-01-01")
            )).seconds < 300:  # 5 minute cache
                self._logger.info("Using cached analysis", target=context.target)
                return cached
        
        # Create progress task if tracker available
        task_id = None
        if self.progress_tracker:
            task = await self.progress_tracker.create_task(
                task_id=f"grok_{context.target}",
                name=f"Grok Analysis: {context.target}",
                metadata={"context": context.dict()}
            )
            task_id = task.task_id
        
        try:
            # Initialize result
            result = GrokResult(context=context)
            
            # Level-based analysis
            if context.level >= GrokLevel.SURFACE:
                await self._analyze_surface(context, code_content, result, task_id)
            
            if context.level >= GrokLevel.STRUCTURAL:
                await self._analyze_structure(context, code_content, result, task_id)
            
            if context.level >= GrokLevel.SEMANTIC:
                await self._analyze_semantics(context, code_content, result, task_id)
            
            if context.level >= GrokLevel.CONTEXTUAL:
                await self._analyze_context(context, code_content, result, task_id)
            
            if context.level >= GrokLevel.DEEP:
                await self._analyze_deep(context, code_content, result, task_id)
            
            # Generate summary
            result.summary = await self._generate_summary(result)
            
            # Calculate confidence
            result.confidence_score = self._calculate_confidence(result)
            
            # Record metrics
            result.analysis_time = (datetime.utcnow() - start_time).total_seconds()
            result.metrics["timestamp"] = datetime.utcnow().isoformat()
            
            # Cache result
            self._cache[cache_key] = result
            
            # Complete progress task
            if task_id and self.progress_tracker:
                await self.progress_tracker.complete_task(
                    task_id,
                    f"Grok analysis complete: {len(result.insights)} insights found",
                    {"insights_count": len(result.insights)}
                )
            
            self._logger.info(
                "Grok analysis complete",
                target=context.target,
                level=context.level,
                insights=len(result.insights),
                time=result.analysis_time
            )
            
            return result
            
        except Exception as e:
            # Fail progress task
            if task_id and self.progress_tracker:
                await self.progress_tracker.fail_task(
                    task_id,
                    str(e)
                )
            
            self._logger.error(f"Grok analysis failed: {e}")
            raise
    
    async def _analyze_surface(
        self,
        context: GrokContext,
        code_content: Optional[str],
        result: GrokResult,
        task_id: Optional[str]
    ) -> None:
        """Surface level analysis - basic structure and syntax"""
        if task_id and self.progress_tracker:
            await self.progress_tracker.update_progress(
                task_id, 10, "Analyzing surface structure..."
            )
        
        # Parse code to identify entities
        entities = await self._parse_code_entities(context.target, code_content)
        result.entities.extend(entities)
        
        # Basic metrics
        result.metrics["total_entities"] = len(entities)
        result.metrics["entity_types"] = defaultdict(int)
        
        for entity in entities:
            result.metrics["entity_types"][entity.type] += 1
            
            # Check for basic issues
            if entity.type == "function" and not entity.docstring:
                result.insights.append(GrokInsight(
                    type="issue",
                    severity="warning",
                    title="Missing docstring",
                    description=f"Function '{entity.name}' lacks documentation",
                    location=f"{entity.file_path}:{entity.line_start}",
                    suggestions=["Add a docstring explaining the function's purpose"],
                    confidence=1.0
                ))
    
    async def _analyze_structure(
        self,
        context: GrokContext,
        code_content: Optional[str],
        result: GrokResult,
        task_id: Optional[str]
    ) -> None:
        """Structural analysis - dependencies and relationships"""
        if task_id and self.progress_tracker:
            await self.progress_tracker.update_progress(
                task_id, 30, "Analyzing code structure..."
            )
        
        # Build dependency graph
        for entity in result.entities:
            self._code_graph.add_node(entity.name, entity=entity)
            
            for dep in entity.dependencies:
                self._code_graph.add_edge(entity.name, dep)
        
        # Analyze graph structure
        if self._code_graph.number_of_nodes() > 0:
            # Detect circular dependencies
            cycles = list(nx.simple_cycles(self._code_graph))
            for cycle in cycles:
                result.insights.append(GrokInsight(
                    type="issue",
                    severity="critical",
                    title="Circular dependency detected",
                    description=f"Circular dependency: {' -> '.join(cycle + [cycle[0]])}",
                    evidence=cycle,
                    suggestions=["Refactor to break the circular dependency"],
                    confidence=1.0
                ))
            
            # Find highly connected nodes
            centrality = nx.degree_centrality(self._code_graph)
            high_centrality = {
                node: score for node, score in centrality.items()
                if score > 0.5
            }
            
            for node, score in high_centrality.items():
                result.insights.append(GrokInsight(
                    type="pattern",
                    severity="info",
                    title="High coupling detected",
                    description=f"'{node}' has high coupling (score: {score:.2f})",
                    suggestions=["Consider refactoring to reduce dependencies"],
                    confidence=0.8
                ))
        
        # Store dependencies
        result.dependencies = {
            entity.name: entity.dependencies
            for entity in result.entities
        }
    
    async def _analyze_semantics(
        self,
        context: GrokContext,
        code_content: Optional[str],
        result: GrokResult,
        task_id: Optional[str]
    ) -> None:
        """Semantic analysis - meaning and intent"""
        if task_id and self.progress_tracker:
            await self.progress_tracker.update_progress(
                task_id, 50, "Analyzing semantics and patterns..."
            )
        
        # Use agent to understand code intent
        if code_content:
            prompt = f"""Analyze the semantic meaning and intent of this code:

{code_content[:2000]}  # Truncate for context

Focus on:
1. What is the main purpose?
2. What patterns are used?
3. What are potential issues?
4. What improvements could be made?

Provide specific insights."""

            analysis = await self.agent.think(prompt)
            
            # Parse agent insights
            # This is simplified - in production, use structured output
            if "purpose" in analysis.lower():
                result.summary = analysis.split('\n')[0]
            
            # Extract patterns from agent analysis
            if context.include_patterns:
                patterns = await self._extract_patterns(code_content, analysis)
                result.patterns.update(patterns)
        
        # Validate with knowledge base if available
        if self.knowledge_base and context.include_patterns:
            # Search for similar code patterns
            similar = await self.knowledge_base.search(
                query=code_content[:500] if code_content else context.target,
                category="code_patterns",
                limit=5
            )
            
            for doc, score in similar:
                if score > 0.8:
                    result.insights.append(GrokInsight(
                        type="pattern",
                        severity="info",
                        title="Similar pattern found",
                        description=f"Found similar pattern: {doc.metadata.get('pattern_name', 'Unknown')}",
                        evidence=[doc.content[:200]],
                        confidence=score
                    ))
    
    async def _analyze_context(
        self,
        context: GrokContext,
        code_content: Optional[str],
        result: GrokResult,
        task_id: Optional[str]
    ) -> None:
        """Contextual analysis - broader system understanding"""
        if task_id and self.progress_tracker:
            await self.progress_tracker.update_progress(
                task_id, 70, "Analyzing context and relationships..."
            )
        
        # Analyze in broader context
        if context.include_dependencies:
            # Find related entities
            related_entities = set()
            for entity in result.entities:
                # Get neighbors in dependency graph
                if entity.name in self._code_graph:
                    neighbors = list(self._code_graph.neighbors(entity.name))
                    related_entities.update(neighbors)
            
            # Analyze impact
            if related_entities:
                result.metrics["impact_scope"] = len(related_entities)
                
                if len(related_entities) > 10:
                    result.insights.append(GrokInsight(
                        type="pattern",
                        severity="warning",
                        title="High impact scope",
                        description=f"Changes here could affect {len(related_entities)} other components",
                        evidence=list(related_entities)[:5],
                        suggestions=["Consider the broad impact of changes"],
                        confidence=0.7
                    ))
        
        # Check validation rules if available
        if self.validation_gate:
            validation_data = {
                "type": "code",
                "content": code_content or "",
                "language": "python",
                "context": context.dict()
            }
            
            validation_result = await self.validation_gate.validate(
                data=validation_data,
                tags=["code", "quality"]
            )
            
            # Convert validation issues to insights
            for error in validation_result.get("errors", []):
                result.insights.append(GrokInsight(
                    type="issue",
                    severity="critical",
                    title=error["rule_name"],
                    description=error["message"],
                    confidence=0.9
                ))
    
    async def _analyze_deep(
        self,
        context: GrokContext,
        code_content: Optional[str],
        result: GrokResult,
        task_id: Optional[str]
    ) -> None:
        """Deep analysis - advanced patterns and insights"""
        if task_id and self.progress_tracker:
            await self.progress_tracker.update_progress(
                task_id, 90, "Performing deep analysis..."
            )
        
        # Complex pattern detection
        patterns_found = defaultdict(list)
        
        # Design pattern detection
        design_patterns = await self._detect_design_patterns(result.entities)
        patterns_found["design_patterns"] = design_patterns
        
        # Anti-pattern detection
        anti_patterns = await self._detect_anti_patterns(result.entities, code_content)
        patterns_found["anti_patterns"] = anti_patterns
        
        # Code smell detection
        code_smells = await self._detect_code_smells(result.entities, code_content)
        patterns_found["code_smells"] = code_smells
        
        # Add to results
        for pattern_type, patterns in patterns_found.items():
            if patterns:
                result.patterns[pattern_type] = patterns
                
                # Create insights for significant findings
                for pattern in patterns[:3]:  # Top 3
                    severity = "critical" if pattern_type == "anti_patterns" else "warning"
                    result.insights.append(GrokInsight(
                        type="pattern",
                        severity=severity,
                        title=f"{pattern_type.replace('_', ' ').title()} detected",
                        description=pattern,
                        suggestions=self._get_pattern_suggestions(pattern_type, pattern),
                        confidence=0.8
                    ))
        
        # Performance analysis
        if context.focus_areas and "performance" in context.focus_areas:
            perf_insights = await self._analyze_performance(code_content)
            result.insights.extend(perf_insights)
        
        # Security analysis
        if context.focus_areas and "security" in context.focus_areas:
            sec_insights = await self._analyze_security(code_content)
            result.insights.extend(sec_insights)
    
    async def _parse_code_entities(
        self,
        target: str,
        code_content: Optional[str]
    ) -> List[CodeEntity]:
        """Parse code to extract entities"""
        entities = []
        
        # Simplified parsing - in production, use proper AST parsing
        if code_content:
            lines = code_content.split('\n')
            
            for i, line in enumerate(lines):
                # Detect functions
                if line.strip().startswith("def "):
                    name = line.split("def ")[1].split("(")[0]
                    entities.append(CodeEntity(
                        name=name,
                        type="function",
                        file_path=target,
                        line_start=i + 1,
                        line_end=i + 10,  # Simplified
                        signature=line.strip()
                    ))
                
                # Detect classes
                elif line.strip().startswith("class "):
                    name = line.split("class ")[1].split("(")[0].split(":")[0]
                    entities.append(CodeEntity(
                        name=name,
                        type="class",
                        file_path=target,
                        line_start=i + 1,
                        line_end=i + 20,  # Simplified
                        signature=line.strip()
                    ))
        
        return entities
    
    async def _extract_patterns(
        self,
        code_content: str,
        analysis: str
    ) -> Dict[str, List[str]]:
        """Extract patterns from code and analysis"""
        patterns = defaultdict(list)
        
        # Simple pattern matching - enhance with ML in production
        pattern_keywords = {
            "singleton": ["singleton", "instance", "_instance"],
            "factory": ["factory", "create", "build"],
            "observer": ["observer", "subscribe", "notify"],
            "decorator": ["decorator", "@", "wrapper"],
            "strategy": ["strategy", "algorithm", "policy"]
        }
        
        for pattern_name, keywords in pattern_keywords.items():
            if any(kw in code_content.lower() for kw in keywords):
                patterns["design_patterns"].append(pattern_name)
        
        return dict(patterns)
    
    async def _detect_design_patterns(
        self,
        entities: List[CodeEntity]
    ) -> List[str]:
        """Detect design patterns in code"""
        patterns = []
        
        # Simplified detection logic
        class_names = [e.name for e in entities if e.type == "class"]
        function_names = [e.name for e in entities if e.type == "function"]
        
        # Singleton pattern
        if any("instance" in name.lower() for name in function_names):
            patterns.append("Singleton pattern")
        
        # Factory pattern
        if any("create" in name.lower() or "build" in name.lower() for name in function_names):
            patterns.append("Factory pattern")
        
        # Observer pattern
        if any("observer" in name.lower() or "listener" in name.lower() for name in class_names):
            patterns.append("Observer pattern")
        
        return patterns
    
    async def _detect_anti_patterns(
        self,
        entities: List[CodeEntity],
        code_content: Optional[str]
    ) -> List[str]:
        """Detect anti-patterns in code"""
        anti_patterns = []
        
        # God class detection
        large_classes = [e for e in entities if e.type == "class" and (e.line_end - e.line_start) > 200]
        if large_classes:
            anti_patterns.append(f"God class: {large_classes[0].name}")
        
        # Long method detection
        long_methods = [e for e in entities if e.type == "function" and (e.line_end - e.line_start) > 50]
        if long_methods:
            anti_patterns.append(f"Long method: {long_methods[0].name}")
        
        return anti_patterns
    
    async def _detect_code_smells(
        self,
        entities: List[CodeEntity],
        code_content: Optional[str]
    ) -> List[str]:
        """Detect code smells"""
        smells = []
        
        if code_content:
            # Duplicate code detection (simplified)
            lines = code_content.split('\n')
            line_counts = defaultdict(int)
            
            for line in lines:
                stripped = line.strip()
                if len(stripped) > 20:  # Significant lines only
                    line_counts[stripped] += 1
            
            duplicates = [line for line, count in line_counts.items() if count > 2]
            if duplicates:
                smells.append("Duplicate code detected")
            
            # Magic numbers
            import re
            magic_numbers = re.findall(r'\b\d{2,}\b', code_content)
            if len(magic_numbers) > 5:
                smells.append("Magic numbers detected")
        
        return smells
    
    async def _analyze_performance(self, code_content: Optional[str]) -> List[GrokInsight]:
        """Analyze performance aspects"""
        insights = []
        
        if code_content:
            # Look for performance issues
            perf_patterns = {
                r'for .+ in .+:\s*for .+ in': "Nested loops detected",
                r'\.append\(.+\) in .+ loop': "List append in loop - consider list comprehension",
                r'time\.sleep': "Blocking sleep detected - consider async",
                r'\*\*': "Expensive operation (power) detected"
            }
            
            import re
            for pattern, message in perf_patterns.items():
                if re.search(pattern, code_content):
                    insights.append(GrokInsight(
                        type="optimization",
                        severity="warning",
                        title="Performance concern",
                        description=message,
                        suggestions=["Review for optimization opportunities"],
                        confidence=0.7
                    ))
        
        return insights
    
    async def _analyze_security(self, code_content: Optional[str]) -> List[GrokInsight]:
        """Analyze security aspects"""
        insights = []
        
        if code_content:
            # Look for security issues
            sec_patterns = {
                r'eval\(': "eval() usage - security risk",
                r'exec\(': "exec() usage - security risk",
                r'pickle\.loads': "Pickle deserialization - security risk",
                r'password\s*=\s*["\']': "Hardcoded password detected",
                r'SECRET|KEY|TOKEN': "Possible hardcoded secrets"
            }
            
            import re
            for pattern, message in sec_patterns.items():
                if re.search(pattern, code_content, re.IGNORECASE):
                    insights.append(GrokInsight(
                        type="issue",
                        severity="critical",
                        title="Security concern",
                        description=message,
                        suggestions=["Review and fix security issue"],
                        confidence=0.9
                    ))
        
        return insights
    
    def _get_pattern_suggestions(self, pattern_type: str, pattern: str) -> List[str]:
        """Get suggestions for detected patterns"""
        suggestions_map = {
            "anti_patterns": {
                "God class": ["Break down into smaller, focused classes", "Apply Single Responsibility Principle"],
                "Long method": ["Extract smaller methods", "Reduce cyclomatic complexity"]
            },
            "code_smells": {
                "Duplicate code": ["Extract common functionality", "Use DRY principle"],
                "Magic numbers": ["Replace with named constants", "Use configuration"]
            }
        }
        
        return suggestions_map.get(pattern_type, {}).get(pattern, ["Review and refactor"])
    
    async def _generate_summary(self, result: GrokResult) -> str:
        """Generate analysis summary"""
        parts = []
        
        # Entity summary
        if result.entities:
            entity_summary = ", ".join(
                f"{count} {etype}s"
                for etype, count in result.metrics.get("entity_types", {}).items()
            )
            parts.append(f"Found {entity_summary}")
        
        # Insight summary
        if result.insights:
            critical = len([i for i in result.insights if i.severity == "critical"])
            warnings = len([i for i in result.insights if i.severity == "warning"])
            
            if critical:
                parts.append(f"{critical} critical issues")
            if warnings:
                parts.append(f"{warnings} warnings")
        
        # Pattern summary
        if result.patterns:
            pattern_count = sum(len(p) for p in result.patterns.values())
            parts.append(f"{pattern_count} patterns detected")
        
        return ". ".join(parts) if parts else "Analysis complete"
    
    def _calculate_confidence(self, result: GrokResult) -> float:
        """Calculate overall confidence score"""
        if not result.insights:
            return 1.0
        
        # Average confidence of insights
        confidences = [i.confidence for i in result.insights]
        return sum(confidences) / len(confidences)
    
    async def explain_insight(
        self,
        insight: GrokInsight,
        verbose: bool = False
    ) -> str:
        """Explain an insight in detail
        
        Args:
            insight: Insight to explain
            verbose: Include verbose explanation
            
        Returns:
            Detailed explanation
        """
        explanation = f"""**{insight.title}**

**Type:** {insight.type}
**Severity:** {insight.severity}
**Confidence:** {insight.confidence:.0%}

**Description:**
{insight.description}
"""
        
        if insight.location:
            explanation += f"\n**Location:** {insight.location}"
        
        if insight.evidence:
            explanation += f"\n\n**Evidence:**"
            for evidence in insight.evidence[:3]:
                explanation += f"\n- {evidence}"
        
        if insight.suggestions:
            explanation += f"\n\n**Suggestions:**"
            for suggestion in insight.suggestions:
                explanation += f"\n- {suggestion}"
        
        if verbose and self.agent:
            # Get detailed explanation from agent
            detailed = await self.agent.think(
                f"Explain this code insight in detail: {insight.title}\n"
                f"Context: {insight.description}"
            )
            explanation += f"\n\n**Detailed Explanation:**\n{detailed}"
        
        return explanation
    
    def get_insights_by_severity(
        self,
        result: GrokResult,
        min_severity: str = "info"
    ) -> List[GrokInsight]:
        """Get insights filtered by severity
        
        Args:
            result: Grok result
            min_severity: Minimum severity level
            
        Returns:
            Filtered insights
        """
        severity_order = ["info", "warning", "critical"]
        min_index = severity_order.index(min_severity)
        
        return [
            insight for insight in result.insights
            if severity_order.index(insight.severity) >= min_index
        ]
    
    def clear_cache(self) -> None:
        """Clear analysis cache"""
        self._cache.clear()
        self._code_graph.clear()
        self._logger.info("Cache cleared")