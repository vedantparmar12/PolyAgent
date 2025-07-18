"""Validation pipeline for multi-stage validation"""

from typing import Dict, Any, List, Optional, Callable
from datetime import datetime
from enum import Enum
import asyncio
from .validation_gate import ValidationGate, ValidationResult, ValidationLevel
from .auto_validator import AutoValidator
from .knowledge_base import KnowledgeBase
import logfire


class PipelineStage(str, Enum):
    """Pipeline stage types"""
    PRE_VALIDATION = "pre_validation"
    SYNTAX_CHECK = "syntax_check"
    SECURITY_CHECK = "security_check"
    QUALITY_CHECK = "quality_check"
    KNOWLEDGE_CHECK = "knowledge_check"
    POST_VALIDATION = "post_validation"


class ValidationPipeline:
    """Multi-stage validation pipeline"""
    
    def __init__(
        self,
        validation_gate: ValidationGate,
        auto_validator: Optional[AutoValidator] = None,
        knowledge_base: Optional[KnowledgeBase] = None
    ):
        """Initialize validation pipeline
        
        Args:
            validation_gate: Validation gate instance
            auto_validator: Optional auto validator
            knowledge_base: Optional knowledge base
        """
        self.validation_gate = validation_gate
        self.auto_validator = auto_validator
        self.knowledge_base = knowledge_base
        self._logger = logfire.span("validation_pipeline")
        
        # Pipeline stages
        self._stages: Dict[PipelineStage, List[Callable]] = {
            PipelineStage.PRE_VALIDATION: [],
            PipelineStage.SYNTAX_CHECK: [],
            PipelineStage.SECURITY_CHECK: [],
            PipelineStage.QUALITY_CHECK: [],
            PipelineStage.KNOWLEDGE_CHECK: [],
            PipelineStage.POST_VALIDATION: []
        }
        
        # Stage configuration
        self._stage_config: Dict[PipelineStage, Dict[str, Any]] = {
            stage: {
                "enabled": True,
                "required": stage in [PipelineStage.SYNTAX_CHECK, PipelineStage.SECURITY_CHECK],
                "timeout": 30,
                "parallel": False
            }
            for stage in PipelineStage
        }
        
        # Initialize default stages
        self._initialize_default_stages()
    
    def _initialize_default_stages(self):
        """Initialize default validation stages"""
        # Pre-validation
        self.add_stage_handler(
            PipelineStage.PRE_VALIDATION,
            self._pre_validation_handler
        )
        
        # Syntax check
        self.add_stage_handler(
            PipelineStage.SYNTAX_CHECK,
            self._syntax_check_handler
        )
        
        # Security check
        self.add_stage_handler(
            PipelineStage.SECURITY_CHECK,
            self._security_check_handler
        )
        
        # Quality check
        self.add_stage_handler(
            PipelineStage.QUALITY_CHECK,
            self._quality_check_handler
        )
        
        # Knowledge check
        if self.knowledge_base:
            self.add_stage_handler(
                PipelineStage.KNOWLEDGE_CHECK,
                self._knowledge_check_handler
            )
        
        # Post-validation
        self.add_stage_handler(
            PipelineStage.POST_VALIDATION,
            self._post_validation_handler
        )
    
    def add_stage_handler(
        self,
        stage: PipelineStage,
        handler: Callable,
        priority: int = 0
    ) -> None:
        """Add handler to a pipeline stage
        
        Args:
            stage: Pipeline stage
            handler: Handler function
            priority: Handler priority (higher runs first)
        """
        if stage not in self._stages:
            raise ValueError(f"Invalid stage: {stage}")
        
        # Add with priority
        self._stages[stage].append((priority, handler))
        
        # Sort by priority
        self._stages[stage].sort(key=lambda x: x[0], reverse=True)
        
        self._logger.info(
            "Stage handler added",
            stage=stage,
            handler=handler.__name__,
            priority=priority
        )
    
    def configure_stage(
        self,
        stage: PipelineStage,
        config: Dict[str, Any]
    ) -> None:
        """Configure a pipeline stage
        
        Args:
            stage: Pipeline stage
            config: Stage configuration
        """
        if stage not in self._stage_config:
            raise ValueError(f"Invalid stage: {stage}")
        
        self._stage_config[stage].update(config)
        
        self._logger.info(
            "Stage configured",
            stage=stage,
            config=config
        )
    
    async def validate(
        self,
        data: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
        skip_stages: Optional[List[PipelineStage]] = None
    ) -> Dict[str, Any]:
        """Run validation pipeline
        
        Args:
            data: Data to validate
            context: Optional validation context
            skip_stages: Stages to skip
            
        Returns:
            Validation results
        """
        start_time = datetime.utcnow()
        skip_stages = skip_stages or []
        
        # Initialize results
        results = {
            "passed": True,
            "stages": {},
            "errors": [],
            "warnings": [],
            "info": [],
            "metadata": {
                "pipeline_version": "1.0.0",
                "timestamp": start_time.isoformat(),
                "context": context or {}
            }
        }
        
        # Run pipeline stages
        for stage in PipelineStage:
            if stage in skip_stages:
                continue
            
            stage_config = self._stage_config[stage]
            
            if not stage_config["enabled"]:
                continue
            
            # Run stage
            try:
                stage_result = await self._run_stage(
                    stage,
                    data,
                    context,
                    results
                )
                
                results["stages"][stage] = stage_result
                
                # Check if stage failed
                if not stage_result["passed"]:
                    if stage_config["required"]:
                        results["passed"] = False
                        break  # Stop pipeline on required stage failure
                    
            except asyncio.TimeoutError:
                error_result = {
                    "passed": False,
                    "error": f"Stage {stage} timed out",
                    "timeout": stage_config["timeout"]
                }
                results["stages"][stage] = error_result
                
                if stage_config["required"]:
                    results["passed"] = False
                    break
                    
            except Exception as e:
                self._logger.error(
                    "Stage execution failed",
                    stage=stage,
                    error=str(e)
                )
                
                error_result = {
                    "passed": False,
                    "error": str(e)
                }
                results["stages"][stage] = error_result
                
                if stage_config["required"]:
                    results["passed"] = False
                    break
        
        # Calculate execution time
        execution_time = (datetime.utcnow() - start_time).total_seconds()
        results["metadata"]["execution_time_seconds"] = execution_time
        
        # Generate summary
        results["summary"] = self._generate_summary(results)
        
        self._logger.info(
            "Pipeline validation complete",
            passed=results["passed"],
            stages_run=len(results["stages"]),
            execution_time=execution_time
        )
        
        return results
    
    async def _run_stage(
        self,
        stage: PipelineStage,
        data: Dict[str, Any],
        context: Optional[Dict[str, Any]],
        current_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Run a pipeline stage"""
        stage_config = self._stage_config[stage]
        handlers = self._stages[stage]
        
        if not handlers:
            return {
                "passed": True,
                "message": "No handlers configured"
            }
        
        stage_start = datetime.utcnow()
        
        # Prepare stage context
        stage_context = {
            "stage": stage,
            "data": data,
            "context": context,
            "previous_results": current_results,
            "config": stage_config
        }
        
        # Run handlers
        handler_results = []
        
        if stage_config.get("parallel", False):
            # Run handlers in parallel
            tasks = []
            for _, handler in handlers:
                task = asyncio.create_task(
                    asyncio.wait_for(
                        handler(stage_context),
                        timeout=stage_config["timeout"]
                    )
                )
                tasks.append(task)
            
            handler_results = await asyncio.gather(*tasks, return_exceptions=True)
            
        else:
            # Run handlers sequentially
            for _, handler in handlers:
                result = await asyncio.wait_for(
                    handler(stage_context),
                    timeout=stage_config["timeout"]
                )
                handler_results.append(result)
                
                # Stop on failure if required
                if isinstance(result, dict) and not result.get("passed", True):
                    if stage_config.get("fail_fast", True):
                        break
        
        # Aggregate results
        stage_result = self._aggregate_handler_results(handler_results)
        
        # Add execution time
        stage_result["execution_time_ms"] = (
            (datetime.utcnow() - stage_start).total_seconds() * 1000
        )
        
        return stage_result
    
    def _aggregate_handler_results(
        self,
        handler_results: List[Any]
    ) -> Dict[str, Any]:
        """Aggregate results from multiple handlers"""
        aggregated = {
            "passed": True,
            "handlers_run": len(handler_results),
            "handler_results": [],
            "errors": [],
            "warnings": [],
            "info": []
        }
        
        for result in handler_results:
            if isinstance(result, Exception):
                aggregated["passed"] = False
                aggregated["errors"].append(str(result))
                continue
            
            if isinstance(result, dict):
                aggregated["handler_results"].append(result)
                
                if not result.get("passed", True):
                    aggregated["passed"] = False
                
                # Collect messages
                if "errors" in result:
                    aggregated["errors"].extend(result["errors"])
                if "warnings" in result:
                    aggregated["warnings"].extend(result["warnings"])
                if "info" in result:
                    aggregated["info"].extend(result["info"])
        
        return aggregated
    
    async def _pre_validation_handler(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Default pre-validation handler"""
        data = context["data"]
        
        # Check data format
        if not isinstance(data, dict):
            return {
                "passed": False,
                "errors": ["Data must be a dictionary"]
            }
        
        # Check required fields
        if "type" not in data:
            return {
                "passed": False,
                "errors": ["Missing required field: type"]
            }
        
        return {
            "passed": True,
            "info": ["Pre-validation passed"]
        }
    
    async def _syntax_check_handler(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Default syntax check handler"""
        data = context["data"]
        
        # Use validation gate for syntax check
        results = await self.validation_gate.validate(
            data=data,
            tags=["syntax"]
        )
        
        return {
            "passed": results["passed"],
            "errors": [e["message"] for e in results.get("errors", [])],
            "warnings": [w["message"] for w in results.get("warnings", [])]
        }
    
    async def _security_check_handler(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Default security check handler"""
        data = context["data"]
        
        # Use validation gate for security check
        results = await self.validation_gate.validate(
            data=data,
            tags=["security"]
        )
        
        return {
            "passed": results["passed"],
            "errors": [e["message"] for e in results.get("errors", [])],
            "warnings": [w["message"] for w in results.get("warnings", [])]
        }
    
    async def _quality_check_handler(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Default quality check handler"""
        data = context["data"]
        
        # Use auto validator if available
        if self.auto_validator:
            results = await self.auto_validator.validate_auto(
                data=data,
                context=context.get("context")
            )
            
            return {
                "passed": results["passed"],
                "errors": [e["message"] for e in results.get("errors", [])],
                "warnings": [w["message"] for w in results.get("warnings", [])],
                "recommendations": results.get("recommendations", [])
            }
        else:
            # Use validation gate for quality check
            results = await self.validation_gate.validate(
                data=data,
                tags=["quality"]
            )
            
            return {
                "passed": results["passed"],
                "errors": [e["message"] for e in results.get("errors", [])],
                "warnings": [w["message"] for w in results.get("warnings", [])]
            }
    
    async def _knowledge_check_handler(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Default knowledge check handler"""
        if not self.knowledge_base:
            return {
                "passed": True,
                "info": ["Knowledge check skipped - no knowledge base"]
            }
        
        data = context["data"]
        
        # Check against knowledge base
        warnings = []
        info = []
        
        # Check for similar code/content
        if data.get("type") == "code" and "content" in data:
            similar = await self.knowledge_base.search(
                query=data["content"][:200],  # First 200 chars
                category="code_examples",
                limit=3,
                threshold=0.8
            )
            
            if similar:
                info.append(f"Found {len(similar)} similar examples in knowledge base")
                
                # Check if any similar code had issues
                for doc, score in similar:
                    if doc.metadata.get("has_issues", False):
                        warnings.append(
                            f"Similar code had issues: {doc.metadata.get('issue_description', 'Unknown')}"
                        )
        
        return {
            "passed": len(warnings) == 0,
            "warnings": warnings,
            "info": info
        }
    
    async def _post_validation_handler(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Default post-validation handler"""
        # Perform any cleanup or final checks
        return {
            "passed": True,
            "info": ["Post-validation complete"]
        }
    
    def _generate_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate validation summary"""
        summary = {
            "overall_passed": results["passed"],
            "stages_total": len(PipelineStage),
            "stages_run": len(results["stages"]),
            "stages_passed": sum(
                1 for r in results["stages"].values()
                if r.get("passed", False)
            ),
            "total_errors": sum(
                len(r.get("errors", []))
                for r in results["stages"].values()
            ),
            "total_warnings": sum(
                len(r.get("warnings", []))
                for r in results["stages"].values()
            ),
            "execution_time_seconds": results["metadata"].get("execution_time_seconds", 0)
        }
        
        # Identify failed stages
        failed_stages = [
            stage for stage, result in results["stages"].items()
            if not result.get("passed", True)
        ]
        
        if failed_stages:
            summary["failed_stages"] = failed_stages
            summary["first_failure"] = failed_stages[0]
        
        return summary
    
    def get_stage_status(self) -> Dict[str, Any]:
        """Get current pipeline stage status"""
        status = {}
        
        for stage in PipelineStage:
            config = self._stage_config[stage]
            handlers = self._stages[stage]
            
            status[stage] = {
                "enabled": config["enabled"],
                "required": config["required"],
                "timeout": config["timeout"],
                "parallel": config.get("parallel", False),
                "handler_count": len(handlers)
            }
        
        return status