"""Advisor agent for providing context and examples"""

from typing import List, Dict, Any
from pydantic_ai import RunContext
from .base_agent import BaseAgent
from .dependencies import AdvisorDependencies
from .models import ContextOutput
import logfire


class AdvisorAgent(BaseAgent[AdvisorDependencies, ContextOutput]):
    """Agent that provides relevant context and examples"""
    
    def __init__(self):
        """Initialize the advisor agent"""
        super().__init__(
            model='openai:gpt-4',
            deps_type=AdvisorDependencies,
            result_type=ContextOutput,
            enable_logfire=True
        )
    
    def get_system_prompt(self) -> str:
        """Get the system prompt for the advisor agent"""
        return """You are an expert advisor that provides relevant context, examples, and recommendations.
        
        Your responsibilities:
        1. Search for relevant code examples and patterns
        2. Provide contextual information to help with the task
        3. Make specific recommendations based on best practices
        4. Assess confidence in your recommendations
        
        Always:
        - Cite specific examples when available
        - Explain the relevance of each example
        - Provide actionable recommendations
        - Be honest about confidence levels
        
        Output structured information that helps other agents complete their tasks effectively."""
    
    def _register_tools(self):
        """Register tools for the advisor agent"""
        self.agent.tool(self.search_knowledge_base)
        self.agent.tool(self.find_similar_examples)
        self.agent.tool(self.analyze_context)
        self.agent.tool(self.get_best_practices)
    
    async def search_knowledge_base(
        self,
        ctx: RunContext[AdvisorDependencies],
        query: str,
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """Search the knowledge base for relevant information
        
        Args:
            ctx: Run context with dependencies
            query: Search query
            limit: Maximum number of results
            
        Returns:
            List of relevant documents
        """
        logfire.info("searching_knowledge_base", query=query, limit=limit)
        
        # If vector client is available, use it
        if ctx.deps.vector_client:
            try:
                # Perform vector search
                response = await ctx.deps.vector_client.rpc(
                    'match_documents',
                    {
                        'query_embedding': query,  # This would need to be an embedding
                        'match_count': limit
                    }
                ).execute()
                
                return response.data if response.data else []
            except Exception as e:
                logfire.error("vector_search_failed", error=str(e))
        
        # Fallback to file-based search
        return self._search_local_examples(ctx.deps.examples_path, query, limit)
    
    async def find_similar_examples(
        self,
        ctx: RunContext[AdvisorDependencies],
        code_snippet: str,
        language: str = "python"
    ) -> List[str]:
        """Find similar code examples
        
        Args:
            ctx: Run context
            code_snippet: Code to find similar examples for
            language: Programming language
            
        Returns:
            List of similar examples
        """
        logfire.info("finding_similar_examples", language=language)
        
        # Search for similar patterns
        examples = []
        
        # Extract key patterns from the code snippet
        patterns = self._extract_patterns(code_snippet)
        
        # Search for each pattern
        for pattern in patterns[:3]:  # Limit to top 3 patterns
            results = await self.search_knowledge_base(ctx, pattern, 2)
            for result in results:
                if 'content' in result:
                    examples.append(result['content'])
        
        return examples[:ctx.deps.context_limit]
    
    async def analyze_context(
        self,
        ctx: RunContext[AdvisorDependencies],
        task_description: str
    ) -> Dict[str, Any]:
        """Analyze the context of a task
        
        Args:
            ctx: Run context
            task_description: Description of the task
            
        Returns:
            Context analysis
        """
        logfire.info("analyzing_context", task=task_description)
        
        analysis = {
            "task_type": self._classify_task(task_description),
            "complexity": self._assess_complexity(task_description),
            "required_knowledge": self._identify_required_knowledge(task_description),
            "potential_challenges": self._identify_challenges(task_description)
        }
        
        return analysis
    
    async def get_best_practices(
        self,
        ctx: RunContext[AdvisorDependencies],
        topic: str
    ) -> List[str]:
        """Get best practices for a topic
        
        Args:
            ctx: Run context
            topic: Topic to get best practices for
            
        Returns:
            List of best practices
        """
        logfire.info("getting_best_practices", topic=topic)
        
        # Define common best practices
        practices_db = {
            "error_handling": [
                "Always use specific exception types",
                "Provide meaningful error messages",
                "Log errors with appropriate context",
                "Implement retry logic for transient failures"
            ],
            "code_structure": [
                "Follow single responsibility principle",
                "Keep functions small and focused",
                "Use descriptive variable names",
                "Add type hints for better code clarity"
            ],
            "testing": [
                "Write tests before implementation (TDD)",
                "Aim for high test coverage (>80%)",
                "Test edge cases and error conditions",
                "Use mocks for external dependencies"
            ],
            "security": [
                "Never hardcode sensitive information",
                "Validate all user inputs",
                "Use parameterized queries for databases",
                "Implement proper authentication and authorization"
            ]
        }
        
        # Find relevant practices
        relevant_practices = []
        topic_lower = topic.lower()
        
        for category, practices in practices_db.items():
            if category in topic_lower or topic_lower in category:
                relevant_practices.extend(practices)
        
        # If no specific match, provide general practices
        if not relevant_practices:
            relevant_practices = [
                "Write clean, readable code",
                "Document your code thoroughly",
                "Follow established coding standards",
                "Consider performance implications"
            ]
        
        return relevant_practices
    
    def _search_local_examples(
        self,
        examples_path: str,
        query: str,
        limit: int
    ) -> List[Dict[str, Any]]:
        """Search local examples directory
        
        Args:
            examples_path: Path to examples
            query: Search query
            limit: Maximum results
            
        Returns:
            List of examples
        """
        import os
        from pathlib import Path
        
        examples = []
        examples_dir = Path(examples_path)
        
        if not examples_dir.exists():
            return examples
        
        # Simple file-based search
        query_lower = query.lower()
        
        for file_path in examples_dir.rglob("*.py"):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                if query_lower in content.lower():
                    examples.append({
                        "file": str(file_path),
                        "content": content[:500],  # First 500 chars
                        "relevance": content.lower().count(query_lower)
                    })
                    
                if len(examples) >= limit:
                    break
                    
            except Exception as e:
                logfire.error("file_read_error", file=str(file_path), error=str(e))
        
        # Sort by relevance
        examples.sort(key=lambda x: x['relevance'], reverse=True)
        
        return examples[:limit]
    
    def _extract_patterns(self, code_snippet: str) -> List[str]:
        """Extract key patterns from code
        
        Args:
            code_snippet: Code to analyze
            
        Returns:
            List of patterns
        """
        patterns = []
        
        # Extract function/class names
        import re
        
        # Function definitions
        func_pattern = r'def\s+(\w+)\s*\('
        functions = re.findall(func_pattern, code_snippet)
        patterns.extend(functions)
        
        # Class definitions
        class_pattern = r'class\s+(\w+)\s*[\(:]'
        classes = re.findall(class_pattern, code_snippet)
        patterns.extend(classes)
        
        # Import statements
        import_pattern = r'(?:from|import)\s+([\w.]+)'
        imports = re.findall(import_pattern, code_snippet)
        patterns.extend(imports)
        
        return patterns
    
    def _classify_task(self, task_description: str) -> str:
        """Classify the type of task
        
        Args:
            task_description: Task description
            
        Returns:
            Task classification
        """
        task_lower = task_description.lower()
        
        if any(word in task_lower for word in ['implement', 'create', 'build', 'develop']):
            return "implementation"
        elif any(word in task_lower for word in ['fix', 'debug', 'resolve', 'troubleshoot']):
            return "debugging"
        elif any(word in task_lower for word in ['refactor', 'optimize', 'improve', 'enhance']):
            return "refactoring"
        elif any(word in task_lower for word in ['test', 'validate', 'verify', 'check']):
            return "testing"
        elif any(word in task_lower for word in ['document', 'explain', 'describe', 'comment']):
            return "documentation"
        else:
            return "general"
    
    def _assess_complexity(self, task_description: str) -> str:
        """Assess task complexity
        
        Args:
            task_description: Task description
            
        Returns:
            Complexity level
        """
        # Simple heuristic based on keywords and length
        complex_keywords = [
            'system', 'architecture', 'integration', 'multiple',
            'concurrent', 'distributed', 'scalable', 'production'
        ]
        
        task_lower = task_description.lower()
        complexity_score = 0
        
        # Check for complex keywords
        for keyword in complex_keywords:
            if keyword in task_lower:
                complexity_score += 1
        
        # Check task length
        if len(task_description) > 200:
            complexity_score += 1
        
        # Determine complexity
        if complexity_score >= 3:
            return "high"
        elif complexity_score >= 1:
            return "medium"
        else:
            return "low"
    
    def _identify_required_knowledge(self, task_description: str) -> List[str]:
        """Identify required knowledge areas
        
        Args:
            task_description: Task description
            
        Returns:
            List of knowledge areas
        """
        knowledge_map = {
            "api": ["REST", "HTTP", "endpoints", "requests"],
            "database": ["sql", "query", "database", "table", "schema"],
            "frontend": ["ui", "react", "vue", "angular", "css", "html"],
            "backend": ["server", "api", "endpoint", "service", "microservice"],
            "testing": ["test", "pytest", "unittest", "mock", "assertion"],
            "devops": ["docker", "kubernetes", "deploy", "ci/cd", "pipeline"],
            "security": ["auth", "secure", "encrypt", "permission", "access"],
            "performance": ["optimize", "performance", "speed", "cache", "scale"]
        }
        
        task_lower = task_description.lower()
        required = []
        
        for area, keywords in knowledge_map.items():
            if any(keyword in task_lower for keyword in keywords):
                required.append(area)
        
        return required or ["general"]
    
    def _identify_challenges(self, task_description: str) -> List[str]:
        """Identify potential challenges
        
        Args:
            task_description: Task description
            
        Returns:
            List of potential challenges
        """
        challenges = []
        task_lower = task_description.lower()
        
        # Common challenge patterns
        if "legacy" in task_lower or "existing" in task_lower:
            challenges.append("Working with existing codebase")
        
        if "performance" in task_lower or "optimize" in task_lower:
            challenges.append("Performance optimization required")
        
        if "scale" in task_lower or "concurrent" in task_lower:
            challenges.append("Scalability considerations")
        
        if "integrate" in task_lower or "third-party" in task_lower:
            challenges.append("Third-party integration complexity")
        
        if "secure" in task_lower or "auth" in task_lower:
            challenges.append("Security implementation")
        
        return challenges or ["Standard implementation challenges"]