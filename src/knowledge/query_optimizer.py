"""Query optimizer for knowledge base searches"""

from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
import re
from collections import defaultdict
import logfire


class QueryOptimizer:
    """Optimizer for knowledge base queries"""
    
    def __init__(self):
        """Initialize query optimizer"""
        self._logger = logfire.span("query_optimizer")
        
        # Query cache
        self._query_cache: Dict[str, Tuple[str, datetime]] = {}
        self._cache_ttl = timedelta(hours=1)
        
        # Query statistics
        self._query_stats = defaultdict(lambda: {
            "count": 0,
            "total_time_ms": 0,
            "avg_results": 0,
            "last_used": None
        })
        
        # Synonym dictionary
        self._synonyms = self._load_synonyms()
        
        # Stop words
        self._stop_words = self._load_stop_words()
        
        # Query templates
        self._query_templates = self._load_query_templates()
    
    def optimize_query(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Optimize a search query
        
        Args:
            query: Original query
            context: Optional query context
            
        Returns:
            Optimized query
        """
        # Check cache
        cache_key = f"{query}:{str(context)}"
        if cache_key in self._query_cache:
            cached_query, cached_time = self._query_cache[cache_key]
            if datetime.utcnow() - cached_time < self._cache_ttl:
                self._logger.info("Query cache hit", query=query[:50])
                return cached_query
        
        # Apply optimizations
        optimized = query
        
        # 1. Clean and normalize
        optimized = self._normalize_query(optimized)
        
        # 2. Expand with synonyms
        optimized = self._expand_synonyms(optimized, context)
        
        # 3. Remove stop words (unless quoted)
        optimized = self._remove_stop_words(optimized)
        
        # 4. Apply query templates
        optimized = self._apply_templates(optimized, context)
        
        # 5. Add context boost terms
        optimized = self._add_context_terms(optimized, context)
        
        # 6. Optimize for vector search
        optimized = self._optimize_for_vectors(optimized)
        
        # Cache result
        self._query_cache[cache_key] = (optimized, datetime.utcnow())
        
        # Update statistics
        self._update_stats(query, optimized)
        
        self._logger.info(
            "Query optimized",
            original=query[:50],
            optimized=optimized[:50]
        )
        
        return optimized
    
    def _normalize_query(self, query: str) -> str:
        """Normalize query text"""
        # Convert to lowercase (preserve quoted sections)
        parts = re.split(r'("[^"]*")', query)
        normalized_parts = []
        
        for i, part in enumerate(parts):
            if i % 2 == 0:  # Not quoted
                part = part.lower().strip()
                # Remove extra whitespace
                part = re.sub(r'\s+', ' ', part)
            normalized_parts.append(part)
        
        normalized = ''.join(normalized_parts)
        
        # Remove special characters (except in quotes)
        normalized = re.sub(r'[^\w\s"]+', ' ', normalized)
        
        return normalized.strip()
    
    def _expand_synonyms(self, query: str, context: Optional[Dict[str, Any]]) -> str:
        """Expand query with synonyms"""
        # Get context-specific synonyms
        domain = context.get("domain", "general") if context else "general"
        synonyms = self._synonyms.get(domain, {})
        
        # Split query preserving quoted sections
        parts = re.split(r'("[^"]*")', query)
        expanded_parts = []
        
        for i, part in enumerate(parts):
            if i % 2 == 0:  # Not quoted
                words = part.split()
                expanded_words = []
                
                for word in words:
                    # Check for synonyms
                    if word in synonyms:
                        # Add original and synonyms
                        expanded_words.append(f"({word} OR {' OR '.join(synonyms[word])})")
                    else:
                        expanded_words.append(word)
                
                expanded_parts.append(' '.join(expanded_words))
            else:
                expanded_parts.append(part)
        
        return ''.join(expanded_parts)
    
    def _remove_stop_words(self, query: str) -> str:
        """Remove stop words from query"""
        # Split query preserving quoted sections
        parts = re.split(r'("[^"]*")', query)
        cleaned_parts = []
        
        for i, part in enumerate(parts):
            if i % 2 == 0:  # Not quoted
                words = part.split()
                # Keep words that are not stop words or are in specific positions
                kept_words = []
                for j, word in enumerate(words):
                    # Keep if not a stop word, or if it's important position
                    if (word not in self._stop_words or 
                        j == 0 or  # First word
                        j == len(words) - 1 or  # Last word
                        word.isupper()):  # Acronym
                        kept_words.append(word)
                
                cleaned_parts.append(' '.join(kept_words))
            else:
                cleaned_parts.append(part)
        
        return ''.join(cleaned_parts)
    
    def _apply_templates(self, query: str, context: Optional[Dict[str, Any]]) -> str:
        """Apply query templates based on patterns"""
        if not context:
            return query
        
        query_type = self._detect_query_type(query, context)
        
        if query_type in self._query_templates:
            template = self._query_templates[query_type]
            
            # Extract key terms from query
            key_terms = self._extract_key_terms(query)
            
            # Apply template
            templated_query = template.format(
                terms=' '.join(key_terms),
                category=context.get("category", ""),
                language=context.get("language", "python")
            )
            
            return templated_query
        
        return query
    
    def _add_context_terms(self, query: str, context: Optional[Dict[str, Any]]) -> str:
        """Add context-specific boost terms"""
        if not context:
            return query
        
        boost_terms = []
        
        # Add category boost
        if "category" in context:
            boost_terms.append(f"+category:{context['category']}")
        
        # Add language boost
        if "language" in context:
            boost_terms.append(f"+language:{context['language']}")
        
        # Add recency boost
        if context.get("prefer_recent", False):
            boost_terms.append("+recent:true")
        
        # Add quality boost
        if context.get("high_quality_only", False):
            boost_terms.append("+quality:high")
        
        if boost_terms:
            return f"{query} {' '.join(boost_terms)}"
        
        return query
    
    def _optimize_for_vectors(self, query: str) -> str:
        """Optimize query for vector similarity search"""
        # Remove boost terms for vector search
        vector_query = re.sub(r'\+\w+:\w+', '', query).strip()
        
        # Expand important terms
        important_terms = self._identify_important_terms(vector_query)
        
        if important_terms:
            # Repeat important terms for emphasis
            emphasized = vector_query
            for term in important_terms[:3]:  # Top 3 terms
                emphasized += f" {term}"
            
            return emphasized.strip()
        
        return vector_query
    
    def _detect_query_type(self, query: str, context: Dict[str, Any]) -> str:
        """Detect the type of query"""
        query_lower = query.lower()
        
        # Code search patterns
        if any(pattern in query_lower for pattern in ["function", "class", "method", "implement"]):
            return "code_search"
        
        # Error search patterns
        if any(pattern in query_lower for pattern in ["error", "exception", "bug", "issue"]):
            return "error_search"
        
        # How-to patterns
        if any(pattern in query_lower for pattern in ["how to", "how do", "example of", "tutorial"]):
            return "howto_search"
        
        # API search patterns
        if any(pattern in query_lower for pattern in ["api", "endpoint", "route", "request"]):
            return "api_search"
        
        # Documentation search
        if any(pattern in query_lower for pattern in ["docs", "documentation", "reference"]):
            return "docs_search"
        
        return "general"
    
    def _extract_key_terms(self, query: str) -> List[str]:
        """Extract key terms from query"""
        # Remove common words and boost terms
        cleaned = re.sub(r'\+\w+:\w+', '', query)
        cleaned = re.sub(r'\b(OR|AND|NOT)\b', '', cleaned)
        
        # Extract quoted terms first
        quoted_terms = re.findall(r'"([^"]+)"', cleaned)
        
        # Remove quoted sections
        for term in quoted_terms:
            cleaned = cleaned.replace(f'"{term}"', '')
        
        # Extract remaining words
        words = cleaned.split()
        
        # Filter and rank
        key_terms = []
        
        # Add quoted terms (highest priority)
        key_terms.extend(quoted_terms)
        
        # Add capitalized words (likely proper nouns)
        key_terms.extend([w for w in words if w[0].isupper()])
        
        # Add technical terms
        technical_patterns = [
            r'^\w+_\w+$',  # snake_case
            r'^\w+[A-Z]\w+$',  # camelCase
            r'^\w+\.\w+$',  # dotted notation
        ]
        
        for word in words:
            if any(re.match(pattern, word) for pattern in technical_patterns):
                if word not in key_terms:
                    key_terms.append(word)
        
        # Add remaining non-stop words
        for word in words:
            if (word not in self._stop_words and 
                word not in key_terms and 
                len(word) > 2):
                key_terms.append(word)
        
        return key_terms[:10]  # Limit to top 10 terms
    
    def _identify_important_terms(self, query: str) -> List[str]:
        """Identify the most important terms in a query"""
        words = query.split()
        
        # Score each word
        word_scores = {}
        
        for word in words:
            score = 0
            
            # Length score
            score += min(len(word) / 10, 1.0)
            
            # Capitalization score
            if word[0].isupper():
                score += 0.5
            
            # Technical term score
            if '_' in word or '.' in word:
                score += 0.5
            
            # Position score (earlier is better)
            position = words.index(word)
            score += (len(words) - position) / len(words)
            
            word_scores[word] = score
        
        # Sort by score
        sorted_words = sorted(word_scores.items(), key=lambda x: x[1], reverse=True)
        
        return [word for word, _ in sorted_words]
    
    def _update_stats(self, original: str, optimized: str):
        """Update query statistics"""
        stats = self._query_stats[original]
        stats["count"] += 1
        stats["last_used"] = datetime.utcnow()
        
        # Clean old stats
        if len(self._query_stats) > 1000:
            # Remove least recently used
            sorted_queries = sorted(
                self._query_stats.items(),
                key=lambda x: x[1]["last_used"] or datetime.min
            )
            for query, _ in sorted_queries[:100]:
                del self._query_stats[query]
    
    def _load_synonyms(self) -> Dict[str, Dict[str, List[str]]]:
        """Load synonym dictionary"""
        return {
            "general": {
                "create": ["make", "build", "construct", "generate"],
                "delete": ["remove", "destroy", "drop", "erase"],
                "update": ["modify", "change", "edit", "alter"],
                "find": ["search", "locate", "discover", "get"],
                "error": ["exception", "bug", "issue", "problem"],
                "fast": ["quick", "rapid", "speedy", "efficient"],
                "big": ["large", "huge", "massive", "enormous"]
            },
            "code": {
                "function": ["method", "procedure", "routine"],
                "variable": ["var", "attribute", "property"],
                "class": ["object", "type", "model"],
                "import": ["include", "require", "use"],
                "return": ["yield", "output", "result"]
            },
            "api": {
                "endpoint": ["route", "path", "url"],
                "request": ["call", "query", "fetch"],
                "response": ["reply", "answer", "result"],
                "auth": ["authentication", "authorization", "login"]
            }
        }
    
    def _load_stop_words(self) -> set:
        """Load stop words"""
        return {
            "a", "an", "and", "are", "as", "at", "be", "by", "for",
            "from", "has", "he", "in", "is", "it", "its", "of", "on",
            "that", "the", "to", "was", "will", "with", "the", "this",
            "but", "they", "have", "had", "what", "when", "where", "who",
            "which", "why", "how", "all", "would", "there", "their"
        }
    
    def _load_query_templates(self) -> Dict[str, str]:
        """Load query templates"""
        return {
            "code_search": "{terms} implementation code example {language}",
            "error_search": "{terms} error exception fix solution debug",
            "howto_search": "tutorial example {terms} step guide {category}",
            "api_search": "{terms} api endpoint documentation request response",
            "docs_search": "{terms} documentation reference guide manual"
        }
    
    def get_query_suggestions(
        self,
        partial_query: str,
        limit: int = 5
    ) -> List[str]:
        """Get query suggestions based on partial input"""
        suggestions = []
        
        # Check recent queries
        recent_queries = sorted(
            [(q, s["last_used"]) for q, s in self._query_stats.items()
             if partial_query.lower() in q.lower()],
            key=lambda x: x[1] or datetime.min,
            reverse=True
        )
        
        for query, _ in recent_queries[:limit]:
            suggestions.append(query)
        
        # Add template-based suggestions
        if len(suggestions) < limit:
            query_type = self._detect_query_type(partial_query, {})
            if query_type in self._query_templates:
                template = self._query_templates[query_type]
                suggestion = template.format(
                    terms=partial_query,
                    category="",
                    language="python"
                ).strip()
                suggestions.append(suggestion)
        
        return suggestions[:limit]
    
    def analyze_query_performance(
        self,
        query: str,
        results_count: int,
        execution_time_ms: float
    ) -> Dict[str, Any]:
        """Analyze query performance"""
        stats = self._query_stats[query]
        
        # Update statistics
        stats["total_time_ms"] += execution_time_ms
        stats["avg_results"] = (
            (stats["avg_results"] * (stats["count"] - 1) + results_count) / 
            stats["count"]
        )
        
        # Analyze performance
        analysis = {
            "query": query,
            "execution_time_ms": execution_time_ms,
            "results_count": results_count,
            "historical_avg_time": stats["total_time_ms"] / stats["count"],
            "historical_avg_results": stats["avg_results"],
            "performance_rating": self._calculate_performance_rating(
                execution_time_ms,
                results_count,
                stats
            )
        }
        
        # Provide optimization suggestions
        if analysis["performance_rating"] < 0.5:
            analysis["suggestions"] = self._get_optimization_suggestions(
                query,
                results_count,
                execution_time_ms
            )
        
        return analysis
    
    def _calculate_performance_rating(
        self,
        execution_time: float,
        results_count: int,
        stats: Dict[str, Any]
    ) -> float:
        """Calculate query performance rating (0-1)"""
        rating = 1.0
        
        # Time factor (target: < 100ms)
        if execution_time > 100:
            rating -= min((execution_time - 100) / 1000, 0.5)
        
        # Results factor (sweet spot: 5-20 results)
        if results_count == 0:
            rating -= 0.3
        elif results_count > 50:
            rating -= 0.2
        elif results_count < 3:
            rating -= 0.1
        
        # Historical comparison
        if stats["count"] > 5:
            avg_time = stats["total_time_ms"] / stats["count"]
            if execution_time > avg_time * 1.5:
                rating -= 0.1
        
        return max(0.0, rating)
    
    def _get_optimization_suggestions(
        self,
        query: str,
        results_count: int,
        execution_time: float
    ) -> List[str]:
        """Get query optimization suggestions"""
        suggestions = []
        
        if results_count == 0:
            suggestions.append("Try broadening your search terms")
            suggestions.append("Check for typos or use synonyms")
            suggestions.append("Remove restrictive filters")
        
        elif results_count > 50:
            suggestions.append("Add more specific terms to narrow results")
            suggestions.append("Use filters to refine your search")
            suggestions.append("Try exact phrase matching with quotes")
        
        if execution_time > 500:
            suggestions.append("Consider using more specific terms")
            suggestions.append("Limit search to specific categories")
        
        if len(query.split()) > 10:
            suggestions.append("Try using fewer, more relevant terms")
        
        return suggestions