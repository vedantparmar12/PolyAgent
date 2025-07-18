"""Knowledge base for storing and retrieving contextual information"""

from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
from .vector_store import VectorStore, Document
import logfire
import json
from pathlib import Path


class KnowledgeBase:
    """Knowledge base with semantic search and context management"""
    
    def __init__(self, vector_store: VectorStore):
        """Initialize knowledge base
        
        Args:
            vector_store: Vector store instance
        """
        self.vector_store = vector_store
        self._logger = logfire.span("knowledge_base")
        self._cache: Dict[str, Tuple[List[Document], datetime]] = {}
        self._cache_ttl = timedelta(minutes=15)
    
    async def add_knowledge(
        self,
        content: str,
        category: str,
        tags: Optional[List[str]] = None,
        source: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Add knowledge to the base
        
        Args:
            content: Knowledge content
            category: Knowledge category
            tags: Optional tags
            source: Optional source reference
            metadata: Additional metadata
            
        Returns:
            Document ID
        """
        # Build metadata
        doc_metadata = {
            "category": category,
            "tags": tags or [],
            "source": source,
            "added_at": datetime.utcnow().isoformat()
        }
        
        if metadata:
            doc_metadata.update(metadata)
        
        # Create document
        document = Document(
            content=content,
            metadata=doc_metadata
        )
        
        # Add to vector store
        doc_id = await self.vector_store.add_document(document)
        
        self._logger.info(
            "Knowledge added",
            doc_id=doc_id,
            category=category,
            tags=tags
        )
        
        # Clear cache for this category
        self._clear_category_cache(category)
        
        return doc_id
    
    async def add_file_knowledge(
        self,
        file_path: Path,
        category: str,
        tags: Optional[List[str]] = None,
        chunk_size: int = 1000,
        chunk_overlap: int = 200
    ) -> List[str]:
        """Add knowledge from a file
        
        Args:
            file_path: Path to file
            category: Knowledge category
            tags: Optional tags
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
            
        Returns:
            List of document IDs
        """
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Read file content
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Split into chunks
        chunks = self._split_into_chunks(content, chunk_size, chunk_overlap)
        
        # Add each chunk
        doc_ids = []
        for i, chunk in enumerate(chunks):
            doc_id = await self.add_knowledge(
                content=chunk,
                category=category,
                tags=tags,
                source=str(file_path),
                metadata={
                    "file_name": file_path.name,
                    "chunk_index": i,
                    "total_chunks": len(chunks)
                }
            )
            doc_ids.append(doc_id)
        
        self._logger.info(
            "File knowledge added",
            file=str(file_path),
            chunks=len(chunks),
            category=category
        )
        
        return doc_ids
    
    async def search(
        self,
        query: str,
        category: Optional[str] = None,
        tags: Optional[List[str]] = None,
        limit: int = 10,
        threshold: float = 0.7
    ) -> List[Tuple[Document, float]]:
        """Search knowledge base
        
        Args:
            query: Search query
            category: Optional category filter
            tags: Optional tag filter
            limit: Maximum results
            threshold: Similarity threshold
            
        Returns:
            List of (document, score) tuples
        """
        # Check cache
        cache_key = self._get_cache_key(query, category, tags)
        if cache_key in self._cache:
            cached_results, cached_time = self._cache[cache_key]
            if datetime.utcnow() - cached_time < self._cache_ttl:
                self._logger.info("Cache hit", query=query[:50])
                return [(doc, 1.0) for doc in cached_results]
        
        # Build metadata filter
        metadata_filter = {}
        if category:
            metadata_filter["category"] = category
        
        # Search vector store
        results = await self.vector_store.search(
            query=query,
            limit=limit * 2 if tags else limit,  # Get more if we need to filter by tags
            threshold=threshold,
            metadata_filter=metadata_filter
        )
        
        # Filter by tags if specified
        if tags:
            filtered_results = []
            for doc, score in results:
                doc_tags = doc.metadata.get("tags", [])
                if any(tag in doc_tags for tag in tags):
                    filtered_results.append((doc, score))
                if len(filtered_results) >= limit:
                    break
            results = filtered_results
        
        # Cache results
        self._cache[cache_key] = ([doc for doc, _ in results], datetime.utcnow())
        
        self._logger.info(
            "Search completed",
            query=query[:50],
            results=len(results),
            category=category,
            tags=tags
        )
        
        return results
    
    async def get_context(
        self,
        query: str,
        max_context_length: int = 2000,
        include_metadata: bool = False
    ) -> str:
        """Get relevant context for a query
        
        Args:
            query: Query to get context for
            max_context_length: Maximum context length
            include_metadata: Whether to include metadata
            
        Returns:
            Formatted context string
        """
        # Search for relevant documents
        results = await self.search(query, limit=5)
        
        if not results:
            return ""
        
        # Build context
        context_parts = []
        current_length = 0
        
        for doc, score in results:
            # Format document
            if include_metadata:
                doc_text = f"[Source: {doc.metadata.get('source', 'Unknown')}, Score: {score:.2f}]\n{doc.content}"
            else:
                doc_text = doc.content
            
            # Check if it fits
            if current_length + len(doc_text) + 2 > max_context_length:
                # Try to fit partial content
                remaining = max_context_length - current_length - 2
                if remaining > 100:  # Only include if meaningful
                    doc_text = doc_text[:remaining] + "..."
                    context_parts.append(doc_text)
                break
            
            context_parts.append(doc_text)
            current_length += len(doc_text) + 2  # +2 for newlines
        
        return "\n\n".join(context_parts)
    
    async def get_examples(
        self,
        category: str,
        limit: int = 5
    ) -> List[Document]:
        """Get example documents from a category
        
        Args:
            category: Category to get examples from
            limit: Maximum number of examples
            
        Returns:
            List of example documents
        """
        # Search for documents in category
        results = await self.vector_store.search(
            query=f"example {category}",
            limit=limit,
            threshold=0.5,
            metadata_filter={"category": category}
        )
        
        return [doc for doc, _ in results]
    
    async def update_knowledge(
        self,
        doc_id: str,
        content: Optional[str] = None,
        metadata_updates: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Update existing knowledge
        
        Args:
            doc_id: Document ID
            content: New content (optional)
            metadata_updates: Metadata updates (optional)
            
        Returns:
            True if successful
        """
        # Get existing document
        doc = await self.vector_store.get_document(doc_id)
        if not doc:
            return False
        
        # Update metadata
        if metadata_updates:
            doc.metadata.update(metadata_updates)
            doc.metadata["updated_at"] = datetime.utcnow().isoformat()
        
        # Update in vector store
        success = await self.vector_store.update_document(
            doc_id=doc_id,
            content=content,
            metadata=doc.metadata if metadata_updates else None
        )
        
        if success:
            # Clear relevant caches
            category = doc.metadata.get("category")
            if category:
                self._clear_category_cache(category)
        
        return success
    
    async def delete_knowledge(self, doc_id: str) -> bool:
        """Delete knowledge
        
        Args:
            doc_id: Document ID
            
        Returns:
            True if successful
        """
        # Get document for cache clearing
        doc = await self.vector_store.get_document(doc_id)
        
        # Delete from vector store
        success = await self.vector_store.delete_document(doc_id)
        
        if success and doc:
            # Clear relevant caches
            category = doc.metadata.get("category")
            if category:
                self._clear_category_cache(category)
        
        return success
    
    async def get_categories(self) -> List[str]:
        """Get all knowledge categories
        
        Returns:
            List of category names
        """
        # This would be more efficient with a dedicated query
        # For now, get statistics and extract categories
        stats = await self.vector_store.get_statistics()
        
        # In production, this would query distinct categories
        # For now, return common categories
        return [
            "examples",
            "documentation",
            "patterns",
            "best_practices",
            "troubleshooting",
            "reference"
        ]
    
    async def get_tags(self, category: Optional[str] = None) -> List[str]:
        """Get all tags, optionally filtered by category
        
        Args:
            category: Optional category filter
            
        Returns:
            List of tag names
        """
        # This would be more efficient with a dedicated query
        # For now, return common tags
        tags = [
            "python",
            "javascript",
            "api",
            "database",
            "security",
            "performance",
            "testing",
            "deployment"
        ]
        
        return tags
    
    async def export_knowledge(
        self,
        output_path: Path,
        category: Optional[str] = None,
        format: str = "json"
    ) -> bool:
        """Export knowledge to file
        
        Args:
            output_path: Output file path
            category: Optional category filter
            format: Export format (json, markdown)
            
        Returns:
            True if successful
        """
        try:
            # Get all documents (this would be paginated in production)
            if category:
                results = await self.search(
                    query="",  # Empty query to get all
                    category=category,
                    limit=1000,
                    threshold=0.0
                )
                documents = [doc for doc, _ in results]
            else:
                # Would need a get_all method in production
                documents = []
            
            if format == "json":
                # Export as JSON
                export_data = {
                    "exported_at": datetime.utcnow().isoformat(),
                    "category": category,
                    "total_documents": len(documents),
                    "documents": [
                        {
                            "id": doc.id,
                            "content": doc.content,
                            "metadata": doc.metadata,
                            "created_at": doc.created_at.isoformat() if doc.created_at else None
                        }
                        for doc in documents
                    ]
                }
                
                with open(output_path, 'w') as f:
                    json.dump(export_data, f, indent=2)
                    
            elif format == "markdown":
                # Export as Markdown
                with open(output_path, 'w') as f:
                    f.write(f"# Knowledge Base Export\n\n")
                    f.write(f"Exported at: {datetime.utcnow().isoformat()}\n\n")
                    
                    if category:
                        f.write(f"Category: {category}\n\n")
                    
                    f.write(f"Total documents: {len(documents)}\n\n")
                    
                    for doc in documents:
                        f.write(f"## {doc.id}\n\n")
                        f.write(f"**Category:** {doc.metadata.get('category', 'Unknown')}\n")
                        f.write(f"**Tags:** {', '.join(doc.metadata.get('tags', []))}\n")
                        f.write(f"**Source:** {doc.metadata.get('source', 'Unknown')}\n\n")
                        f.write(f"{doc.content}\n\n")
                        f.write("---\n\n")
            
            self._logger.info(
                "Knowledge exported",
                path=str(output_path),
                format=format,
                documents=len(documents)
            )
            
            return True
            
        except Exception as e:
            self._logger.error("Export failed", error=str(e))
            return False
    
    async def import_knowledge(
        self,
        input_path: Path,
        category_override: Optional[str] = None
    ) -> int:
        """Import knowledge from file
        
        Args:
            input_path: Input file path
            category_override: Override category for all documents
            
        Returns:
            Number of documents imported
        """
        if not input_path.exists():
            raise FileNotFoundError(f"File not found: {input_path}")
        
        try:
            with open(input_path, 'r') as f:
                data = json.load(f)
            
            documents = data.get("documents", [])
            imported = 0
            
            for doc_data in documents:
                # Override category if specified
                if category_override:
                    doc_data["metadata"]["category"] = category_override
                
                # Create document
                doc = Document(
                    content=doc_data["content"],
                    metadata=doc_data["metadata"]
                )
                
                # Add to knowledge base
                await self.vector_store.add_document(doc)
                imported += 1
            
            self._logger.info(
                "Knowledge imported",
                path=str(input_path),
                documents=imported
            )
            
            # Clear cache
            self._cache.clear()
            
            return imported
            
        except Exception as e:
            self._logger.error("Import failed", error=str(e))
            raise
    
    def _split_into_chunks(
        self,
        text: str,
        chunk_size: int,
        chunk_overlap: int
    ) -> List[str]:
        """Split text into overlapping chunks
        
        Args:
            text: Text to split
            chunk_size: Size of each chunk
            chunk_overlap: Overlap between chunks
            
        Returns:
            List of text chunks
        """
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            
            # Try to break at sentence boundary
            if end < len(text):
                last_period = chunk.rfind('.')
                if last_period > chunk_size * 0.8:
                    chunk = chunk[:last_period + 1]
                    end = start + last_period + 1
            
            chunks.append(chunk.strip())
            start = end - chunk_overlap
        
        return chunks
    
    def _get_cache_key(
        self,
        query: str,
        category: Optional[str],
        tags: Optional[List[str]]
    ) -> str:
        """Generate cache key"""
        parts = [query]
        if category:
            parts.append(f"cat:{category}")
        if tags:
            parts.append(f"tags:{','.join(sorted(tags))}")
        
        return "|".join(parts)
    
    def _clear_category_cache(self, category: str):
        """Clear cache entries for a category"""
        keys_to_remove = []
        for key in self._cache:
            if f"cat:{category}" in key:
                keys_to_remove.append(key)
        
        for key in keys_to_remove:
            del self._cache[key]