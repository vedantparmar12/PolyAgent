"""Vector storage system using Supabase and OpenAI embeddings"""

from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from pydantic import BaseModel, Field
import numpy as np
from supabase import create_client, Client
import openai
import logfire
import asyncio
from tenacity import retry, stop_after_attempt, wait_exponential


class Document(BaseModel):
    """Document model for vector storage"""
    id: Optional[str] = Field(default=None, description="Document ID")
    content: str = Field(description="Document content")
    embedding: Optional[List[float]] = Field(default=None, description="Document embedding")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Document metadata")
    created_at: Optional[datetime] = Field(default=None, description="Creation timestamp")
    updated_at: Optional[datetime] = Field(default=None, description="Update timestamp")


class VectorStore:
    """Vector storage implementation using Supabase and OpenAI"""
    
    def __init__(
        self,
        supabase_url: str,
        supabase_key: str,
        openai_api_key: str,
        table_name: str = "documents",
        embedding_dimension: int = 1536
    ):
        """Initialize vector store
        
        Args:
            supabase_url: Supabase project URL
            supabase_key: Supabase API key
            openai_api_key: OpenAI API key
            table_name: Table name for documents
            embedding_dimension: Embedding vector dimension
        """
        self.supabase: Client = create_client(supabase_url, supabase_key)
        self.openai_client = openai.AsyncOpenAI(api_key=openai_api_key)
        self.table_name = table_name
        self.embedding_dimension = embedding_dimension
        self._logger = logfire.span("vector_store")
        
        # Initialize table if needed
        self._initialize_table()
    
    def _initialize_table(self):
        """Initialize vector storage table"""
        # This would typically be done via Supabase migrations
        # Table structure:
        # - id: uuid primary key
        # - content: text
        # - embedding: vector(1536)
        # - metadata: jsonb
        # - created_at: timestamp
        # - updated_at: timestamp
        self._logger.info("Vector store initialized", table=self.table_name)
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def _generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for text using OpenAI
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector
        """
        try:
            response = await self.openai_client.embeddings.create(
                model="text-embedding-ada-002",
                input=text
            )
            
            embedding = response.data[0].embedding
            
            self._logger.info(
                "Embedding generated",
                text_length=len(text),
                embedding_dim=len(embedding)
            )
            
            return embedding
            
        except Exception as e:
            self._logger.error("Failed to generate embedding", error=str(e))
            raise
    
    async def add_document(self, document: Document) -> str:
        """Add document to vector store
        
        Args:
            document: Document to add
            
        Returns:
            Document ID
        """
        # Generate embedding if not provided
        if not document.embedding:
            document.embedding = await self._generate_embedding(document.content)
        
        # Prepare data
        data = {
            "content": document.content,
            "embedding": document.embedding,
            "metadata": document.metadata,
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat()
        }
        
        # Insert into Supabase
        try:
            result = self.supabase.table(self.table_name).insert(data).execute()
            
            doc_id = result.data[0]["id"]
            
            self._logger.info(
                "Document added",
                doc_id=doc_id,
                content_length=len(document.content)
            )
            
            return doc_id
            
        except Exception as e:
            self._logger.error("Failed to add document", error=str(e))
            raise
    
    async def search(
        self,
        query: str,
        limit: int = 10,
        threshold: float = 0.7,
        metadata_filter: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[Document, float]]:
        """Search for similar documents
        
        Args:
            query: Search query
            limit: Maximum number of results
            threshold: Similarity threshold
            metadata_filter: Optional metadata filter
            
        Returns:
            List of (document, similarity_score) tuples
        """
        # Generate query embedding
        query_embedding = await self._generate_embedding(query)
        
        # Search by embedding
        results = await self.search_by_embedding(
            embedding=query_embedding,
            limit=limit,
            threshold=threshold,
            metadata_filter=metadata_filter
        )
        
        self._logger.info(
            "Search completed",
            query=query[:50],
            results=len(results)
        )
        
        return results
    
    async def search_by_embedding(
        self,
        embedding: List[float],
        limit: int = 10,
        threshold: float = 0.7,
        metadata_filter: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[Document, float]]:
        """Search by embedding vector
        
        Args:
            embedding: Embedding vector
            limit: Maximum number of results
            threshold: Similarity threshold
            metadata_filter: Optional metadata filter
            
        Returns:
            List of (document, similarity_score) tuples
        """
        try:
            # Build RPC call for vector similarity search
            rpc_params = {
                "query_embedding": embedding,
                "match_threshold": threshold,
                "match_count": limit
            }
            
            # Add metadata filter if provided
            if metadata_filter:
                rpc_params["filter"] = metadata_filter
            
            # Execute similarity search
            # Note: This assumes a Supabase function 'match_documents' exists
            # that performs cosine similarity search
            result = self.supabase.rpc(
                "match_documents",
                rpc_params
            ).execute()
            
            # Convert results to Document objects
            documents = []
            for row in result.data:
                doc = Document(
                    id=row["id"],
                    content=row["content"],
                    embedding=row["embedding"],
                    metadata=row["metadata"],
                    created_at=datetime.fromisoformat(row["created_at"]),
                    updated_at=datetime.fromisoformat(row["updated_at"])
                )
                similarity = row["similarity"]
                documents.append((doc, similarity))
            
            return documents
            
        except Exception as e:
            self._logger.error("Search failed", error=str(e))
            raise
    
    async def get_document(self, doc_id: str) -> Optional[Document]:
        """Get document by ID
        
        Args:
            doc_id: Document ID
            
        Returns:
            Document or None if not found
        """
        try:
            result = self.supabase.table(self.table_name).select("*").eq("id", doc_id).execute()
            
            if not result.data:
                return None
            
            row = result.data[0]
            return Document(
                id=row["id"],
                content=row["content"],
                embedding=row["embedding"],
                metadata=row["metadata"],
                created_at=datetime.fromisoformat(row["created_at"]),
                updated_at=datetime.fromisoformat(row["updated_at"])
            )
            
        except Exception as e:
            self._logger.error("Failed to get document", doc_id=doc_id, error=str(e))
            return None
    
    async def update_document(
        self,
        doc_id: str,
        content: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Update document
        
        Args:
            doc_id: Document ID
            content: New content (optional)
            metadata: New metadata (optional)
            
        Returns:
            True if successful
        """
        try:
            update_data = {
                "updated_at": datetime.utcnow().isoformat()
            }
            
            # Update content and regenerate embedding if needed
            if content is not None:
                update_data["content"] = content
                update_data["embedding"] = await self._generate_embedding(content)
            
            # Update metadata
            if metadata is not None:
                update_data["metadata"] = metadata
            
            # Execute update
            result = self.supabase.table(self.table_name).update(update_data).eq("id", doc_id).execute()
            
            self._logger.info("Document updated", doc_id=doc_id)
            
            return len(result.data) > 0
            
        except Exception as e:
            self._logger.error("Failed to update document", doc_id=doc_id, error=str(e))
            return False
    
    async def delete_document(self, doc_id: str) -> bool:
        """Delete document
        
        Args:
            doc_id: Document ID
            
        Returns:
            True if successful
        """
        try:
            result = self.supabase.table(self.table_name).delete().eq("id", doc_id).execute()
            
            self._logger.info("Document deleted", doc_id=doc_id)
            
            return len(result.data) > 0
            
        except Exception as e:
            self._logger.error("Failed to delete document", doc_id=doc_id, error=str(e))
            return False
    
    async def batch_add_documents(self, documents: List[Document]) -> List[str]:
        """Add multiple documents in batch
        
        Args:
            documents: List of documents to add
            
        Returns:
            List of document IDs
        """
        doc_ids = []
        
        # Process in parallel with concurrency limit
        semaphore = asyncio.Semaphore(5)  # Limit concurrent operations
        
        async def add_with_semaphore(doc: Document) -> str:
            async with semaphore:
                return await self.add_document(doc)
        
        # Add all documents
        tasks = [add_with_semaphore(doc) for doc in documents]
        doc_ids = await asyncio.gather(*tasks)
        
        self._logger.info("Batch add completed", count=len(doc_ids))
        
        return doc_ids
    
    async def clear_collection(self, metadata_filter: Optional[Dict[str, Any]] = None) -> int:
        """Clear documents from collection
        
        Args:
            metadata_filter: Optional metadata filter
            
        Returns:
            Number of documents deleted
        """
        try:
            query = self.supabase.table(self.table_name).select("id")
            
            # Apply metadata filter if provided
            if metadata_filter:
                for key, value in metadata_filter.items():
                    query = query.eq(f"metadata->{key}", value)
            
            # Get document IDs
            result = query.execute()
            doc_ids = [row["id"] for row in result.data]
            
            # Delete documents
            if doc_ids:
                delete_result = self.supabase.table(self.table_name).delete().in_("id", doc_ids).execute()
                
                self._logger.info("Collection cleared", count=len(doc_ids))
                
                return len(delete_result.data)
            
            return 0
            
        except Exception as e:
            self._logger.error("Failed to clear collection", error=str(e))
            raise
    
    async def get_statistics(self) -> Dict[str, Any]:
        """Get vector store statistics
        
        Returns:
            Statistics dictionary
        """
        try:
            # Get total count
            count_result = self.supabase.table(self.table_name).select("count", count="exact").execute()
            total_documents = count_result.count
            
            # Get metadata statistics (would need custom RPC in production)
            stats = {
                "total_documents": total_documents,
                "table_name": self.table_name,
                "embedding_dimension": self.embedding_dimension,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            self._logger.info("Statistics retrieved", total=total_documents)
            
            return stats
            
        except Exception as e:
            self._logger.error("Failed to get statistics", error=str(e))
            raise
    
    def calculate_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """Calculate cosine similarity between embeddings
        
        Args:
            embedding1: First embedding
            embedding2: Second embedding
            
        Returns:
            Cosine similarity score
        """
        # Convert to numpy arrays
        vec1 = np.array(embedding1)
        vec2 = np.array(embedding2)
        
        # Calculate cosine similarity
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        similarity = dot_product / (norm1 * norm2)
        
        return float(similarity)