"""Persistent conversation memory for workflows"""

import json
import os
import sqlite3
from typing import Dict, Any, List, Optional
from datetime import datetime
from pathlib import Path
import hashlib


class ConversationMemory:
    """Manages persistent conversation memory across sessions"""
    
    def __init__(self, memory_dir: str = "./memory"):
        """Initialize conversation memory
        
        Args:
            memory_dir: Directory to store memory data
        """
        self.memory_dir = Path(memory_dir)
        self.memory_dir.mkdir(exist_ok=True)
        
        self.db_path = self.memory_dir / "conversations.db"
        self._init_database()
    
    def _init_database(self):
        """Initialize the SQLite database"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Create conversations table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS conversations (
                    id TEXT PRIMARY KEY,
                    thread_id TEXT NOT NULL,
                    user_id TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    metadata TEXT
                )
            """)
            
            # Create messages table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    conversation_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    metadata TEXT,
                    FOREIGN KEY (conversation_id) REFERENCES conversations(id)
                )
            """)
            
            # Create memory entries table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS memory_entries (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    conversation_id TEXT NOT NULL,
                    entry_type TEXT NOT NULL,
                    content TEXT NOT NULL,
                    embedding TEXT,
                    importance REAL DEFAULT 0.5,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    metadata TEXT,
                    FOREIGN KEY (conversation_id) REFERENCES conversations(id)
                )
            """)
            
            # Create indices
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_thread_id ON conversations(thread_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_conv_messages ON messages(conversation_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_conv_memory ON memory_entries(conversation_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_importance ON memory_entries(importance)")
            
            conn.commit()
    
    def create_conversation(
        self,
        thread_id: str,
        user_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Create a new conversation
        
        Args:
            thread_id: Thread identifier
            user_id: Optional user identifier
            metadata: Optional metadata
            
        Returns:
            Conversation ID
        """
        conversation_id = self._generate_conversation_id(thread_id, user_id)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR IGNORE INTO conversations (id, thread_id, user_id, metadata)
                VALUES (?, ?, ?, ?)
            """, (
                conversation_id,
                thread_id,
                user_id,
                json.dumps(metadata or {})
            ))
            conn.commit()
        
        return conversation_id
    
    def add_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Add a message to the conversation
        
        Args:
            conversation_id: Conversation ID
            role: Message role (user/assistant/system)
            content: Message content
            metadata: Optional metadata
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO messages (conversation_id, role, content, metadata)
                VALUES (?, ?, ?, ?)
            """, (
                conversation_id,
                role,
                content,
                json.dumps(metadata or {})
            ))
            
            # Update conversation timestamp
            cursor.execute("""
                UPDATE conversations
                SET updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
            """, (conversation_id,))
            
            conn.commit()
    
    def get_conversation_history(
        self,
        conversation_id: str,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Get conversation history
        
        Args:
            conversation_id: Conversation ID
            limit: Optional limit on number of messages
            
        Returns:
            List of messages
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            query = """
                SELECT role, content, timestamp, metadata
                FROM messages
                WHERE conversation_id = ?
                ORDER BY timestamp DESC
            """
            
            if limit:
                query += f" LIMIT {limit}"
            
            cursor.execute(query, (conversation_id,))
            
            messages = []
            for row in cursor.fetchall():
                messages.append({
                    "role": row[0],
                    "content": row[1],
                    "timestamp": row[2],
                    "metadata": json.loads(row[3] or "{}")
                })
            
            # Return in chronological order
            return list(reversed(messages))
    
    def add_memory_entry(
        self,
        conversation_id: str,
        entry_type: str,
        content: str,
        importance: float = 0.5,
        embedding: Optional[List[float]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Add a memory entry
        
        Args:
            conversation_id: Conversation ID
            entry_type: Type of memory entry (fact/insight/summary/etc)
            content: Memory content
            importance: Importance score (0-1)
            embedding: Optional embedding vector
            metadata: Optional metadata
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO memory_entries 
                (conversation_id, entry_type, content, importance, embedding, metadata)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                conversation_id,
                entry_type,
                content,
                importance,
                json.dumps(embedding) if embedding else None,
                json.dumps(metadata or {})
            ))
            conn.commit()
    
    def get_relevant_memories(
        self,
        conversation_id: str,
        query: Optional[str] = None,
        entry_types: Optional[List[str]] = None,
        min_importance: float = 0.0,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Get relevant memories for a conversation
        
        Args:
            conversation_id: Conversation ID
            query: Optional query for similarity search
            entry_types: Optional list of entry types to filter
            min_importance: Minimum importance threshold
            limit: Maximum number of results
            
        Returns:
            List of memory entries
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Build query
            conditions = ["conversation_id = ?", f"importance >= {min_importance}"]
            params = [conversation_id]
            
            if entry_types:
                placeholders = ",".join(["?" for _ in entry_types])
                conditions.append(f"entry_type IN ({placeholders})")
                params.extend(entry_types)
            
            query_str = f"""
                SELECT entry_type, content, importance, created_at, metadata
                FROM memory_entries
                WHERE {' AND '.join(conditions)}
                ORDER BY importance DESC, created_at DESC
                LIMIT {limit}
            """
            
            cursor.execute(query_str, params)
            
            memories = []
            for row in cursor.fetchall():
                memories.append({
                    "entry_type": row[0],
                    "content": row[1],
                    "importance": row[2],
                    "created_at": row[3],
                    "metadata": json.loads(row[4] or "{}")
                })
            
            return memories
    
    def summarize_conversation(
        self,
        conversation_id: str,
        summary: str,
        importance: float = 0.8
    ):
        """Add a conversation summary to memory
        
        Args:
            conversation_id: Conversation ID
            summary: Conversation summary
            importance: Importance of the summary
        """
        self.add_memory_entry(
            conversation_id=conversation_id,
            entry_type="summary",
            content=summary,
            importance=importance,
            metadata={"auto_generated": True}
        )
    
    def extract_facts(
        self,
        conversation_id: str,
        facts: List[str],
        importance: float = 0.6
    ):
        """Extract and store facts from conversation
        
        Args:
            conversation_id: Conversation ID
            facts: List of facts
            importance: Importance of facts
        """
        for fact in facts:
            self.add_memory_entry(
                conversation_id=conversation_id,
                entry_type="fact",
                content=fact,
                importance=importance
            )
    
    def get_conversation_stats(self, conversation_id: str) -> Dict[str, Any]:
        """Get statistics for a conversation
        
        Args:
            conversation_id: Conversation ID
            
        Returns:
            Dictionary of statistics
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Get message count
            cursor.execute("""
                SELECT COUNT(*) FROM messages WHERE conversation_id = ?
            """, (conversation_id,))
            message_count = cursor.fetchone()[0]
            
            # Get memory entry count
            cursor.execute("""
                SELECT COUNT(*) FROM memory_entries WHERE conversation_id = ?
            """, (conversation_id,))
            memory_count = cursor.fetchone()[0]
            
            # Get conversation metadata
            cursor.execute("""
                SELECT created_at, updated_at, metadata
                FROM conversations WHERE id = ?
            """, (conversation_id,))
            
            row = cursor.fetchone()
            if row:
                return {
                    "message_count": message_count,
                    "memory_count": memory_count,
                    "created_at": row[0],
                    "updated_at": row[1],
                    "metadata": json.loads(row[2] or "{}")
                }
            
            return {}
    
    def _generate_conversation_id(self, thread_id: str, user_id: Optional[str]) -> str:
        """Generate a unique conversation ID
        
        Args:
            thread_id: Thread ID
            user_id: Optional user ID
            
        Returns:
            Conversation ID
        """
        components = [thread_id]
        if user_id:
            components.append(user_id)
        components.append(datetime.utcnow().isoformat())
        
        return hashlib.sha256("-".join(components).encode()).hexdigest()[:16]
    
    def cleanup_old_conversations(self, days: int = 30):
        """Clean up old conversations
        
        Args:
            days: Number of days to keep conversations
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Delete old conversations
            cursor.execute("""
                DELETE FROM conversations
                WHERE updated_at < datetime('now', '-{} days')
            """.format(days))
            
            conn.commit()


class WorkingMemory:
    """Short-term working memory for active tasks"""
    
    def __init__(self, capacity: int = 10):
        """Initialize working memory
        
        Args:
            capacity: Maximum number of items in working memory
        """
        self.capacity = capacity
        self.memory: List[Dict[str, Any]] = []
    
    def add(self, item: Dict[str, Any]):
        """Add item to working memory
        
        Args:
            item: Item to add
        """
        self.memory.append({
            **item,
            "timestamp": datetime.utcnow().isoformat()
        })
        
        # Remove oldest items if over capacity
        if len(self.memory) > self.capacity:
            self.memory = self.memory[-self.capacity:]
    
    def get_recent(self, n: int = 5) -> List[Dict[str, Any]]:
        """Get n most recent items
        
        Args:
            n: Number of items to get
            
        Returns:
            List of recent items
        """
        return self.memory[-n:]
    
    def find_related(self, query: str) -> List[Dict[str, Any]]:
        """Find items related to query
        
        Args:
            query: Search query
            
        Returns:
            List of related items
        """
        query_lower = query.lower()
        related = []
        
        for item in self.memory:
            content = str(item.get("content", "")).lower()
            if query_lower in content:
                related.append(item)
        
        return related
    
    def clear(self):
        """Clear working memory"""
        self.memory.clear()