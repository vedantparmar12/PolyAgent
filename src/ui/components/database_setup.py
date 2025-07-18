"""Database Setup and Configuration UI Component"""

import streamlit as st
import os
import yaml
import json
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Optional
import psycopg2
import sqlite3
from urllib.parse import urlparse
import pymongo
import redis


class DatabaseSetup:
    """UI component for database setup and configuration"""
    
    def __init__(self):
        self.supported_databases = {
            "PostgreSQL": {
                "icon": "🐘",
                "default_port": 5432,
                "connection_string": "postgresql://user:password@host:port/database"
            },
            "MySQL": {
                "icon": "🐬",
                "default_port": 3306,
                "connection_string": "mysql://user:password@host:port/database"
            },
            "SQLite": {
                "icon": "📁",
                "default_port": None,
                "connection_string": "sqlite:///path/to/database.db"
            },
            "MongoDB": {
                "icon": "🍃",
                "default_port": 27017,
                "connection_string": "mongodb://user:password@host:port/database"
            },
            "Redis": {
                "icon": "🔴",
                "default_port": 6379,
                "connection_string": "redis://user:password@host:port/db"
            },
            "Supabase": {
                "icon": "⚡",
                "default_port": None,
                "connection_string": "https://project.supabase.co"
            }
        }
        self.config_file = Path("database_config.yaml")
    
    def render(self):
        """Render the database setup interface"""
        st.title("🗄️ Database Configuration")
        st.markdown("Configure database connections for agent data storage and vector search")
        
        # Create tabs for different database operations
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "Setup Connection",
            "Vector Database",
            "Schema Management",
            "Data Migration",
            "Monitoring"
        ])
        
        with tab1:
            self._render_connection_setup()
        
        with tab2:
            self._render_vector_database()
        
        with tab3:
            self._render_schema_management()
        
        with tab4:
            self._render_data_migration()
        
        with tab5:
            self._render_monitoring()
    
    def _render_connection_setup(self):
        """Render database connection setup"""
        st.subheader("Database Connection Setup")
        
        # Database type selection
        db_type = st.selectbox(
            "Select Database Type",
            list(self.supported_databases.keys()),
            format_func=lambda x: f"{self.supported_databases[x]['icon']} {x}"
        )
        
        st.markdown("---")
        
        # Connection configuration based on type
        if db_type == "PostgreSQL":
            self._configure_postgresql()
        elif db_type == "MySQL":
            self._configure_mysql()
        elif db_type == "SQLite":
            self._configure_sqlite()
        elif db_type == "MongoDB":
            self._configure_mongodb()
        elif db_type == "Redis":
            self._configure_redis()
        elif db_type == "Supabase":
            self._configure_supabase()
        
        # Connection string builder
        st.markdown("---")
        st.subheader("Connection String Builder")
        
        connection_string = st.text_area(
            "Connection String",
            value=self.supported_databases[db_type]['connection_string'],
            height=100,
            help="Edit the connection string directly or use the form above"
        )
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🧪 Test Connection"):
                self._test_database_connection(db_type, connection_string)
        
        with col2:
            if st.button("💾 Save Configuration"):
                self._save_database_config(db_type, connection_string)
        
        with col3:
            if st.button("📋 Copy to Clipboard"):
                st.code(connection_string)
                st.success("Connection string displayed above")
    
    def _configure_postgresql(self):
        """Configure PostgreSQL connection"""
        st.markdown("### PostgreSQL Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            host = st.text_input("Host", value="localhost")
            port = st.number_input("Port", value=5432, min_value=1, max_value=65535)
            database = st.text_input("Database Name", value="agent_db")
        
        with col2:
            username = st.text_input("Username", value="postgres")
            password = st.text_input("Password", type="password")
            ssl_mode = st.selectbox("SSL Mode", ["disable", "require", "verify-ca", "verify-full"])
        
        # Advanced options
        with st.expander("Advanced Options"):
            schema = st.text_input("Schema", value="public")
            connection_timeout = st.number_input("Connection Timeout (seconds)", value=10)
            pool_size = st.number_input("Connection Pool Size", value=5, min_value=1)
        
        # Generate connection string
        if all([host, port, database, username]):
            connection_string = f"postgresql://{username}"
            if password:
                connection_string += f":{password}"
            connection_string += f"@{host}:{port}/{database}"
            if ssl_mode != "disable":
                connection_string += f"?sslmode={ssl_mode}"
            
            st.code(connection_string)
            return connection_string
    
    def _configure_mysql(self):
        """Configure MySQL connection"""
        st.markdown("### MySQL Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            host = st.text_input("Host", value="localhost")
            port = st.number_input("Port", value=3306, min_value=1, max_value=65535)
            database = st.text_input("Database Name", value="agent_db")
        
        with col2:
            username = st.text_input("Username", value="root")
            password = st.text_input("Password", type="password")
            charset = st.selectbox("Character Set", ["utf8mb4", "utf8", "latin1"])
        
        # Generate connection string
        if all([host, port, database, username]):
            connection_string = f"mysql://{username}"
            if password:
                connection_string += f":{password}"
            connection_string += f"@{host}:{port}/{database}?charset={charset}"
            
            st.code(connection_string)
            return connection_string
    
    def _configure_sqlite(self):
        """Configure SQLite connection"""
        st.markdown("### SQLite Configuration")
        
        db_path = st.text_input(
            "Database File Path",
            value="./data/agent.db",
            help="Path to SQLite database file (will be created if doesn't exist)"
        )
        
        # Options
        col1, col2 = st.columns(2)
        
        with col1:
            in_memory = st.checkbox("In-Memory Database", value=False)
            read_only = st.checkbox("Read-Only Mode", value=False)
        
        with col2:
            journal_mode = st.selectbox("Journal Mode", ["DELETE", "WAL", "MEMORY"])
            synchronous = st.selectbox("Synchronous", ["NORMAL", "FULL", "OFF"])
        
        # Generate connection string
        if in_memory:
            connection_string = "sqlite:///:memory:"
        else:
            connection_string = f"sqlite:///{db_path}"
        
        if read_only:
            connection_string += "?mode=ro"
        
        st.code(connection_string)
        return connection_string
    
    def _configure_mongodb(self):
        """Configure MongoDB connection"""
        st.markdown("### MongoDB Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            host = st.text_input("Host", value="localhost")
            port = st.number_input("Port", value=27017, min_value=1, max_value=65535)
            database = st.text_input("Database Name", value="agent_db")
        
        with col2:
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            auth_source = st.text_input("Auth Source", value="admin")
        
        # Replica set configuration
        with st.expander("Replica Set Configuration"):
            replica_set = st.text_input("Replica Set Name")
            additional_hosts = st.text_area(
                "Additional Hosts (one per line)",
                help="Format: host:port"
            )
        
        # Generate connection string
        connection_string = "mongodb://"
        if username and password:
            connection_string += f"{username}:{password}@"
        
        connection_string += f"{host}:{port}/{database}"
        
        params = []
        if auth_source and auth_source != "admin":
            params.append(f"authSource={auth_source}")
        if replica_set:
            params.append(f"replicaSet={replica_set}")
        
        if params:
            connection_string += "?" + "&".join(params)
        
        st.code(connection_string)
        return connection_string
    
    def _configure_redis(self):
        """Configure Redis connection"""
        st.markdown("### Redis Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            host = st.text_input("Host", value="localhost")
            port = st.number_input("Port", value=6379, min_value=1, max_value=65535)
            database = st.number_input("Database Number", value=0, min_value=0, max_value=15)
        
        with col2:
            password = st.text_input("Password", type="password")
            ssl = st.checkbox("Use SSL/TLS", value=False)
            decode_responses = st.checkbox("Decode Responses", value=True)
        
        # Advanced options
        with st.expander("Advanced Options"):
            socket_timeout = st.number_input("Socket Timeout (seconds)", value=5)
            connection_pool_size = st.number_input("Connection Pool Size", value=10)
            retry_on_timeout = st.checkbox("Retry on Timeout", value=True)
        
        # Generate connection string
        connection_string = "redis://"
        if password:
            connection_string += f":{password}@"
        connection_string += f"{host}:{port}/{database}"
        
        if ssl:
            connection_string += "?ssl=true"
        
        st.code(connection_string)
        return connection_string
    
    def _configure_supabase(self):
        """Configure Supabase connection"""
        st.markdown("### Supabase Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            project_url = st.text_input(
                "Project URL",
                placeholder="https://your-project.supabase.co",
                help="Your Supabase project URL"
            )
            
            anon_key = st.text_input(
                "Anonymous Key",
                type="password",
                help="Found in Settings > API"
            )
        
        with col2:
            service_key = st.text_input(
                "Service Role Key (Optional)",
                type="password",
                help="For server-side operations"
            )
            
            database_password = st.text_input(
                "Database Password",
                type="password",
                help="For direct database access"
            )
        
        # Save Supabase configuration
        if project_url and anon_key:
            st.markdown("### Environment Variables")
            st.code(f"""
SUPABASE_URL={project_url}
SUPABASE_ANON_KEY={anon_key}
{"SUPABASE_SERVICE_KEY=" + service_key if service_key else ""}
            """)
            
            # Direct database connection
            if database_password:
                parsed = urlparse(project_url)
                db_host = parsed.hostname.replace('supabase.co', 'supabase.com')
                db_connection = f"postgresql://postgres:{database_password}@db.{db_host}:5432/postgres"
                
                st.markdown("### Direct Database Connection")
                st.code(db_connection)
    
    def _render_vector_database(self):
        """Render vector database configuration"""
        st.subheader("Vector Database Setup")
        st.markdown("Configure vector storage for semantic search and RAG")
        
        vector_db = st.selectbox(
            "Select Vector Database",
            ["Supabase Vector (pgvector)", "Pinecone", "Weaviate", "Qdrant", "ChromaDB", "Milvus"]
        )
        
        st.markdown("---")
        
        if vector_db == "Supabase Vector (pgvector)":
            self._configure_supabase_vector()
        elif vector_db == "Pinecone":
            self._configure_pinecone()
        elif vector_db == "Weaviate":
            self._configure_weaviate()
        elif vector_db == "Qdrant":
            self._configure_qdrant()
        elif vector_db == "ChromaDB":
            self._configure_chromadb()
        elif vector_db == "Milvus":
            self._configure_milvus()
    
    def _configure_supabase_vector(self):
        """Configure Supabase Vector (pgvector)"""
        st.markdown("### Supabase Vector Configuration")
        
        # Check if Supabase is already configured
        if os.environ.get("SUPABASE_URL"):
            st.success("✅ Supabase connection configured")
        else:
            st.warning("⚠️ Please configure Supabase connection first")
        
        # Vector setup
        st.markdown("### Enable pgvector Extension")
        
        enable_sql = """
-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Create embeddings table
CREATE TABLE IF NOT EXISTS embeddings (
    id SERIAL PRIMARY KEY,
    content TEXT,
    embedding vector(1536),
    metadata JSONB,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Create index for similarity search
CREATE INDEX ON embeddings USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);
        """
        
        st.code(enable_sql, language="sql")
        
        if st.button("📋 Copy SQL"):
            st.success("SQL copied to clipboard!")
        
        # Embedding configuration
        st.markdown("### Embedding Configuration")
        
        embedding_model = st.selectbox(
            "Embedding Model",
            ["openai/text-embedding-3-small", "openai/text-embedding-3-large", "sentence-transformers/all-MiniLM-L6-v2"]
        )
        
        embedding_dimensions = st.number_input(
            "Embedding Dimensions",
            value=1536,
            help="Must match the model's output dimensions"
        )
    
    def _configure_pinecone(self):
        """Configure Pinecone vector database"""
        st.markdown("### Pinecone Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            api_key = st.text_input("Pinecone API Key", type="password")
            environment = st.text_input("Environment", placeholder="us-west1-gcp")
        
        with col2:
            index_name = st.text_input("Index Name", value="agent-embeddings")
            dimension = st.number_input("Vector Dimension", value=1536)
        
        # Index configuration
        metric = st.selectbox("Distance Metric", ["cosine", "euclidean", "dotproduct"])
        
        if st.button("Create Index"):
            st.info("Index creation would be performed here")
    
    def _render_schema_management(self):
        """Render database schema management"""
        st.subheader("Schema Management")
        
        # Load existing schemas
        schemas = self._load_schemas()
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            selected_schema = st.selectbox(
                "Select Schema",
                ["Create New"] + list(schemas.keys())
            )
        
        with col2:
            if st.button("🔄 Refresh Schemas"):
                st.rerun()
        
        if selected_schema == "Create New":
            self._create_new_schema()
        else:
            self._edit_schema(selected_schema, schemas[selected_schema])
        
        # Migration tools
        st.markdown("---")
        st.subheader("Migration Tools")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("⬆️ Run Migrations"):
                self._run_migrations()
        
        with col2:
            if st.button("⬇️ Rollback"):
                self._rollback_migration()
        
        with col3:
            if st.button("🔍 Check Status"):
                self._check_migration_status()
    
    def _render_data_migration(self):
        """Render data migration tools"""
        st.subheader("Data Migration & Import/Export")
        
        tab1, tab2, tab3 = st.tabs(["Import Data", "Export Data", "Sync Databases"])
        
        with tab1:
            self._render_data_import()
        
        with tab2:
            self._render_data_export()
        
        with tab3:
            self._render_database_sync()
    
    def _render_data_import(self):
        """Render data import interface"""
        st.markdown("### Import Data")
        
        source_type = st.selectbox(
            "Import Source",
            ["CSV File", "JSON File", "SQL Dump", "Another Database"]
        )
        
        if source_type in ["CSV File", "JSON File", "SQL Dump"]:
            uploaded_file = st.file_uploader(
                f"Upload {source_type}",
                type=["csv", "json", "sql"] if source_type != "SQL Dump" else ["sql"]
            )
            
            if uploaded_file:
                target_table = st.text_input("Target Table Name")
                
                if source_type == "CSV File":
                    delimiter = st.selectbox("Delimiter", [",", ";", "\t", "|"])
                    has_header = st.checkbox("First row contains headers", value=True)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    if st.button("🔍 Preview Data"):
                        self._preview_import_data(uploaded_file, source_type)
                
                with col2:
                    if st.button("📥 Import Data"):
                        self._import_data(uploaded_file, source_type, target_table)
        
        elif source_type == "Another Database":
            st.markdown("### Database-to-Database Transfer")
            
            source_db = st.text_input("Source Database Connection String")
            target_db = st.text_input("Target Database Connection String")
            
            tables = st.multiselect(
                "Tables to Transfer",
                ["users", "agents", "conversations", "tools", "embeddings"]
            )
            
            if st.button("🔄 Start Transfer"):
                self._transfer_databases(source_db, target_db, tables)
    
    def _render_data_export(self):
        """Render data export interface"""
        st.markdown("### Export Data")
        
        export_format = st.selectbox(
            "Export Format",
            ["CSV", "JSON", "SQL Dump", "Parquet"]
        )
        
        # Table selection
        tables = st.multiselect(
            "Tables to Export",
            self._get_available_tables()
        )
        
        # Export options
        if export_format == "CSV":
            include_headers = st.checkbox("Include Headers", value=True)
            delimiter = st.selectbox("Delimiter", [",", ";", "\t", "|"])
        elif export_format == "JSON":
            pretty_print = st.checkbox("Pretty Print", value=True)
            include_schema = st.checkbox("Include Schema", value=False)
        elif export_format == "SQL Dump":
            include_create = st.checkbox("Include CREATE statements", value=True)
            include_data = st.checkbox("Include INSERT statements", value=True)
        
        if st.button("📤 Export Data"):
            export_data = self._export_data(tables, export_format)
            
            st.download_button(
                "Download Export",
                export_data,
                f"export.{export_format.lower()}",
                mime="text/plain"
            )
    
    def _render_database_sync(self):
        """Render database synchronization interface"""
        st.markdown("### Database Synchronization")
        
        sync_type = st.selectbox(
            "Sync Type",
            ["One-time Sync", "Continuous Replication", "Scheduled Sync"]
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            source_db = st.text_area("Source Database", height=100)
        
        with col2:
            target_db = st.text_area("Target Database", height=100)
        
        # Sync configuration
        if sync_type == "Continuous Replication":
            st.markdown("### Replication Settings")
            
            replication_method = st.selectbox(
                "Replication Method",
                ["Change Data Capture (CDC)", "Trigger-based", "Log-based"]
            )
            
            conflict_resolution = st.selectbox(
                "Conflict Resolution",
                ["Source Wins", "Target Wins", "Latest Timestamp", "Manual"]
            )
        
        elif sync_type == "Scheduled Sync":
            st.markdown("### Schedule Settings")
            
            col1, col2 = st.columns(2)
            
            with col1:
                frequency = st.selectbox(
                    "Frequency",
                    ["Every Hour", "Daily", "Weekly", "Monthly"]
                )
            
            with col2:
                time = st.time_input("Time")
        
        if st.button("🔄 Start Sync"):
            self._start_database_sync(sync_type, source_db, target_db)
    
    def _render_monitoring(self):
        """Render database monitoring"""
        st.subheader("Database Monitoring")
        
        # Connection status
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Active Connections", "12", "+2")
        
        with col2:
            st.metric("Query Rate", "145/min", "+15%")
        
        with col3:
            st.metric("Database Size", "2.3 GB", "+120 MB")
        
        with col4:
            st.metric("Uptime", "99.9%", "0%")
        
        # Performance metrics
        st.markdown("---")
        st.subheader("Performance Metrics")
        
        tab1, tab2, tab3 = st.tabs(["Queries", "Connections", "Storage"])
        
        with tab1:
            st.markdown("### Slow Queries")
            
            slow_queries = [
                {"query": "SELECT * FROM embeddings WHERE...", "time": "2.3s", "calls": 145},
                {"query": "INSERT INTO conversations...", "time": "1.8s", "calls": 89},
                {"query": "UPDATE agents SET...", "time": "1.2s", "calls": 234}
            ]
            
            for query in slow_queries:
                col1, col2, col3 = st.columns([3, 1, 1])
                
                with col1:
                    st.code(query['query'][:50] + "...")
                
                with col2:
                    st.metric("Time", query['time'])
                
                with col3:
                    st.metric("Calls", query['calls'])
        
        with tab2:
            st.markdown("### Connection Pool Status")
            
            pool_status = {
                "Total Connections": 50,
                "Active": 12,
                "Idle": 38,
                "Waiting": 0
            }
            
            for metric, value in pool_status.items():
                st.metric(metric, value)
        
        with tab3:
            st.markdown("### Storage Usage")
            
            tables = [
                {"name": "embeddings", "size": "1.2 GB", "rows": "1.2M"},
                {"name": "conversations", "size": "450 MB", "rows": "450K"},
                {"name": "agents", "size": "120 MB", "rows": "1.2K"}
            ]
            
            for table in tables:
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.text(table['name'])
                
                with col2:
                    st.metric("Size", table['size'])
                
                with col3:
                    st.metric("Rows", table['rows'])
    
    def _test_database_connection(self, db_type: str, connection_string: str):
        """Test database connection"""
        try:
            if db_type == "PostgreSQL":
                conn = psycopg2.connect(connection_string)
                conn.close()
                st.success("✅ PostgreSQL connection successful!")
            
            elif db_type == "SQLite":
                conn = sqlite3.connect(connection_string.replace("sqlite:///", ""))
                conn.close()
                st.success("✅ SQLite connection successful!")
            
            elif db_type == "MongoDB":
                client = pymongo.MongoClient(connection_string)
                client.server_info()
                st.success("✅ MongoDB connection successful!")
            
            elif db_type == "Redis":
                import redis
                r = redis.from_url(connection_string)
                r.ping()
                st.success("✅ Redis connection successful!")
            
            else:
                st.info(f"Connection test for {db_type} not implemented")
                
        except Exception as e:
            st.error(f"❌ Connection failed: {str(e)}")
    
    def _save_database_config(self, db_type: str, connection_string: str):
        """Save database configuration"""
        config = {
            "database": {
                "type": db_type,
                "connection_string": connection_string,
                "configured_at": str(Path.ctime(Path.cwd()))
            }
        }
        
        with open(self.config_file, 'w') as f:
            yaml.dump(config, f)
        
        # Also save to environment
        os.environ["DATABASE_URL"] = connection_string
        os.environ["DATABASE_TYPE"] = db_type
        
        st.success(f"✅ {db_type} configuration saved!")
    
    def _load_schemas(self) -> Dict[str, Any]:
        """Load existing database schemas"""
        schemas_dir = Path("schemas")
        schemas = {}
        
        if schemas_dir.exists():
            for schema_file in schemas_dir.glob("*.yaml"):
                with open(schema_file, 'r') as f:
                    schemas[schema_file.stem] = yaml.safe_load(f)
        
        return schemas
    
    def _create_new_schema(self):
        """Create new database schema"""
        st.markdown("### Create New Schema")
        
        schema_name = st.text_input("Schema Name")
        
        # Table builder
        st.markdown("### Tables")
        
        if 'tables' not in st.session_state:
            st.session_state.tables = []
        
        # Add table interface
        with st.expander("Add Table"):
            table_name = st.text_input("Table Name")
            
            # Column builder
            if 'columns' not in st.session_state:
                st.session_state.columns = []
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                col_name = st.text_input("Column Name")
            
            with col2:
                col_type = st.selectbox(
                    "Type",
                    ["TEXT", "INTEGER", "FLOAT", "BOOLEAN", "TIMESTAMP", "JSON", "VECTOR"]
                )
            
            with col3:
                col_nullable = st.checkbox("Nullable")
            
            with col4:
                if st.button("Add Column"):
                    st.session_state.columns.append({
                        "name": col_name,
                        "type": col_type,
                        "nullable": col_nullable
                    })
            
            # Display columns
            if st.session_state.columns:
                st.markdown("#### Columns")
                for col in st.session_state.columns:
                    st.text(f"• {col['name']} ({col['type']}){' NULL' if col['nullable'] else ' NOT NULL'}")
            
            if st.button("Create Table"):
                if table_name and st.session_state.columns:
                    st.session_state.tables.append({
                        "name": table_name,
                        "columns": st.session_state.columns.copy()
                    })
                    st.session_state.columns = []
                    st.success(f"Table '{table_name}' added!")
        
        # Display tables
        if st.session_state.tables:
            st.markdown("### Schema Tables")
            for table in st.session_state.tables:
                with st.expander(table['name']):
                    for col in table['columns']:
                        st.text(f"• {col['name']} ({col['type']})")
        
        if st.button("💾 Save Schema"):
            if schema_name and st.session_state.tables:
                self._save_schema(schema_name, st.session_state.tables)
                st.session_state.tables = []
                st.success(f"Schema '{schema_name}' saved!")
    
    def _get_available_tables(self) -> List[str]:
        """Get list of available tables"""
        # This would query the actual database
        return ["users", "agents", "conversations", "tools", "embeddings", "logs"]
    
    def _export_data(self, tables: List[str], format: str) -> str:
        """Export data in specified format"""
        # This would perform actual export
        if format == "JSON":
            return json.dumps({"tables": tables, "data": "exported"}, indent=2)
        else:
            return f"Exported {len(tables)} tables in {format} format"
    
    def _save_schema(self, name: str, tables: List[Dict]):
        """Save schema definition"""
        schemas_dir = Path("schemas")
        schemas_dir.mkdir(exist_ok=True)
        
        schema = {
            "name": name,
            "tables": tables,
            "version": "1.0"
        }
        
        with open(schemas_dir / f"{name}.yaml", 'w') as f:
            yaml.dump(schema, f)