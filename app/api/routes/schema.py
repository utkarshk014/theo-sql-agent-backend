# app/api/routes/schema.py
from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks
from sqlalchemy.orm import Session
import uuid
import psycopg2
import psycopg2.extras
import google.generativeai as genai
from typing import Dict, Any
from pydantic import BaseModel
from app.db.database import get_db
from app.db.models import DatabaseConnection, User
from app.core.auth import get_current_user
from app.config import settings

router = APIRouter(prefix="/api/schema", tags=["schema"])

class SchemaExtractionResponse(BaseModel):
    connection_id: uuid.UUID
    status: str
    message: str

def extract_and_embed_schema(connection_id: uuid.UUID, user_id: uuid.UUID, db: Session):
    """Extract schema from source database and store embeddings in vector database"""
    # Get the connection details
    connection = db.query(DatabaseConnection).filter(
        DatabaseConnection.id == connection_id,
        DatabaseConnection.user_id == user_id
    ).first()
    
    if not connection:
        return {"status": "error", "message": "Connection not found"}
    
    try:
        # Connect to the source database (user's target DB)
        source_conn = psycopg2.connect(connection.target_db_url)
        source_cursor = source_conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        
        # Query to extract schema information
        query = """
        SELECT 
            t.table_schema,
            t.table_name,
            c.column_name,
            c.data_type,
            c.is_nullable
        FROM information_schema.tables t
        JOIN information_schema.columns c
            ON t.table_catalog = c.table_catalog
            AND t.table_schema = c.table_schema
            AND t.table_name = c.table_name
        WHERE t.table_schema NOT IN ('pg_catalog', 'information_schema')
            AND t.table_type = 'BASE TABLE'
        ORDER BY t.table_schema, t.table_name, c.ordinal_position;
        """
        
        source_cursor.execute(query)
        schema_rows = source_cursor.fetchall()
        
        # Clean up source database connection
        source_cursor.close()
        source_conn.close()
        
        # Group by table
        tables = {}
        for row in schema_rows:
            table_key = f"{row['table_schema']}.{row['table_name']}"
            if table_key not in tables:
                tables[table_key] = []
            tables[table_key].append(dict(row))
        
        # Configure Gemini API
        genai.configure(api_key=settings.GEMINI_API_KEY)  # Replace with proper config
        
        embedding_model = "models/embedding-001"
        all_schema_items = []
        
        # Process each table
        for table_key, columns in tables.items():
            # Create table description
            table_desc = f"Table {table_key} with columns: {', '.join(col['column_name'] for col in columns)}"
            
            # Generate embedding using Gemini
            table_embedding = genai.embed_content(
                model=embedding_model,
                content=table_desc
            )["embedding"]
            
            all_schema_items.append({
                'type': 'table',
                'schema': columns[0]['table_schema'],
                'table': columns[0]['table_name'],
                'column': None,
                'data_type': None,
                'description': table_desc,
                'embedding': table_embedding,
                'connection_id': str(connection_id)  # Add connection ID to track ownership
            })
            
            # Process each column
            for col in columns:
                col_desc = (
                    f"Column {col['column_name']} in table {table_key} "
                    f"of type {col['data_type']}"
                    f"{', nullable' if col['is_nullable'] == 'YES' else ', not nullable'}"
                )
                
                # Generate embedding using Gemini
                col_embedding = genai.embed_content(
                    model=embedding_model,
                    content=col_desc
                )["embedding"]
                
                all_schema_items.append({
                    'type': 'column',
                    'schema': col['table_schema'],
                    'table': col['table_name'],
                    'column': col['column_name'],
                    'data_type': col['data_type'],
                    'description': col_desc,
                    'embedding': col_embedding,
                    'connection_id': str(connection_id)  # Add connection ID to track ownership
                })
        
        # Connect to vector database
        vector_conn = psycopg2.connect(connection.vector_db_url)
        vector_cursor = vector_conn.cursor()

        sample_embedding = genai.embed_content(
            model=embedding_model,
            content="Sample text for dimension detection"
        )["embedding"]

        embedding_dimension = len(sample_embedding)
        
        # Ensure the vector extension is enabled
        vector_cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        
        # Create the schema_embeddings table if it doesn't exist
        # Note: We add connection_id to track ownership
        vector_cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS schema_embeddings (
          id SERIAL PRIMARY KEY,
          element_type TEXT,
          table_schema TEXT, 
          table_name TEXT,
          column_name TEXT,
          data_type TEXT,
          description TEXT,
          embedding vector({embedding_dimension}), -- Dimension for Gemini embeddings
          connection_id TEXT
        );
        
        -- Create index for faster similarity search if it doesn't exist
        CREATE INDEX IF NOT EXISTS schema_embeddings_embedding_idx 
        ON schema_embeddings 
        USING ivfflat (embedding vector_cosine_ops)
        WITH (lists = 100);
        """)
        
        # Clear existing embeddings for this connection
        vector_cursor.execute(
            "DELETE FROM schema_embeddings WHERE connection_id = %s",
            (str(connection_id),)
        )
        
        # Store embeddings in vector database
        for item in all_schema_items:
            vector_cursor.execute(
                """
                INSERT INTO schema_embeddings
                (element_type, table_schema, table_name, column_name, data_type, description, embedding, connection_id)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    item['type'],
                    item['schema'],
                    item['table'],
                    item['column'],
                    item['data_type'],
                    item['description'],
                    item['embedding'],
                    item['connection_id']
                )
            )
        
        vector_conn.commit()
        vector_cursor.close()
        vector_conn.close()
        
        # Update the connection in our database
        connection.is_schema_extracted = True
        db.commit()
        
        return {"status": "success", "message": f"Successfully stored {len(all_schema_items)} schema embeddings"}
    
    except Exception as e:
        return {"status": "error", "message": f"Error extracting schema: {str(e)}"}

@router.post("/{connection_id}/extract", response_model=SchemaExtractionResponse)
def extract_schema(
    connection_id: uuid.UUID,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Extract schema from a database connection"""
    # Check if connection exists
    connection = db.query(DatabaseConnection).filter(
        DatabaseConnection.id == connection_id,
        DatabaseConnection.user_id == current_user.id
    ).first()
    
    if not connection:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Connection not found"
        )
    
    # Extract schema (synchronously)
    result = extract_and_embed_schema(connection_id, current_user.id, db)
    
    return {
        "connection_id": connection_id,
        "status": result["status"],
        "message": result["message"]
    }