from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks
from sqlalchemy.orm import Session
import uuid
import psycopg2
import psycopg2.extras
import google.generativeai as genai
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
from app.db.database import get_db
from app.db.models import DatabaseConnection, User
from app.core.auth import get_current_user
from app.config import settings

router = APIRouter(prefix="/api/connections", tags=["connections"])

# Pydantic models for request/response
class ConnectionCreate(BaseModel):
    name: str
    vector_db_url: str
    target_db_url: str

class ConnectionResponse(BaseModel):
    id: uuid.UUID
    name: str
    vector_db_url: str
    target_db_url: str
    schema_status: Optional[str] = None
    
    class Config:
        from_attributes = True

class ExtendedConnectionResponse(ConnectionResponse):
    schema_extraction: Dict[str, Any] = None

class SchemaStatusResponse(BaseModel):
    connection_id: uuid.UUID
    name: str
    status: str
    error: Optional[str] = None

def validate_db_url(db_url: str, db_type: str) -> Dict[str, Any]:
    """Validate if database URL is legitimate by attempting to connect"""
    try:
        conn = psycopg2.connect(db_url)
        conn.close()
        return {"status": "success", "message": f"{db_type} connection successful"}
    except Exception as e:
        return {"status": "error", "message": f"{db_type} connection failed: {str(e)}"}

def extract_and_embed_schema(connection_id: uuid.UUID, user_id: uuid.UUID, db: Session) -> Dict[str, Any]:
    """Extract schema from source database and store embeddings in vector database"""
    print(f"[SCHEMA EXTRACTION] Starting for connection ID: {connection_id}")
    # Get the connection details
    connection = db.query(DatabaseConnection).filter(
        DatabaseConnection.id == connection_id,
        DatabaseConnection.user_id == user_id
    ).first()
    
    if not connection:
        print(f"[SCHEMA EXTRACTION] ERROR: Connection {connection_id} not found")
        return {"status": "error", "message": "Connection not found"}
    
    try:
        print(f"[SCHEMA EXTRACTION] Connecting to source database: {connection.target_db_url[:20]}...")
        # Connect to the source database (user's target DB)
        source_conn = psycopg2.connect(connection.target_db_url)
        source_cursor = source_conn.cursor(cursor_factory=psycopg2.extras.DictCursor)

        print(f"[SCHEMA EXTRACTION] Connected to source DB. Extracting schema...")
        
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
        print(f"[SCHEMA EXTRACTION] Fetched {len(schema_rows)} rows from schema")

        # Clean up source database connection
        source_cursor.close()
        source_conn.close()
        print(f"[SCHEMA EXTRACTION] Closed source DB connection")

        # Group by table
        tables = {}
        for row in schema_rows:
            table_key = f"{row['table_schema']}.{row['table_name']}"
            if table_key not in tables:
                tables[table_key] = []
            tables[table_key].append(dict(row))
        
        print(f"[SCHEMA EXTRACTION] Grouped into {len(tables)} tables")
        print(f"[SCHEMA EXTRACTION] Configuring Gemini API")
        # Configure Gemini API
        genai.configure(api_key=settings.GEMINI_API_KEY)
        
        embedding_model = "models/embedding-001"
        all_schema_items = []

        print(f"[SCHEMA EXTRACTION] Starting to process tables and generate embeddings")
        
        # Process each table
        for table_key, columns in enumerate(tables.items()):
            print(f"[SCHEMA EXTRACTION] Processing table: {table_key}")
            table_key, columns = columns  # Unpack the tuple
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
                })
        print(f"[SCHEMA EXTRACTION] Generated {len(all_schema_items)} embeddings. Connecting to vector DB...")
        # Connect to vector database
        vector_conn = psycopg2.connect(connection.vector_db_url)
        vector_cursor = vector_conn.cursor()
        print(f"[SCHEMA EXTRACTION] Connected to vector DB")

        sample_embedding = genai.embed_content(
            model=embedding_model,
            content="Sample text for dimension detection"
        )["embedding"]

        embedding_dimension = len(sample_embedding)
        print(f"[SCHEMA EXTRACTION] Embedding dimension: {embedding_dimension}")
        
        # Ensure the vector extension is enabled
        print(f"[SCHEMA EXTRACTION] Creating vector extension if needed")
        vector_cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        print(f"[SCHEMA EXTRACTION] Creating schema_embeddings table if needed")
        # Create the schema_embeddings table if it doesn't exist
        vector_cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS schema_embeddings (
          id SERIAL PRIMARY KEY,
          element_type TEXT,
          table_schema TEXT, 
          table_name TEXT,
          column_name TEXT,
          data_type TEXT,
          description TEXT,
          embedding vector({embedding_dimension})
        );
        """)
        
        # Create index for faster similarity search if it doesn't exist
        try:
            print(f"[SCHEMA EXTRACTION] Creating vector index")
            vector_cursor.execute(f"""
            CREATE INDEX IF NOT EXISTS schema_embeddings_embedding_idx 
            ON schema_embeddings 
            USING ivfflat (embedding vector_cosine_ops)
            WITH (lists = 100);
            """)
        except Exception as e:
            # Log the error but continue - index is for performance only
            print(f"Error creating index: {str(e)}")
        
        # Clear existing embeddings for this connection
        vector_cursor.execute(
            "DELETE FROM schema_embeddings"
        )
        print(f"[SCHEMA EXTRACTION] Storing {len(all_schema_items)} embeddings in vector DB")
        # Store embeddings in vector database
        for item in all_schema_items:
            vector_cursor.execute(
                """
                INSERT INTO schema_embeddings
                (element_type, table_schema, table_name, column_name, data_type, description, embedding)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    item['type'],
                    item['schema'],
                    item['table'],
                    item['column'],
                    item['data_type'],
                    item['description'],
                    item['embedding']
                )
            )
        print(f"[SCHEMA EXTRACTION] Committing vector DB changes")
        vector_conn.commit()
        vector_cursor.close()
        vector_conn.close()
        print(f"[SCHEMA EXTRACTION] Closed vector DB connection")
        result = {
            "status": "success", 
            "message": f"Successfully stored {len(all_schema_items)} schema embeddings",
            "tables_processed": len(tables),
            "embeddings_created": len(all_schema_items)
        }
        
        # Update connection in database
        connection.schema_status = "completed"
        connection.schema_details = {
            "tables_processed": len(tables),
            "embeddings_created": len(all_schema_items)
        }
        db.commit()
        
        return result
    
    except Exception as e:
        error_message = f"Error extracting schema: {str(e)}"
        
        # Update connection in database
        connection.schema_status = "failed"
        connection.schema_error = error_message
        db.commit()
        
        return {"status": "error", "message": error_message}

def extract_and_embed_schema_background(connection_id: uuid.UUID, user_id: uuid.UUID):
    """Background task for schema extraction"""
    print(f"[BACKGROUND TASK] Starting schema extraction for connection {connection_id}")
    # Create a new database session for this background task
    try:
        db = next(get_db())
        print(f"[BACKGROUND TASK] Fetching connection {connection_id}")
        # Get the connection
        connection = db.query(DatabaseConnection).filter(
            DatabaseConnection.id == connection_id,
            DatabaseConnection.user_id == user_id
        ).first()
        
        if not connection:
            print(f"Connection {connection_id} not found for background task")
            return
        
        # Update connection status
        print(f"[BACKGROUND TASK] Updating connection status to processing")
        connection.schema_status = "processing"
        db.commit()
        
        # Run the extraction
        result = extract_and_embed_schema(connection_id, user_id, db)
        print(f"[BACKGROUND TASK] Extraction completed with status: {result['status']}")
        db.close()
        
    except Exception as e:
        error_message = f"Error in background task: {str(e)}"
        print(error_message)
    
    finally:
        print(f"[BACKGROUND TASK] Closing database session")

@router.post("/", response_model=ExtendedConnectionResponse, status_code=status.HTTP_201_CREATED)
def create_connection(
    connection: ConnectionCreate,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    print("[API] create_connection endpoint called")
    """Create a new database connection with validation and background schema extraction"""
    # Validate both database URLs
    vector_db_validation = validate_db_url(connection.vector_db_url, "Vector database")
    target_db_validation = validate_db_url(connection.target_db_url, "Target database")
    
    # If either validation fails, return error with details
    if vector_db_validation["status"] == "error" or target_db_validation["status"] == "error":
        error_message = ""
        if vector_db_validation["status"] == "error":
            error_message += vector_db_validation["message"] + " "
        if target_db_validation["status"] == "error":
            error_message += target_db_validation["message"]
            
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=error_message.strip()
        )
    
    # Create the database connection
    db_connection = DatabaseConnection(
        id=uuid.uuid4(),
        user_id=current_user.id,
        name=connection.name,
        vector_db_url=connection.vector_db_url,
        target_db_url=connection.target_db_url,
        schema_status="pending"
    )
    
    db.add(db_connection)
    db.commit()
    db.refresh(db_connection)
    
    # Schedule schema extraction as a background task
    background_tasks.add_task(
        extract_and_embed_schema_background,
        connection_id=db_connection.id,
        user_id=current_user.id
    )
    
    # Return the connection immediately with pending status
    return ExtendedConnectionResponse(
        id=db_connection.id,
        name=db_connection.name,
        vector_db_url=db_connection.vector_db_url,
        target_db_url=db_connection.target_db_url,
        schema_status="pending",
        schema_extraction={"status": "pending", "message": "Schema extraction started in background"}
    )

@router.get("/", response_model=List[ConnectionResponse])
def get_connections(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get all database connections for the current user"""
    connections = db.query(DatabaseConnection).filter(
        DatabaseConnection.user_id == current_user.id
    ).all()
    
    return connections

@router.get("/{connection_id}", response_model=ConnectionResponse)
def get_connection(
    connection_id: uuid.UUID,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get a specific database connection"""
    connection = db.query(DatabaseConnection).filter(
        DatabaseConnection.id == connection_id,
        DatabaseConnection.user_id == current_user.id
    ).first()
    
    if not connection:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Connection not found"
        )
    
    return connection

@router.get("/{connection_id}/status", response_model=SchemaStatusResponse)
def get_extraction_status(
    connection_id: uuid.UUID,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get the schema extraction status for a specific connection"""
    connection = db.query(DatabaseConnection).filter(
        DatabaseConnection.id == connection_id,
        DatabaseConnection.user_id == current_user.id
    ).first()
    
    if not connection:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Connection not found"
        )
    
    return SchemaStatusResponse(
        connection_id=connection.id,
        name=connection.name,
        status=connection.schema_status or "unknown",
        error=connection.schema_error
    )

@router.delete("/{connection_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_connection(
    connection_id: uuid.UUID,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Delete a database connection"""
    connection = db.query(DatabaseConnection).filter(
        DatabaseConnection.id == connection_id,
        DatabaseConnection.user_id == current_user.id
    ).first()
    
    if not connection:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Connection not found"
        )
    
    db.delete(connection)
    db.commit()
    
    return None

@router.post("/{connection_id}/extract", response_model=SchemaStatusResponse)
def extract_schema(
    connection_id: uuid.UUID,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Start schema extraction in the background for an existing connection"""
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
    
    # Update connection status
    connection.schema_status = "pending"
    connection.schema_error = None
    connection.schema_details = None
    db.commit()
    
    # Schedule schema extraction as a background task
    background_tasks.add_task(
        extract_and_embed_schema_background,
        connection_id=connection_id,
        user_id=current_user.id
    )
    
    return SchemaStatusResponse(
        connection_id=connection_id,
        name=connection.name,
        status="pending"
    )