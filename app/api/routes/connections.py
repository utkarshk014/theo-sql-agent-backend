# app/api/routes/connections.py
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
import uuid
import psycopg2
import psycopg2.extras
import google.generativeai as genai
from typing import List, Dict, Any
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
    
    class Config:
        from_attributes = True

class ExtendedConnectionResponse(ConnectionResponse):
    schema_extraction: Dict[str, Any] = None

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
        genai.configure(api_key=settings.GEMINI_API_KEY)
        
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
        vector_cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS schema_embeddings (
          id SERIAL PRIMARY KEY,
          element_type TEXT,
          table_schema TEXT, 
          table_name TEXT,
          column_name TEXT,
          data_type TEXT,
          description TEXT,
          embedding vector({embedding_dimension}),
        );
        
        -- Create index for faster similarity search if it doesn't exist
        CREATE INDEX IF NOT EXISTS schema_embeddings_embedding_idx 
        ON schema_embeddings 
        USING ivfflat (embedding vector_cosine_ops)
        WITH (lists = 100);
        """)
        
        # Clear existing embeddings for this connection
        vector_cursor.execute(
            "DELETE FROM schema_embeddings",
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
                    item['embedding']
                )
            )
        
        vector_conn.commit()
        vector_cursor.close()
        vector_conn.close()
        
        db.commit()
        
        return {
            "status": "success", 
            "message": f"Successfully stored {len(all_schema_items)} schema embeddings",
            "tables_processed": len(tables),
            "embeddings_created": len(all_schema_items)
        }
    
    except Exception as e:
        return {"status": "error", "message": f"Error extracting schema: {str(e)}"}

@router.post("/", response_model=ExtendedConnectionResponse, status_code=status.HTTP_201_CREATED)
def create_connection(
    connection: ConnectionCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Create a new database connection with validation and schema extraction"""
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
    )
    
    db.add(db_connection)
    db.commit()
    db.refresh(db_connection)
    
    # Extract and embed schema
    schema_result = extract_and_embed_schema(db_connection.id, current_user.id, db)
    
    # Refresh the connection to get updated is_schema_extracted value
    db.refresh(db_connection)
    
    # Return extended response with extraction details
    return ExtendedConnectionResponse(
        id=db_connection.id,
        name=db_connection.name,
        vector_db_url=db_connection.vector_db_url,
        target_db_url=db_connection.target_db_url,
        schema_extraction=schema_result
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

@router.post("/{connection_id}/extract", response_model=Dict[str, Any])
def extract_schema(
    connection_id: uuid.UUID,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Extract schema from a database connection (kept for backward compatibility)"""
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
    
    # Extract schema
    result = extract_and_embed_schema(connection_id, current_user.id, db)
    
    return {
        "connection_id": connection_id,
        "status": result["status"],
        "message": result["message"]
    }

# # app/api/routes/connections.py
# from fastapi import APIRouter, Depends, HTTPException, status
# from sqlalchemy.orm import Session
# import uuid
# from typing import List
# from pydantic import BaseModel
# from app.db.database import get_db
# from app.db.models import DatabaseConnection, User
# from app.core.auth import get_current_user

# router = APIRouter(prefix="/api/connections", tags=["connections"])

# # Pydantic models for request/response
# class ConnectionCreate(BaseModel):
#     name: str
#     vector_db_url: str
#     target_db_url: str

# class ConnectionResponse(BaseModel):
#     id: uuid.UUID
#     name: str
#     vector_db_url: str
#     target_db_url: str
    
#     class Config:
#         from_attributes = True

# @router.post("/", response_model=ConnectionResponse, status_code=status.HTTP_201_CREATED)
# def create_connection(
#     connection: ConnectionCreate,
#     db: Session = Depends(get_db),
#     current_user: User = Depends(get_current_user)
# ):
#     """Create a new database connection"""
#     db_connection = DatabaseConnection(
#         id=uuid.uuid4(),
#         user_id=current_user.id,
#         name=connection.name,
#         vector_db_url=connection.vector_db_url,
#         target_db_url=connection.target_db_url
#     )
    
#     db.add(db_connection)
#     db.commit()
#     db.refresh(db_connection)
    
#     return db_connection

# @router.get("/", response_model=List[ConnectionResponse])
# def get_connections(
#     db: Session = Depends(get_db),
#     current_user: User = Depends(get_current_user)
# ):
#     """Get all database connections for the current user"""
#     connections = db.query(DatabaseConnection).filter(
#         DatabaseConnection.user_id == current_user.id
#     ).all()
    
#     return connections

# @router.get("/{connection_id}", response_model=ConnectionResponse)
# def get_connection(
#     connection_id: uuid.UUID,
#     db: Session = Depends(get_db),
#     current_user: User = Depends(get_current_user)
# ):
#     """Get a specific database connection"""
#     connection = db.query(DatabaseConnection).filter(
#         DatabaseConnection.id == connection_id,
#         DatabaseConnection.user_id == current_user.id
#     ).first()
    
#     if not connection:
#         raise HTTPException(
#             status_code=status.HTTP_404_NOT_FOUND,
#             detail="Connection not found"
#         )
    
#     return connection

# @router.delete("/{connection_id}", status_code=status.HTTP_204_NO_CONTENT)
# def delete_connection(
#     connection_id: uuid.UUID,
#     db: Session = Depends(get_db),
#     current_user: User = Depends(get_current_user)
# ):
#     """Delete a database connection"""
#     connection = db.query(DatabaseConnection).filter(
#         DatabaseConnection.id == connection_id,
#         DatabaseConnection.user_id == current_user.id
#     ).first()
    
#     if not connection:
#         raise HTTPException(
#             status_code=status.HTTP_404_NOT_FOUND,
#             detail="Connection not found"
#         )
    
#     db.delete(connection)
#     db.commit()
    
#     return None