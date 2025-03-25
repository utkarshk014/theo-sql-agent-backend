# app/core/embedding_utils.py
import google.generativeai as genai
import psycopg2
import psycopg2.extras
import uuid
from typing import List, Dict, Any
from app.config import settings

def generate_embeddings(text: str) -> List[float]:
    """
    Generate embeddings for the given text using the Gemini API
    """
    # Configure Gemini API
    genai.configure(api_key=settings.GEMINI_API_KEY)
    
    # Generate embedding using Gemini
    embedding_model = "models/embedding-001"
    response = genai.embed_content(
        model=embedding_model,
        content=text
    )
    
    # Return the embedding vector
    return response["embedding"]

def retrieve_schema(connection_id: uuid.UUID, query_text: str, vector_db_url: str, limit: int = 10) -> List[Dict[str, Any]]:
    """
    Retrieve schema elements relevant to the query using vector similarity
    """
    try:
        # Generate embedding for the query
        query_embedding = generate_embeddings(query_text)
        
        # Connect to vector database
        conn = psycopg2.connect(vector_db_url)
        cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        
        # First register the vector type
        cursor.execute("SELECT NULL::vector")
        
        # Convert to a string in the format pgvector expects
        embedding_str = f"[{','.join(str(x) for x in query_embedding)}]"
        
        try:
            # Try query with connection_id filter
            cursor.execute(
                """
                SELECT
                    element_type,
                    table_schema,
                    table_name,
                    column_name,
                    data_type,
                    description,
                    1 - (embedding <=> %s::vector) as similarity
                FROM
                    schema_embeddings
                ORDER BY
                    embedding <=> %s::vector
                LIMIT %s;
                """,
                (embedding_str, embedding_str, limit)
            )
            
            results = [dict(row) for row in cursor.fetchall()]
            
            # If we got results, return them
            if results:
                cursor.close()
                conn.close()
                print(f"Retrieved {len(results)} schema elements for query: {query_text}")
                return results
                
            # If no results with connection_id, try without it (fallback)
            cursor.execute(
                """
                SELECT
                    element_type,
                    table_schema,
                    table_name,
                    column_name,
                    data_type,
                    description,
                    1 - (embedding <=> %s::vector) as similarity
                FROM
                    schema_embeddings
                ORDER BY
                    embedding <=> %s::vector
                LIMIT %s;
                """,
                (embedding_str, embedding_str, limit)
            )
            
            results = [dict(row) for row in cursor.fetchall()]
            
        except psycopg2.Error as e:
            # If there's any database error, try the query without connection_id filter
            print(f"Database error in schema retrieval: {str(e)}, trying without connection_id filter")
            cursor.execute(
                """
                SELECT
                    element_type,
                    table_schema,
                    table_name,
                    column_name,
                    data_type,
                    description,
                    1 - (embedding <=> %s::vector) as similarity
                FROM
                    schema_embeddings
                ORDER BY
                    embedding <=> %s::vector
                LIMIT %s;
                """,
                (embedding_str, embedding_str, limit)
            )
            
            results = [dict(row) for row in cursor.fetchall()]
        
        cursor.close()
        conn.close()
        
        print(f"Retrieved {len(results)} schema elements for query: {query_text}")
        return results
        
    except Exception as e:
        print(f"Error in retrieve_schema: {str(e)}")
        # Return an empty list in case of any error
        return []