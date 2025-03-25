# app/api/routes/queries.py
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
import uuid
import json
import psycopg2
import psycopg2.extras
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
from app.db.database import get_db
from app.db.models import DatabaseConnection, User, Conversation
from app.core.auth import get_current_user
from app.config import settings
from google import genai
from app.core.embedding_utils import retrieve_schema
import datetime

router = APIRouter(prefix="/api/queries", tags=["queries"])

class QueryRequest(BaseModel):
    connection_id: uuid.UUID
    query_text: str

class SQLResponse(BaseModel):
    sql: Optional[str] = None
    error: Optional[str] = None

class QueryResult(BaseModel):
    columns: List[str]
    rows: List[Dict[str, Any]]

class QueryResponse(BaseModel):
    query_text: str
    generated_sql: Optional[str] = None
    result: Optional[QueryResult] = None
    error: Optional[str] = None

class DateTimeEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles datetime objects."""
    def default(self, obj):
        if isinstance(obj, (datetime.datetime, datetime.date, datetime.time)):
            return obj.isoformat()
        return super().default(obj)

def get_recent_conversations(user_id: uuid.UUID, connection_id: uuid.UUID, limit: int = 3) -> list:
    """
    Get recent conversations for a user and connection in descending order (newest first).
    """
    try:
        # Use the system database instead of a separate vector DB
        db = next(get_db())
        
        conversations = db.query(Conversation).filter(
            Conversation.user_id == user_id,
            Conversation.connection_id == connection_id,
            Conversation.error.is_(None)
        ).order_by(
            Conversation.timestamp.desc()
        ).limit(limit).all()
        
        return conversations
    
    except Exception as e:
        print(f"Error retrieving conversations: {str(e)}")
        return []

def format_conversation_history(conversations: list) -> str:
    """
    Format conversations as JSON context for Gemini.
    """
    if not conversations:
        return ""
    
    formatted_conversations = []
    
    for conv in conversations:
        conversation_item = {
            "timestamp": conv.timestamp.isoformat() if conv.timestamp else None,
            "user_query": conv.user_query,
            "generated_sql": conv.generated_sql if conv.generated_sql else None
        }
        
        # Handle query results or errors
        if conv.error:
            conversation_item["error"] = conv.error
            conversation_item["query_result"] = None
        elif conv.query_result:
            try:
                conversation_item["query_result"] = json.loads(conv.query_result)
            except:
                conversation_item["query_result"] = conv.query_result
        else:
            conversation_item["query_result"] = None
            conversation_item["error"] = "No results available"
        
        formatted_conversations.append(conversation_item)
    
    # Include header text with the JSON
    return "# Recent Conversation History (Newest First)\nIf no conversations text sent means it is a new conversation\n" + json.dumps(formatted_conversations, indent=2)

# def retrieve_schema(connection_id: uuid.UUID, query_text: str, vector_db_url: str, limit: int = 10):
#     """Retrieve schema elements relevant to the query"""
#     try:
#         # Initialize Gemini client for embeddings
#         client = genai.Client(api_key=settings.GEMINI_API_KEY)
        
#         # Generate embedding for the query using Gemini
#         embedding_model = "models/embedding-001"
#         embedding_response = client.embed_content(
#             model=embedding_model,
#             content=query_text
#         )
#         query_embedding = embedding_response.embedding
        
#         # Connect to vector database
#         conn = psycopg2.connect(vector_db_url)
#         cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        
#         # First register the vector type
#         cursor.execute("SELECT NULL::vector")
        
#         # Convert to a string in the format pgvector expects
#         embedding_str = f"[{','.join(str(x) for x in query_embedding)}]"
        
#         try:
#             # Try query with connection_id filter
#             cursor.execute(
#                 """
#                 SELECT
#                     element_type,
#                     table_schema,
#                     table_name,
#                     column_name,
#                     data_type,
#                     description,
#                     1 - (embedding <=> %s::vector) as similarity
#                 FROM
#                     schema_embeddings
#                 WHERE
#                     connection_id = %s
#                 ORDER BY
#                     embedding <=> %s::vector
#                 LIMIT %s;
#                 """,
#                 (embedding_str, str(connection_id), embedding_str, limit)
#             )
            
#             results = [dict(row) for row in cursor.fetchall()]
            
#             # If we got results, return them
#             if results:
#                 cursor.close()
#                 conn.close()
#                 print(f"Retrieved {len(results)} schema elements for query: {query_text}")
#                 return results
                
#             # If no results with connection_id, try without it (fallback)
#             cursor.execute(
#                 """
#                 SELECT
#                     element_type,
#                     table_schema,
#                     table_name,
#                     column_name,
#                     data_type,
#                     description,
#                     1 - (embedding <=> %s::vector) as similarity
#                 FROM
#                     schema_embeddings
#                 ORDER BY
#                     embedding <=> %s::vector
#                 LIMIT %s;
#                 """,
#                 (embedding_str, embedding_str, limit)
#             )
            
#             results = [dict(row) for row in cursor.fetchall()]
            
#         except psycopg2.Error as e:
#             # If there's any database error, try the query without connection_id filter
#             print(f"Database error in schema retrieval: {str(e)}, trying without connection_id filter")
#             cursor.execute(
#                 """
#                 SELECT
#                     element_type,
#                     table_schema,
#                     table_name,
#                     column_name,
#                     data_type,
#                     description,
#                     1 - (embedding <=> %s::vector) as similarity
#                 FROM
#                     schema_embeddings
#                 ORDER BY
#                     embedding <=> %s::vector
#                 LIMIT %s;
#                 """,
#                 (embedding_str, embedding_str, limit)
#             )
            
#             results = [dict(row) for row in cursor.fetchall()]
        
#         cursor.close()
#         conn.close()
        
#         print(f"Retrieved {len(results)} schema elements for query: {query_text}")
#         return results
        
#     except Exception as e:
#         print(f"Error in retrieve_schema: {str(e)}")
#         # Return an empty list in case of any error
#         return []

def format_schema_for_llm(search_results):
    """Format the schema search results into a structured representation for the LLM."""
    # Group results by table
    tables = {}
    
    # First, collect all tables
    for item in search_results:
        if item['element_type'] == 'table':
            table_key = f"{item['table_schema']}.{item['table_name']}"
            if table_key not in tables:
                tables[table_key] = {
                    'schema': item['table_schema'],
                    'name': item['table_name'],
                    'columns': [],
                    'similarity': item['similarity']
                }
    
    # Then, add all columns to their respective tables
    for item in search_results:
        if item['element_type'] == 'column':
            table_key = f"{item['table_schema']}.{item['table_name']}"
            # If this column's table wasn't in the top results, add it
            if table_key not in tables:
                tables[table_key] = {
                    'schema': item['table_schema'],
                    'name': item['table_name'],
                    'columns': [],
                    'similarity': 0
                }
            
            # Add the column
            tables[table_key]['columns'].append({
                'name': item['column_name'],
                'type': item['data_type'],
                'description': item['description'],
                'similarity': item['similarity']
            })
    
    # Format into a structured schema document
    schema_doc = "# Database Schema for Query\n\n"
    
    # Sort tables by similarity
    sorted_tables = sorted(tables.values(), key=lambda x: x['similarity'], reverse=True)
    
    for table in sorted_tables:
        schema_doc += f"## Table: {table['schema']}.{table['name']}\n\n"
        
        # Sort columns by similarity
        sorted_columns = sorted(table['columns'], key=lambda x: x['similarity'], reverse=True)
        
        if sorted_columns:
            schema_doc += "| Column | Type | Description |\n"
            schema_doc += "|--------|------|-------------|\n"
            
            for col in sorted_columns:
                schema_doc += f"| {col['name']} | {col['type']} | {col['description']} |\n"
            
            schema_doc += "\n"
    
    return schema_doc

def generate_sql_with_gemini(query_text: str, schema_context: str, user_id: uuid.UUID, connection_id: uuid.UUID) -> SQLResponse:
    """Generate SQL using Gemini based on the query and schema context"""
    # Configure Gemini - using the pattern from your reference code
    client = genai.Client(api_key=settings.GEMINI_API_KEY)
    # Get conversation history
    recent_conversations = get_recent_conversations(user_id, connection_id)
    conversation_context = format_conversation_history(recent_conversations)
    
    prompt = f"""
You are an expert SQL assistant. You generate PostgreSQL queries based on the user's question and the provided database schema.

# Database Schema
{schema_context}

# User Question
{query_text}

# Requirements
- Generate a single PostgreSQL query that answers the user's question
- Return ONLY a valid SQL query with no explanation or commentary
- Only perform READ operations (SELECT statements)
- If the user's request implies any data modification (INSERT, UPDATE, DELETE, etc.), respond with an error message
- Use valid PostgreSQL syntax
- Make sure to include proper JOINs when data spans multiple tables
- Include proper WHERE clauses that correspond to the user's requirements
- Use appropriate aliases for clarity when joining multiple tables
- you need to respond in json format the way we want in sql we only want sql code and no more text and in error stating why you cannot produce sql based on the teh limitations
- remember that you sometimes will receive context, that is only to answer follow up questions or any question with less information than you can take the help of teh context (previous done conversation) to build sql query. If the prompt is clear and you can make sql query than do not consider context 

{conversation_context}
"""
    
    try:
        # Generate response
        response = client.models.generate_content(
            model='gemini-2.0-flash',
            contents=prompt,
            config={
                'response_mime_type': 'application/json',
                'response_schema': list[SQLResponse],
            },
        )

        sql_responses: list[SQLResponse] = response.parsed
        
        if sql_responses and len(sql_responses) > 0:
            sql_response = sql_responses[0]
            
            # Check for write operations
            if sql_response.sql and any(keyword in sql_response.sql.upper() for keyword in 
                                    ["INSERT", "UPDATE", "DELETE", "DROP", "CREATE", "ALTER", "TRUNCATE"]):
                return SQLResponse(error="Request includes write operations which are not permitted")
            
            return sql_response
        else:
            return SQLResponse(error="No SQL was generated")
    
    except Exception as e:
        return SQLResponse(error=f"Failed to generate SQL: {str(e)}")

def execute_sql_query(sql: str, target_db_url: str) -> tuple:
    """Execute SQL query and return results"""
    try:
        conn = psycopg2.connect(target_db_url)
        cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        
        cursor.execute(sql)
        
        # Get column names
        column_names = [desc[0] for desc in cursor.description] if cursor.description else []
        
        # Fetch all rows
        rows = cursor.fetchall()
        
        # Convert rows to list of dicts
        result_rows = []
        for row in rows:
            result_rows.append({column: row[column] for column in column_names})
        
        cursor.close()
        conn.close()
        
        return column_names, result_rows, None
    
    except Exception as e:
        return [], [], str(e)

@router.post("/", response_model=QueryResponse)
async def process_query(
    query_request: QueryRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Process natural language query and return SQL query and results"""
    # Check if connection exists and belongs to user
    connection = db.query(DatabaseConnection).filter(
        DatabaseConnection.id == query_request.connection_id,
        DatabaseConnection.user_id == current_user.id
    ).first()
    
    if not connection:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Connection not found"
        )
    
    try:
        # Retrieve relevant schema - now with better error handling
        schema_results = retrieve_schema(
            connection_id=connection.id,
            query_text=query_request.query_text,
            vector_db_url=connection.vector_db_url
        )
        
        # Format schema for LLM (even if results are empty)
        schema_context = format_schema_for_llm(schema_results)
        
        # Generate SQL using Gemini with conversation context
        sql_response = generate_sql_with_gemini(
            query_text=query_request.query_text,
            schema_context=schema_context,
            user_id=current_user.id,
            connection_id=connection.id
        )
        
        # Create a new conversation entry
        conversation = Conversation(
            id=uuid.uuid4(),
            user_id=current_user.id,
            connection_id=connection.id,
            user_query=query_request.query_text,
            generated_sql=sql_response.sql,
            error=sql_response.error
        )
        
        if sql_response.error:
            # Save conversation with error
            db.add(conversation)
            db.commit()
            
            return QueryResponse(
                query_text=query_request.query_text,
                error=sql_response.error
            )
        
        # Execute the SQL
        columns, rows, error = execute_sql_query(sql_response.sql, connection.target_db_url)
        
        if error:
            # Save conversation with execution error
            conversation.error = error
            db.add(conversation)
            db.commit()
            
            return QueryResponse(
                query_text=query_request.query_text,
                generated_sql=sql_response.sql,
                error=error
            )
        
        try:
            # Make sure rows is always a list - fix for the validation error
            if not isinstance(rows, list):
                rows = [rows]
            
            # Store the result as a JSON string for retrieval later
            conversation.query_result = json.dumps(rows, cls=DateTimeEncoder)
            db.add(conversation)
            db.commit()
            
            return QueryResponse(
                query_text=query_request.query_text,
                generated_sql=sql_response.sql,
                result=QueryResult(columns=columns, rows=rows)
            )
        except Exception as e:
            print(f"Error formatting result: {str(e)}")
            conversation.error = f"Error formatting result: {str(e)}"
            db.add(conversation)
            db.commit()
            
            return QueryResponse(
                query_text=query_request.query_text,
                generated_sql=sql_response.sql,
                error=f"Error formatting result: {str(e)}"
            )
    
    except Exception as e:
        # Log the full error for debugging
        print(f"Error processing query: {str(e)}")
        
        return QueryResponse(
            query_text=query_request.query_text,
            error=f"Error processing query: {str(e)}"
        )