# app/api/routes/conversations.py
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
import uuid
import json
from typing import List, Optional, Dict, Any
from pydantic import BaseModel
from app.db.database import get_db
from app.db.models import Conversation, User, DatabaseConnection
from app.core.auth import get_current_user

router = APIRouter(prefix="/api/conversations", tags=["conversations"])

class ConversationResponse(BaseModel):
    id: uuid.UUID
    connection_id: uuid.UUID
    timestamp: str
    user_query: str
    generated_sql: Optional[str] = None
    query_result: Optional[List[Dict[str, Any]]] = None
    error: Optional[str] = None

def get_user_connection(db: Session, user_id: uuid.UUID) -> DatabaseConnection:
    """Helper function to get the user's connection (assuming one connection per user)"""
    connection = db.query(DatabaseConnection).filter(
        DatabaseConnection.user_id == user_id
    ).first()
    
    if not connection:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No database connection found for this user. Please set up a connection first."
        )
    
    return connection

@router.get("/", response_model=List[ConversationResponse])
async def get_conversations(
    limit: int = 25,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get conversation history for the current user"""
    # Get the user's connection 
    connection = get_user_connection(db, current_user.id)
    
    # Query conversations for this user and connection
    conversations = db.query(Conversation).filter(
        Conversation.user_id == current_user.id,
        Conversation.connection_id == connection.id
    ).order_by(Conversation.timestamp.desc()).limit(limit).all()
    
    # Format response
    result = []
    for conv in conversations:
        # Parse query_result if it exists
        query_result = None
        if conv.query_result is not None:
            try:
                # Parse the JSON string into a Python object
                query_result = json.loads(conv.query_result)
            except json.JSONDecodeError:
                query_result = None
        
        result.append({
            "id": conv.id,
            "connection_id": conv.connection_id,
            "timestamp": conv.timestamp.isoformat(),
            "user_query": conv.user_query,
            "generated_sql": conv.generated_sql,
            "query_result": query_result,
            "error": conv.error
        })
    
    return result

@router.post("/", response_model=ConversationResponse, status_code=status.HTTP_201_CREATED)
async def create_conversation(
    user_query: str,
    generated_sql: Optional[str] = None,
    query_result: Optional[List[Dict[str, Any]]] = None,
    error: Optional[str] = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Store a new conversation entry"""
    # Get the user's connection
    connection = get_user_connection(db, current_user.id)
    
    # Create new conversation record
    conversation = Conversation(
        id=uuid.uuid4(),
        user_id=current_user.id,
        connection_id=connection.id,
        user_query=user_query,
        generated_sql=generated_sql,
        query_result=query_result,
        error=error
    )
    
    db.add(conversation)
    db.commit()
    db.refresh(conversation)
    
    # Return the created conversation
    return {
        "id": conversation.id,
        "connection_id": conversation.connection_id,
        "timestamp": conversation.timestamp.isoformat(),
        "user_query": conversation.user_query,
        "generated_sql": conversation.generated_sql,
        "query_result": conversation.query_result,
        "error": conversation.error
    }

@router.delete("/clear", status_code=status.HTTP_200_OK)
async def clear_conversations(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Clear conversation history for the current user"""
    # Get the user's connection
    connection = get_user_connection(db, current_user.id)
    
    # Delete conversations for this user and connection
    deleted_count = db.query(Conversation).filter(
        Conversation.user_id == current_user.id,
        Conversation.connection_id == connection.id
    ).delete(synchronize_session=False)
    
    db.commit()
    
    return {"message": f"Successfully cleared {deleted_count} conversations"}

@router.post("/execute", response_model=ConversationResponse)
async def execute_query(
    query: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Execute a natural language query against the user's database"""
    # Get the user's connection
    connection = get_user_connection(db, current_user.id)
    
    # TODO: Here you would implement the SQL Agent logic:
    # 1. Use the natural language query to generate SQL
    # 2. Execute the SQL against the target database
    # 3. Process the results
    # 4. Store the conversation record
    
    # For now, we'll create a placeholder response
    conversation = Conversation(
        id=uuid.uuid4(),
        user_id=current_user.id,
        connection_id=connection.id,
        user_query=query,
        generated_sql="SELECT 'This is a placeholder' AS message",
        query_result=[{"message": "This is a placeholder response. SQL Agent functionality not implemented yet."}],
        error=None
    )
    
    db.add(conversation)
    db.commit()
    db.refresh(conversation)
    
    return {
        "id": conversation.id,
        "connection_id": conversation.connection_id,
        "timestamp": conversation.timestamp.isoformat(),
        "user_query": conversation.user_query,
        "generated_sql": conversation.generated_sql,
        "query_result": conversation.query_result,
        "error": conversation.error
    }