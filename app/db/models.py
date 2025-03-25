# app/db/models.py
import uuid
from sqlalchemy import Column, String, Text, DateTime, JSON, ForeignKey
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship

Base = declarative_base()

class User(Base):
    __tablename__ = "users"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    clerk_user_id = Column(String, unique=True, index=True, nullable=False)
    email = Column(String, unique=True, index=True, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    last_login = Column(DateTime(timezone=True), nullable=True)
    
    # Relationships
    connections = relationship("DatabaseConnection", back_populates="user", cascade="all, delete-orphan")
    conversations = relationship("Conversation", back_populates="user", cascade="all, delete-orphan")


class DatabaseConnection(Base):
    __tablename__ = "database_connections"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    name = Column(String, nullable=False)
    vector_db_url = Column(Text, nullable=False)
    target_db_url = Column(Text, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    user = relationship("User", back_populates="connections")
    conversations = relationship("Conversation", back_populates="connection", cascade="all, delete-orphan")


class Conversation(Base):
    __tablename__ = "sql_agent_conversations"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    connection_id = Column(UUID(as_uuid=True), ForeignKey("database_connections.id", ondelete="CASCADE"), nullable=False)
    timestamp = Column(DateTime(timezone=True), server_default=func.now())
    user_query = Column(Text, nullable=False)
    generated_sql = Column(Text, nullable=True)
    query_result = Column(JSON, nullable=True)
    error = Column(Text, nullable=True)
    
    # Relationships
    user = relationship("User", back_populates="conversations")
    connection = relationship("DatabaseConnection", back_populates="conversations")