# app/api/routes/base.py
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.db.database import get_db

router = APIRouter(prefix="/api", tags=["base"])

@router.get("/health")
def health_check():
    return {"status": "healthy"}