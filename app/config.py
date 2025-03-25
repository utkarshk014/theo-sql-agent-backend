# app/config.py
from pydantic_settings import BaseSettings
import os
import base64

class Settings(BaseSettings):
    # Environment
    ENVIRONMENT: str = "development"  
    
    # Database
    DATABASE_URL: str
    
    # Clerk
    CLERK_SECRET_KEY: str
    CLERK_WEBHOOK_SECRET: str = ""
    
    # API Keys
    GEMINI_API_KEY: str
    
    class Config:
        env_file = ".env"

settings = Settings()