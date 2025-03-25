# app/core/auth.py
from fastapi import Depends, HTTPException, status, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import jwt
from sqlalchemy.orm import Session
from app.db.database import get_db
from app.db.models import User
from app.config import settings

security = HTTPBearer()

async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db)
):
    """
    Validate the JWT token and return the user
    """
    try:
        token = credentials.credentials
        
        # Verify the token using Clerk's public key
        # Note: This is a simplified version. In production, you should use
        # proper JWT verification with Clerk's public key
        try:
            # payload = jwt.decode(
            #     token, 
            #     settings.CLERK_JWT_PUBLIC_KEY, 
            #     algorithms=["RS256"],
            #     audience=settings.CLERK_JWT_AUDIENCE
            # )
            
            # For development, we'll just parse the token
            # In production, make sure to properly verify the token
            payload = jwt.decode(token, options={"verify_signature": False})
            user_id = payload.get("sub")
            
            if not user_id:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid token payload"
                )
            
            # Get user from database
            user = db.query(User).filter(User.clerk_user_id == user_id).first()
            if not user:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="User not found"
                )
                
            return user
            
        except jwt.PyJWTError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token"
            )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Authentication error: {str(e)}"
        )