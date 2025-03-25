# app/api/routes/auth.py
from fastapi import APIRouter, Depends, Request, HTTPException, status
from sqlalchemy.orm import Session
import uuid
import json
import traceback
from app.db.database import get_db
from app.db.models import User
from app.config import settings
import httpx

router = APIRouter(prefix="/api/auth", tags=["auth"])

async def fetch_clerk_user_details(user_id: str):
    """Fetch user details from Clerk API"""
    url = f"https://api.clerk.com/v1/users/{user_id}"
    
    headers = {
        "Authorization": f"Bearer {settings.CLERK_SECRET_KEY}",
        "Content-Type": "application/json"
    }
    
    async with httpx.AsyncClient() as client:
        response = await client.get(url, headers=headers)
        
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Error fetching user details: {response.text}")
            return None

@router.post("/webhook")
async def clerk_webhook(request: Request, db: Session = Depends(get_db)):
    """Handle Clerk webhooks for user creation and updates"""
    try:
        # Parse payload
        payload = await request.json()
        print("Webhook received:", payload.get("type"))
        
        # Get event type and data
        event_type = payload.get("type")
        data = payload.get("data", {})
        
        # Handle user creation
        if event_type == "user.created":
            # Extract email
            email = None
            email_addresses = data.get("email_addresses", [])
            
            if email_addresses and len(email_addresses) > 0:
                if data.get("primary_email_address_id"):
                    for email_obj in email_addresses:
                        if email_obj.get("id") == data.get("primary_email_address_id"):
                            email = email_obj.get("email_address")
                            break
                
                # Fallback to first email
                if not email and email_addresses:
                    email = email_addresses[0].get("email_address")
            
            if not email:
                return {"status": "error", "message": "No email found for user"}
            
            # Create user
            clerk_user_id = data.get("id")
            if not clerk_user_id:
                return {"status": "error", "message": "Missing user ID"}
                
            # Create new user
            new_user = User(
                id=uuid.uuid4(),
                clerk_user_id=clerk_user_id,
                email=email
            )
            db.add(new_user)
            db.commit()
            
            print(f"Created user with clerk_id {clerk_user_id} and email {email}")
            return {"status": "success", "message": "User created"}
            
        # Handle user update
        elif event_type == "user.updated":
            clerk_user_id = data.get("id")
            if not clerk_user_id:
                return {"status": "error", "message": "Missing user ID"}
                
            user = db.query(User).filter(User.clerk_user_id == clerk_user_id).first()
            
            if user:
                # Update email
                email_addresses = data.get("email_addresses", [])
                if email_addresses and len(email_addresses) > 0:
                    if data.get("primary_email_address_id"):
                        for email_obj in email_addresses:
                            if email_obj.get("id") == data.get("primary_email_address_id"):
                                user.email = email_obj.get("email_address")
                                break
                    
                    # Fallback to first email
                    if not user.email and email_addresses:
                        user.email = email_addresses[0].get("email_address")
                
                db.commit()
                print(f"Updated user with clerk_id {clerk_user_id}")
                return {"status": "success", "message": "User updated"}
        
        # Handle session creation
        elif event_type == "session.created":
            # Check if user exists, if not create the user
            user_id = data.get("user_id")
            if not user_id:
                return {"status": "error", "message": "Missing user ID in session data"}
                
            # Check if user already exists
            existing_user = db.query(User).filter(User.clerk_user_id == user_id).first()
            
            if existing_user:
                # Update last login time if needed
                print(f"Session created for existing user: {user_id}")
                return {"status": "success", "message": "Session created for existing user"}
            
            # If we're here, we need to create a user
            user_details = await fetch_clerk_user_details(user_id)
            
            if not user_details:
                return {"status": "error", "message": "Failed to fetch user details from Clerk"}
            
            # Extract email
            email = None
            email_addresses = user_details.get("email_addresses", [])
            
            if email_addresses and len(email_addresses) > 0:
                if user_details.get("primary_email_address_id"):
                    for email_obj in email_addresses:
                        if email_obj.get("id") == user_details.get("primary_email_address_id"):
                            email = email_obj.get("email_address")
                            break
                
                # Fallback to first email
                if not email and email_addresses:
                    email = email_addresses[0].get("email_address")
            
            if not email:
                return {"status": "error", "message": "No email found for user"}
            
            # Create new user
            new_user = User(
                id=uuid.uuid4(),
                clerk_user_id=user_id,
                email=email
            )
            db.add(new_user)
            db.commit()
            
            print(f"Created user with clerk_id {user_id} from session")
            return {"status": "success", "message": "User created from session"}
        
        # Return success for other events
        return {"status": "success", "message": f"Event {event_type} processed"}
    
    except Exception as e:
        print(f"Error in webhook handler: {str(e)}")
        traceback.print_exc()
        return {"status": "error", "message": str(e)}