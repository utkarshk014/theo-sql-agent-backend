# app/main.py
from fastapi import FastAPI
from app.api.routes import base, auth, connections, schema, queries, conversations
from app.db.models import Base
from app.db.database import engine
from fastapi.middleware.cors import CORSMiddleware

# Create tables
Base.metadata.create_all(bind=engine)

app = FastAPI(title="Theo SQL Agent",
    description="API for Theo SQL Agent",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(base.router)
app.include_router(auth.router)
app.include_router(connections.router)
app.include_router(schema.router)
app.include_router(queries.router)
app.include_router(conversations.router)

@app.get("/")
def read_root():
    return {"message": "Welcome to SQL Agent API"}