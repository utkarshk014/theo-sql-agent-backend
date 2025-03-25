# init_db.py
import asyncio
import logging
from app.db.models import Base
from app.db.database import engine
from sqlalchemy_utils import database_exists, create_database
from sqlalchemy import text

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def init_db():
    # Create database if it doesn't exist
    if not database_exists(engine.url):
        create_database(engine.url)
        logger.info("Created database")
    
    # Connect to drop view first
    try:
        with engine.connect() as conn:
            conn.execute(text("DROP VIEW IF EXISTS recent_conversations CASCADE;"))
            conn.commit()
            logger.info("Dropped existing view")
    except Exception as e:
        logger.error(f"Error dropping view: {e}")
    
    # Drop all tables
    Base.metadata.drop_all(bind=engine)
    logger.info("Dropped existing tables")
    
    # Create tables
    Base.metadata.create_all(bind=engine)
    logger.info("Created tables")
    
    logger.info("Database initialization completed")

if __name__ == "__main__":
    logger.info("Creating initial database tables...")
    asyncio.run(init_db())