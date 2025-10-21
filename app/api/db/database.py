# generative-ai-service/app/api/db/database.py
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker

from app.api.db.entities import Base
from app.api.core.config import settings
from typing import Annotated
from fastapi import Depends

database_url = (
f"postgresql+psycopg://{settings.postgres_username}:{settings.postgres_password}@localhost:5432/{settings.postgres_db}"
)

engine = create_async_engine(database_url, echo=True)

async def init_db() -> None:
    async with engine.begin() as con:
        await con.run_sync(Base.metadata.drop_all)
        await con.run_sync(Base.metadata.create_all)

async_session = async_sessionmaker(
    bind=engine, class_=AsyncSession, autocommit=False,autoflush=False
)

async def get_db_session():
    try:
        async with async_session() as session:
            yield session
    except:
        await session.rollback()
        raise
    finally:
        await session.clode()

DBSessionDep = Annotated[AsyncSession, Depends(get_db_session)]