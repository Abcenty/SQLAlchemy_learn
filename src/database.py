from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, DeclarativeBase
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
from src.config import settings
import asyncio

# создаем синхронный движок, который будет создавать подключения к базе данных
sync_engine = create_engine(
    url=settings.DATABASE_URL_psycopg, # 
    echo=False, # возврат логов в консоль
    pool_size=5, # количество одновременных подключений к базе (одновремененных запросов)
    max_overflow=10, # пороговое количество одновременных подключений
)


# создаем асинхронный движок
async_engine = create_async_engine(
    url=settings.DATABASE_URL_asyncpg, 
    echo=False, # возврат логов в консоль
)

session_factory = sessionmaker(sync_engine)
async_session_factory = async_sessionmaker(async_engine)


class Base(DeclarativeBase):
    def __repr__(self):
        """Relationships не используются в repr(), т.к. могут вести к неожиданным подгрузкам"""
        cols = []
        for idx, col in enumerate(self.__table__.columns.keys()):
            if col in self.repr_cols or idx < self.repr_cols_num:
                cols.append(f"{col}={getattr(self, col)}")

        return f"<{self.__class__.__name__} {', '.join(cols)}>"
