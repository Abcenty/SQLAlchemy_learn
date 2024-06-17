import os
import sys
import asyncio

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

"""Добавление папки src в путь для корректной работы импортов"""
sys.path.insert(1, os.path.join(sys.path[0], '..'))

from queries.core import SyncCore
from queries.orm import SyncORM


async def main():

    # ORM
    if "--orm" in sys.argv and "--sync" in sys.argv:
        SyncORM.create_tables()
        SyncORM.insert_workers()
        SyncORM.select_workers()
        # SyncORM.update_worker()
        SyncORM.insert_resumes()
        SyncORM.select_resumes_avg_compensation()
        SyncORM.insert_additional_resumes()
        SyncORM.join_cte_subquery_window_func()
        SyncORM.select_workers_with_lazy_relationship()
        SyncORM.select_workers_with_joined_relationship()
        SyncORM.select_workers_with_selectin_relationship()
        SyncORM.select_workers_with_condition_relationship()
        SyncORM.select_workers_with_condition_relationship_contains_eager()
        # SyncORM.select_workers_with_relationship_contains_eager_with_limit()
        SyncORM.convert_workers_to_dto()
        # SyncORM.add_vacancies_and_replies()
        # SyncORM.select_resumes_with_all_relationships()


def create_fastapi_app():
    app = FastAPI(title="FastAPI")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
    )
        
    @app.get("/workers", tags=["Кандидат"])
    async def get_workers():
        workers = SyncORM.convert_workers_to_dto()
        return workers
    
    return app
    

app = create_fastapi_app()


if __name__ == "__main__":
    if "--webserver" in sys.argv:
        uvicorn.run(
            app="src.main:app",
            reload=True,
        )

