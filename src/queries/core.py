from sqlalchemy import text, insert, select, update
from database import sync_engine, async_engine
from models import workers_table, metadata_obj, Base

def get_123_sync():
    pass

async def get_123_async():
    pass


class SyncCore:
    @staticmethod
    def create_tables():
        sync_engine.echo = False
        metadata_obj.drop_all(sync_engine)
        metadata_obj.create_all(sync_engine)
        sync_engine.echo = True
        
    
    @staticmethod
    def drop_tables():
        sync_engine.echo = False
        metadata_obj.drop_all(sync_engine)
        sync_engine.echo = True
        
    @staticmethod
    def insert_workers():
        with sync_engine.connect() as conn:
            stmt = insert(workers_table).values(
                [
                    {'username': "Jack"},
                    {'username': "Michael"},
                ]
            )
            conn.execute(stmt)
            conn.commit()
            
    @staticmethod
    def select_workers():
        with sync_engine.connect() as conn:
            query = select(workers_table) # SELECT * FROM workers_table;
            result = conn.execute(query)
            workers = result.all()
            print(f"{workers=}")
            
            
    @staticmethod
    def update_workers(worker_id: int = 2, new_username: str = "Misha"):
        with sync_engine.connect() as conn:
            # stmt = text("UPDATE workers SET username=:new_username WHERE id=:worker_id") # сырой запрос с : для подстановки параметра
            # stmt = stmt.bindparams(new_username=new_username, worker_id=worker_id) # защита от sql инъекций!!!!!
            stmt = ( # ОРМ запрос
                update(workers_table)
                .values(username=new_username)
                # .where(workers_table.c.id == worker_id)
                .filter_by(id=worker_id)
            )
            conn.execute(stmt)
            conn.commit()