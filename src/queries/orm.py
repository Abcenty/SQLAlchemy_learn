from sqlalchemy import Integer, and_, func, select, text, insert
from schemas import ResumesRelVacanciesRepliedWithoutVacancyCompensationDTO, WorkersRelDTO
from src.database import sync_engine, session_factory
from sqlalchemy.orm import aliased, joinedload, selectinload, contains_eager
import asyncio
from models import ResumesOrm, VacanciesOrm, WorkersOrm, Base, Workload

class SyncORM:
    @staticmethod
    def create_tables():
        sync_engine.echo = False
        Base.metadata.drop_all(sync_engine) # удаление таблиц перед их созданием заново
        Base.metadata.create_all(sync_engine) # создание всех таблиц, хранящихся в метадате
        sync_engine.echo = True

    @staticmethod
    def drop_tables():
        sync_engine.echo = False
        Base.metadata.drop_all(sync_engine) # удаление таблиц перед их созданием заново
        sync_engine.echo = True

    @staticmethod
    def insert_workers():
        with session_factory() as session:
            worker_jack = WorkersOrm(username='Jack')
            worker_michael = WorkersOrm(username='Michael')
            session.add_all([worker_jack, worker_michael]) # для добавления нескольких
            session.flush() # помещает данные в бд, но не завершает транзакцию
            session.commit()
            
    @staticmethod
    def insert_resumes():
        with session_factory() as session:
            resume_jack_1 = ResumesOrm(
                title="Python Junior Developer", compensation=50000, workload=Workload.fulltime, worker_id=1)
            resume_jack_2 = ResumesOrm(
                title="Python Разработчик", compensation=150000, workload=Workload.fulltime, worker_id=1)
            resume_michael_1 = ResumesOrm(
                title="Python Data Engineer", compensation=250000, workload=Workload.parttime, worker_id=2)
            resume_michael_2 = ResumesOrm(
                title="Data Scientist", compensation=300000, workload=Workload.fulltime, worker_id=2)
            session.add_all([resume_jack_1, resume_jack_2, 
                             resume_michael_1, resume_michael_2])
            session.commit()
            
    @staticmethod
    def insert_additional_resumes():
        with session_factory() as session:
            workers = [
                {"username": "Artem"},  # id 3
                {"username": "Roman"},  # id 4
                {"username": "Petr"},   # id 5
            ]
            resumes = [
                {"title": "Python программист", "compensation": 60000, "workload": "fulltime", "worker_id": 3},
                {"title": "Machine Learning Engineer", "compensation": 70000, "workload": "parttime", "worker_id": 3},
                {"title": "Python Data Scientist", "compensation": 80000, "workload": "parttime", "worker_id": 4},
                {"title": "Python Analyst", "compensation": 90000, "workload": "fulltime", "worker_id": 4},
                {"title": "Python Junior Developer", "compensation": 100000, "workload": "fulltime", "worker_id": 5},
            ]
            insert_workers = insert(WorkersOrm).values(workers)
            insert_resumes = insert(ResumesOrm).values(resumes)
            session.execute(insert_workers)
            session.execute(insert_resumes)
            session.commit()
            
    @staticmethod
    def select_resumes_avg_compensation(like_language: str = "Python"):
        with session_factory() as session:
            query = (
                select(
                    ResumesOrm.workload, # выбрали столбец графика из таблицы резюме
                    # 1 вариант использования cast
                    # cast(func.avg(ResumesOrm.compensation), Integer).label("avg_compensation"),
                    # 2 вариант использования cast (предпочтительный способ)
                    func.avg(ResumesOrm.compensation).cast(Integer).label("avg_compensation"), # ср ЗП
                )
                .select_from(ResumesOrm)  # явно задали таблицу, откуда берем (опционально)
                .filter(and_(
                    ResumesOrm.title.contains(like_language),
                    ResumesOrm.compensation > 40000,
                ))
                .group_by(ResumesOrm.workload)
                .having(func.avg(ResumesOrm.compensation) > 70000)
            )
            # выводит удобочитаем запрос с подстановкой значений, которые мы ввели вместо переменных
            print(query.compile(compile_kwargs={"literal_binds": True}))
            res = session.execute(query)
            result = res.all()
            # выводим параметр avg_compensation первого элементы из ответа на запрос
            print(result[0].avg_compensation)
            
    @staticmethod
    def join_cte_subquery_window_func():
        """
        WITH helper2 AS (
            SELECT *, compensation-avg_workload_compensation AS compensation_diff
            FROM 
            (SELECT
                w.id,
                w.username,
                r.compensation,
                r.workload,
                avg(r.compensation) OVER (PARTITION BY workload)::int AS avg_workload_compensation
            FROM resumes r
            JOIN workers w ON r.worker_id = w.id) helper1
        )
        SELECT * FROM helper2
        ORDER BY compensation_diff DESC;
        """
        with session_factory() as session:
            r = aliased(ResumesOrm) # алиас для таблицы
            w = aliased(WorkersOrm) # алиас для таблицы
            subq = ( # подзапрос
                select(
                    r,
                    w,
                    # оконная функция
                    func.avg(r.compensation).over(partition_by=r.workload).cast(Integer).label("avg_workload_compensation"),
                )
                # .select_from(r) # таблица, откуда достаем данные
                .join(r, r.worker_id == w.id)
                .subquery("helper1") # указываем название подзапроса
            )
            cte = ( # CTE
                select(
                    # явно указываем все столбцы
                    subq.c.worker_id,
                    subq.c.username,
                    subq.c.compensation,
                    subq.c.workload,
                    subq.c.avg_workload_compensation,
                    (subq.c.compensation - subq.c.avg_workload_compensation).label("compensation_diff"),
                )
                .cte("helper2") # указываем название CTE
            )
            query = ( # финальный запрос
                select(cte)
                .order_by(cte.c.compensation_diff.desc()) #.c. для выбора колонок, т.к. у нас модель, а не ОРМ сущнсоть
            )

            res = session.execute(query)
            result = res.all()
            # print(f"{len(result)=}. {result=}")
            
    @staticmethod
    def select_workers_with_lazy_relationship():
        with session_factory() as session:
            query = (
                select(WorkersOrm)
            )
            
            res = session.execute(query)
            result = res.scalars().all()

            worker_1_resumes = result[0].resumes
            print(worker_1_resumes)
            
            worker_2_resumes = result[1].resumes
            print(worker_2_resumes)
            
    @staticmethod
    def select_workers_with_joined_relationship():
        with session_factory() as session:
            query = (
                select(WorkersOrm)
                .options(joinedload(WorkersOrm.resumes))
            )
            
            res = session.execute(query)
            result = res.unique().scalars().all()

            worker_1_resumes = result[0].resumes
            print(worker_1_resumes)
            
            worker_2_resumes = result[1].resumes
            print(worker_2_resumes)

    @staticmethod
    def select_workers_with_selectin_relationship():
        with session_factory() as session:
            query = (
                select(WorkersOrm)
                .options(selectinload(WorkersOrm.resumes))
            )
            
            res = session.execute(query)
            result = res.scalars().all()

            worker_1_resumes = result[0].resumes
            # print(worker_1_resumes)
            
            worker_2_resumes = result[1].resumes
            # print(worker_2_resumes)
            
    @staticmethod
    def select_workers_with_condition_relationship():
        with session_factory() as session:
            query = (
                select(WorkersOrm)
                .options(selectinload(WorkersOrm.resumes_parttime))
            )

            res = session.execute(query)
            result = res.scalars().all()
            print(result)
            
    @staticmethod
    def select_workers_with_condition_relationship_contains_eager():
        with session_factory() as session:
            query = (
                select(WorkersOrm)
                .join(WorkersOrm.resumes)
                .options(contains_eager(WorkersOrm.resumes))
                .filter(ResumesOrm.workload == 'parttime')
            )

            res = session.execute(query)
            result = res.unique().scalars().all()
            print(result)
            
            
    @staticmethod
    def convert_workers_to_dto():
        with session_factory() as session:
            query = (
                select(WorkersOrm)
                .options(selectinload(WorkersOrm.resumes))
                .limit(2)
            )

            res = session.execute(query)
            result_orm = res.scalars().all()
            print(f"{result_orm=}")
            result_dto = [WorkersRelDTO.model_validate(row, from_attributes=True) for row in result_orm]
            print(f"{result_dto=}")
            return result_dto
        
    @staticmethod
    def add_vacancies_and_replies():
        with session_factory() as session:
            new_vacancy = VacanciesOrm(title="Python разработчик", compensation=100000)
            resume_1 = session.get(ResumesOrm, 1)
            resume_2 = session.get(ResumesOrm, 2)
            resume_1.vacancies_replied.append(new_vacancy)
            resume_2.vacancies_replied.append(new_vacancy)
            session.commit()


    @staticmethod
    def select_resumes_with_all_relationships():
        with session_factory() as session:
            query = (
                # используем разные типы подгрузки joinedload и selectinload для разных вложенных сущностей
                select(ResumesOrm)
                .options(joinedload(ResumesOrm.worker))
                .options(selectinload(ResumesOrm.vacancies_replied).load_only(VacanciesOrm.title))
                # load_only для подгрузки конкретных нужных аттрибутов
            )
            res = session.execute(query)
            result_orm = res.unique().scalars().all()
            print(f"{result_orm=}")
            # Обратите внимание, что созданная в видео модель содержала лишний столбец compensation
            # И так как он есть в схеме ResumesRelVacanciesRepliedDTO, столбец compensation был вызван
            # Алхимией через ленивую загрузку. В асинхронном варианте это приводило к краху программы
            result_dto = [ResumesRelVacanciesRepliedWithoutVacancyCompensationDTO.model_validate(row, from_attributes=True) for row in result_orm]
            print(f"{result_dto=}")
            return result_dto
                
    @staticmethod
    def select_workers():
        with session_factory() as session:
            # worker_id = 1
            # worker_jack = session.get(WorkersOrm, worker_id) # получение одного
            query = select(WorkersOrm) # SELECT * FROM workers_table;
            result = session.execute(query)
            workers = result.scalars().all() # scalars возвращает первый элемент каждого кортежа ответа
            print(f"{workers=}")
                
    @staticmethod
    def update_workers(worker_id: int = 2, new_username: str = "Misha"):
        with session_factory() as session:
            worker_michael = session.get(WorkersOrm, worker_id)
            worker_michael.username = new_username
            session.expire_all() # сбрасывает изменения, ничего не будет в БД
            session.refresh(worker_michael) # обновляет данные до состояния, которое сейчас в БД
            session.commit()
            
            






        
# async def async_insert_data():
#     worker_bobr = WorkersOrm(username='Bobr')
#     worker_wolf = WorkersOrm(username='Wolf')
#     async with async_session_factory() as session:
#         # session.add(worker_bobr) # для добавления одного
#         session.add_all([worker_bobr, worker_wolf]) # для добавления нескольких
#         await session.commit()