1) Для начала работы с sqlalchemy нужно создать движок
Движок бывает синхронный и асинхронный

В настройках нужно указать СУБД, библиотеку и креды БД,
через которые будем подключаться к бд, например

f"postgresql+asyncpg://{self.DB_USER}:{self.DB_PASS}@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"

Эту строку нужно прокинуть в url параметр движка при его создании

Для создание таблиц БД можно использовтаь императивный метод описания
(в виде переменной с рядом параметров)

Эту таблицу нужно создать через ее метаданные, прокинув туда движок

metadata_obj.create_all(sync_engine)

В созданную таблицу можно помещать данные и можно их извлекать

Делать это можно с помощью прямых sql запросов

def clean_sql_insert_data():
    """Чистый sql запрос"""
    with sync_engine.connect() as conn:
        stmt = """INSERT INTO workers (username) VALUES
        ('Bobr'),
        ('Volk');""" 
        conn.execute(text(stmt)) 
        conn.commit()

(обязательно коммитим, иначе данные не утвердятся в БД)

Также можно через ОРМ

def orm_insert_data():
    """Запрос с помощью ОРМ"""
    with sync_engine.connect() as conn:
        stmt = insert(workers_table).values(
            [
                {'username': 'Bobr'},
                {'username': 'Volk'},
            ]
        )
        conn.execute(stmt) 
        conn.commit() 

ОРМ медленнее, но удобнее и безопаснее

2) Сессии нужны для транзакций. Они позволяют подтвердить или
откатить запрос к БД. У каждой сессии может быть набор конфигураций.
Часто нужно использовать сессии с одинаковой конфигурацией. Чтобы не
задавать ее вручную каждой сессии, используют sessionmaker и 
async_session_maker. В эту сессию мы передаем соотвествующий движок,
а затем используем ее в запросах, обращаясь к экземплярам фабрики сессий
(session_maker), а точнее к из методам. т.е.:

async def async_insert_data():
    worker_bobr = WorkersOrm(username='Bobr')
    worker_wolf = WorkersOrm(username='Wolf')
    async with async_session_factory() as session:
        # session.add(worker_bobr) # для добавления одного
        session.add_all([worker_bobr, worker_wolf]) # для добавления нескольких
        await session.commit()

3) Декларативный стиль написания моделей - через классы, имеет ряд удобностей.
Во-первых можно удобно использовать перечисляемые типы(см. поле workload):

class Workload(enum.Enum):
    parttime = 'parttime'
    fulltime = 'fulltime'
    
    
class ResumesOrm(Base):
    __tablename__ = 'resumes'
    
    id: Mapped[int] = mapped_column(primary_key=True)
    title: Mapped[str]
    compensation: Mapped[Optional[int]]
    workload: Mapped[Workload]
    worker_id: Mapped[int] = mapped_column(ForeignKey('workers.id', ondelete='CASCADE'))
    created_at: Mapped[datetime.datetime] = mapped_column(server_default=text("TIMEZONE('utc', now())"))
    updated_at: Mapped[datetime.datetime] = mapped_column(
        server_default=text("TIMEZONE('utc', now())"),
        onupdate=datetime.UTC
        )

Во-вторых можно задавать дефолтное время на уровне СУБД через server_default (в отличие от аттрибута default,
который задает время на уровне приложения)

Можно удобно указывать внешние ключи, ссылаясь на поле таблицы, от которой зависят по названию этой таблицы и
собственно самому полю

Также есть некоторый фишки, повышающие DRY:

intpk = Annotated[int, mapped_column(primary_key=True)]

class WorkersOrm(Base):
    __tablename__ = 'workers'
    id: Mapped[intpk]
    ...

Можно через typing.Annotated выносить некоторые повторяющиеся типы

Также можно поступить с updated_at и created_at:

created_at = Annotated[datetime.datetime, mapped_column(server_default=text("TIMEZONE('utc', now())"))]
updated_at = Annotated[datetime.datetime, mapped_column(
        server_default=text("TIMEZONE('utc', now())"),
        onupdate=datetime.UTC
        )]

class ResumesOrm(Base):
    __tablename__ = 'resumes'
    
    ...
    created_at: Mapped[created_at]
    updated_at: Mapped[updated_at]


Также нужно помнить, что при декларативном методе задания моделей, мы наследуемся от класса Base,
поэтому доступ к метаданным получаем через Base.metadata

P.S. Я хер знает почему, но если не отрабатывает наследлование от 

from sqlalchemy.orm import DeclarativeBase

class Base(DeclarativeBase):
    pass

то используй

from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

4) Core запросы максимально оптимальны, они используют sql напрямую
либо же используют функции 

Вот так выглядит core запрос через sql
@staticmethod
def update_workers(worker_id: int = 2, new_username: str = "Misha"):
    with sync_engine.connect() as conn:
        stmt = text("UPDATE workers SET username=:new_username WHERE id=:worker_id") # сырой запрос с : для подстановки параметра
        stmt = stmt.bindparams(new_username=new_username, worker_id=worker_id) # защита от sql инъекций!!!!!
        conn.execute(stmt)
        conn.commit()

Вот так выглядит core запрос через функции алхимии

@staticmethod
def update_workers(worker_id: int = 2, new_username: str = "Misha"):
    with sync_engine.connect() as conn:
        stmt = (
                    update(workers_table)
                    .values(username=new_username)
                    # .where(workers_table.c.id == worker_id)
                    .filter_by(id=worker_id)
                )
        conn.execute(stmt)
        conn.commit()


ОРМ точно также получает различные данные в виде строк, но преобразует
их в модели, что дает прослойку абстракции над данными.

Вот так выглядит запрос через ОРМ
@staticmethod
def select_workers():
    with session_factory() as session:
        # worker_id = 1
        # worker_jack = session.get(WorkersOrm, worker_id) # получение одного
        query = select(WorkersOrm) # SELECT * FROM workers_table;
        result = session.execute(query)
        workers = result.scalars().all() # scalars возвращает первый элемент каждого кортежа ответа
        print(f"{workers=}")


Причем workers = result.all() - дает список кортежей, содержащих модели, которые мы описали
в данном случае это модель WorkersORM. 
result.scalars().all() - дает список первых элементов этих кортежей, что бывает чатсо полезно
т.к. в этих кортежах чатсо по одному элементу, а нам нужно работать именно с ними
Так как это список, то мы можем получать его элементы через result.scalars().all()[0] и тд

flush - помещает данные в бд, но не завершает транзакцию. Например нам нужно отгрузить сущность
к которой прикреплены другие сущности, у которых есть внешних ключ на первую сущность. Мы не сможем
передать ей этот внешний ключ, т.к. сущности еще нет в бд, но с помощью flush мы можем зафиксировать
сущность и досттаь ее внешний ключ, но транзакцию завершим только когда все данные будут притянуты
(завершим через commit, естественно). Вызывается так session.flush()

refresh - обновляет данные до состояния, которое сейчас в бд (до коммита). Например мы рабоатем с какой-то сущностью,
но пока мы ее обрабатывали, ее другой пользователь успел изменить. Если мы хотим продолжить работать с
актуалбнйо сущнсотью, то можно прописать session.refresh(worker_michael), где в скобках указываем сущность,
котору юхотим обновить

expire - сбрасывает все изменения сущности (до коммита). Например мы наизменяли сущнсоть, но произошло условие
которое делает наши изменения неактуальными - мы можем их откатить. Причем можем откатить все изменения ко
всем сущностями, которые трогали через session.expire_all(), а можем изменить конкретную сущность через
session.expire(worker_michael)

5) Select запросы нужны чтобы доставтаь любые данные, упорядоченные любым образом из БД

Мы хотим отобрать имеющиеся резюме по их графику и вывести среднюю запрашиваемую зарплату у питонистов
с этим графиком, причем инфу о средней зарплате будем брать из столбца Зарплата, назвав его Средняя зарплата,
учитвать будем только те резюме, в названии которых есть слово Python и у которых запрашиваемая зарплата не
ниже 40000. Запрос сгруппируем по графику. sql выглядит так

select workload, avg(compensation)::int as avg_compensation
from resumes
where title like '%Python%' and compensation > 40000
group by workload

Метод ОРМ, делающий этот запрос выглядит так:

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



cast - приводит значение к заданному типу
label - переназывает столбец для конкретного запроса

and_ - для объединения условий внутри filter
Фильтрация:
filter_by(id=1...) - принимает условия и пропускает только ответы, содержащие эти значения
where и filter делают то же самое

group_by - упорядочивание по выбранному полю

having - финальная фильтрация отсортированных значений по какому-то признаку


6) Сложные запросы позволяют доставать самые разнообразные данные, объединять результаты
запросов, делать под запросы и многое другое

Пример SQL запроса с JOIN, CTE, подзапросом и с оконной функцией (в нем мы получаем список
сотрудников с их резюме, вычисляем среднее значение ЗП у их всех и сранвнвиаем это значение
с ЗП каждого из работников)

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


Common Table Expressions, CTE, Общие табличные выражения - позволяют с помощью ключевого слова WITH
записывать дополнительные операторы для реализации в больших запросах. Т.Е. в данном примере все внутри
скобок после WITH присвоенно оператору helper2, что позволит проще оперирвоать этой сущностью

JOIN - ключевое слово для извлечения данные из нескольких таблиц. Бывает 4х видов:
INNER JOIN (простое соединение). Возвращает все строки из нескольких таблиц, где выполняется условие соединения.

LEFT OUTER JOIN. Возвращает все строки из таблиц с левосторонним соединением, указанным в условии ON, и только те строки из другой таблицы, где объединяемые поля равны. 

RIGHT OUTER JOIN. Возвращает все строки из таблицы с правосторонним соединением, указанной в условии ON, и только те строки из другой таблицы, где соединенные поля равны. (В АЛХИМИИ ЕГО НЕТ, ТОЛЬКО LEFT!!!)

FULL OUTER JOIN. Возвращает все строки из левой таблицы и правой таблицы с NULL-значениями в месте, где условие соединения не выполняется.

В данном случае использовался INNER JOIN, другие изучи подробнее в курсе по PostgreSQL.

Оконная функция - отдельная сущность, которая проводит вычисления для набора строк

avg(r.compensation) OVER (PARTITION BY workload)::int AS avg_workload_compensation - оконная функция
avg - сама функция (аггрегатная, она сворачивает результат в 1 ответ, т.е. в данном случае считает 9 запросов
и выдает 1 ответ - среднее их значение, когда оконная функция выдаст точно также 9 ответов, т.е. выпишет среднее
для каждой строки, а PARTITION BY - группирует по аргументу(но не упорядочивает, упорядочивание происходит через
GROUP BY)), r.compensation - ее аргумент, OVER - ключевое слово, определяющее разделение
строки запроса для обработки оконной функцией.

Вот так выглядит метод ОРМ, делающий этот запрос:
@staticmethod
    def join_cte_subquery_window_func():
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

7) Relationship в SQLAlchemy позволяет связывать таблицы между собой. Foreign key дает связь между таблицей
и айди сущнсоти, а relationship позвоялет делать дполонительно вложенную сущность

class WorkersOrm(Base):
    __tablename__ = "workers"
    id: Mapped[intpk]
    username: Mapped[str] = mapped_column()
    resumes: Mapped[list["ResumesOrm"]] = relationship()

    
    
class ResumesOrm(Base):
    __tablename__ = "resumes"  
    id: Mapped[intpk]
    title: Mapped[str]
    compensation: Mapped[Optional[int]]
    workload: Mapped[Workload]
    worker_id: Mapped[int] = mapped_column(ForeignKey('workers.id', ondelete='CASCADE'))
    created_at: Mapped[created_at]
    updated_at: Mapped[updated_at]
    worker: Mapped["WorkersOrm"] = relationship()

Внутри Mapped указываем название модели в ковычках, чтобы избежать проблем с циркулярными импортами

Напишем метод, который будет притягивать данные о 2х работниках и резюме:


!!!!!ЛЕНИВАЯ ПОДГРУЗКА НЕ РАБОТАЕТ В АСИНХРОНЕ!!!!! (используй joinedload и selectinload)

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

Проблема этого исполнения в том, что возникает проблема n+1, т.е. мы кидаем запрос на получение
работника, и если хотим посмотреть его резюме, то кидаем отдельный запрос на его резюме и так с
каждым работнкиом, из-за чего БД перегружается. Это можно исправить, подгрузив сразу все данные
одним большим запросом с помощью join:

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

!!!!!Причем при использовании join, т.к. у одного работника может быть несколько резюме, будут
дублироваться данные самого работника. Чтобы это избежать используем чисто питонячий
метод unique(), который не отправляет запрос в БД сам по себе, но на уровне приложения
отсеит дублирующиеся данные. Но при этом на уровне БД эта пробелма все равно осталась.
!!!!!Следует помнить, что joinload работает оптимально только для связей O-2-O и M-2-O.!!!!!

!!!!!Для O-2-M и M-2-M следует использовать selectinload.!!!!!
Он отправляет два запроса, сначала отдельно подгружая сущнсоть, а затем вложенные сущности
В 2 раза больше запросов, но гораздо меньше гоняемого трафика.

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

Чтобы модели в логах выводились красиво переопредели __repr__ у класса, например

def __repr__(self):
    return f"<self.__class__.__name__>"

А еще лучше сделать это в базовом классе вот таким красивым образом:

def __repr__(self):
        """Relationships не используются в repr(), т.к. могут вести к неожиданным подгрузкам"""
        cols = []
        for idx, col in enumerate(self.__table__.columns.keys()):
            if col in self.repr_cols or idx < self.repr_cols_num:
                cols.append(f"{col}={getattr(self, col)}")
        return f"<{self.__class__.__name__} {', '.join(cols)}>"

8) Relationship очень продвинутый инстурмент
у него есть параметр back_populates, который явно указывает название таблицы,
на которую он ссылается.
Также есть backref, но это устаревшая практика. Он по сути создает внутри модели,
на которую мы ссылаемся поле с вложеной структурой. Лучше делать это явно через
back_populates.

С помощью аттрибута primary_join можно указывать дополнительные условия, по которым
будут фильтроваться вложенные сущности (например только резюме с частичной занятостью)

resumes_parttime=ralationship(
    back_populates='worker',
    primaryjoin="and_(WorkersOrm.id == ResumesOrm.worker_id, ResumesOrm.workload == 'parttime')",
)

Также нужно прописать метод, который будет забирать сущнсоти именно по этому полю

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

Точно также с помощью аттрибута order_by мы можем задать условие сортировки
order_by="ResumesOrm.id.desc()", decs и asc это порядок сортировки

также можно с помощью аттрибуа lazy указать тип подгрузки данные, например
selectin или joined, но это считается пмовитоном, так как это неявное указание и
лучше указывать в самом запросе

*Также есть метод contains_eager() который просит подтянуть вложенную сущность если она
есть, если нет - выдаст пустую сущность. Обычно предварительно эти сущности явно джойнятся
но могут и нет, а могут и джойнится вообще другие сущности, хотя contains_eager мы укажем
исходную и тогда она будет подгруджать на место тех вложенных сущнсотей другие сущности.
В общем интересная функция, если че - почитай про нее. А вот метод, содержащий ее:

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

Этот метод позволяет сделать лимит по вложенным сущностям. Если понадобится - иди
по ссылке в комменте

@staticmethod
    def select_workers_with_relationship_contains_eager_with_limit():
        # Горячо рекомендую ознакомиться: https://stackoverflow.com/a/72298903/22259413 
        with session_factory() as session:
            subq = (
                select(ResumesOrm.id.label("parttime_resume_id"))
                .filter(ResumesOrm.worker_id == WorkersOrm.id)
                .order_by(WorkersOrm.id.desc())
                .limit(1)
                .scalar_subquery()
                .correlate(WorkersOrm)
            )

            query = (
                select(WorkersOrm)
                .join(ResumesOrm, ResumesOrm.id.in_(subq))
                .options(contains_eager(WorkersOrm.resumes))
            )

            res = session.execute(query)
            result = res.unique().scalars().all()
            print(result)


Также можно через __table_args__ задавать парамтеры всей модели, например,
чтобы зп была строго больше нуля (checkconstraint)
И можно задавать индексы (Index)

Финальный вид моделей на данный момент:

class WorkersOrm(Base):
    __tablename__ = "workers"

    id: Mapped[intpk]
    username: Mapped[str]

    resumes: Mapped[list["ResumesOrm"]] = relationship(
        back_populates="worker",
    )

    resumes_parttime: Mapped[list["ResumesOrm"]] = relationship(
        back_populates="worker",
        primaryjoin="and_(WorkersOrm.id == ResumesOrm.worker_id, ResumesOrm.workload == 'parttime')",
        order_by="ResumesOrm.id.desc()",
    )


class ResumesOrm(Base):
    __tablename__ = "resumes"

    id: Mapped[intpk]
    title: Mapped[str_256]
    compensation: Mapped[Optional[int]]
    workload: Mapped[Workload]
    worker_id: Mapped[int] = mapped_column(ForeignKey("workers.id", ondelete="CASCADE"))
    created_at: Mapped[created_at]
    updated_at: Mapped[updated_at]

    worker: Mapped["WorkersOrm"] = relationship(
        back_populates="resumes",
    )

    __table_args__ = (
        Index("title_index", "title"),
        CheckConstraint("compensation > 0", name="checl_compensation_positive"),
    )

9) Pydantic - инструмент для валидации структур данных. Модели алхимии следует валидировать,
прописывая к ним схемы. Прниято сначала прописывать схему добавления сущности, а наследуясь от
нее прописывать схему самой сущности. А уже от схему сущнсоти наследовать схему отношения с другими
сущностями

Описываем схемы, например вот так

class WorkersAddDTO(BaseModel):
    username: str
    
    
class WorkersDTO(WorkersAddDTO):
    id: int


class WorkersRelDTO(WorkersDTO):
    resumes: list["ResumesDTO"]

И можем по ним валидировать результат наших запросов, например:

with session.factory() as session:
    query = (
        select(WorkersORM)
        .options(selectinload(WorkersORM.resumes))
        .limit(2)
    )
    res = session.execute(query)
    result_orm = res.scalars().all()
    result_dto = [WorkersRelDTO.model_validate(row, from_attributes=True) for row in result_orm]

Т.е. мы с помощью метода model_validate() итерируемся по каждому элементу ответа на ОРМ запрос,
приводя его в соответствие со схемой, т.е. валидируем ответ. (Сам метод не итерируется, а валидирует)

from_attributes=True - нужно, чтобы pydantic вычленял значения прямо из аттрибутов, т.е. чтобы мог обращаться
как к экземпляру класса, а не только как к словарю.

Ну а как fastapi подключать ты итак знаешь

10) Связь M-2-M через декларативный стиль!!! Ура-ура!!!


Вот пример связующей модели:
То есть по стандарту - 2 внешних ключа на зависимые моедли, они же первичные ключи для этой,
можно опционально добавить еще полей

class VacanciesRepliesOrm(Base):
    __tablename__ = "vacancies_replies"
    resume_id: Mapped[int] = mapped_column(
        ForeignKey("resumes.id", ondelete="CASCADE"),
        primary_key=True,
    )
    vacancy_id: Mapped[int] = mapped_column(
        ForeignKey("vacancies.id", ondelete="CASCADE"),
        primary_key=True,
    )
    cover_letter: Mapped[Optional[str]]

Вот новая зависимая модель вакансий:

class VacanciesOrm(Base):
    __tablename__ = "vacancies"
    id: Mapped[intpk]
    title: Mapped[str_256]
    compensation: Mapped[Optional[int]]
    resumes_replied: Mapped[list["ResumesOrm"]] = relationship(
        back_populates="vacancies_replied", # ссылка на поле в модели с которой связываемся
        secondary="vacancies_replies", # ссылка на модель зависимостей
    )

А вот старая обновленная модель резюме:

class ResumesOrm(Base):
    __tablename__ = "resumes"
    id: Mapped[intpk]
    title: Mapped[str_256]
    compensation: Mapped[Optional[int]]
    workload: Mapped[Workload]
    worker_id: Mapped[int] = mapped_column(ForeignKey("workers.id", ondelete="CASCADE"))
    created_at: Mapped[created_at]
    updated_at: Mapped[updated_at]
    worker: Mapped["WorkersOrm"] = relationship(
        back_populates="resumes",
    )
    vacancies_replied: Mapped[list["VacanciesOrm"]] = relationship(
        back_populates="resumes_replied", # ссылка на поле в модели с которой связываемся
        secondary="vacancies_replies", # ссылка на модель зависимостей
    )
    __table_args__ = (
        Index("title_index", "title"),
        CheckConstraint("compensation > 0", name="checl_compensation_positive"),
    )

Вот так мы добавляем сущности, то есть получаем сущность, обращаемся к relationship полю
соответствующему у нее и добавляем к ней нужную сущнсоть, затем коммитим, все просто:
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
                select(ResumesOrm)
                .options(joinedload(ResumesOrm.worker))
                .options(selectinload(ResumesOrm.vacancies_replied).load_only(VacanciesOrm.title))
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

11) Миграции крутая штука для версионирвоания изменений в БД. Знаем, применяли, продублируем.

alembic - либа для миграций

alembic init src/migrations - для инициализации миграций

alembic.ini - автоматически сгенереный файл конфигурации алембика

script_location - хранит путь до миграций

prepend_sys_path - хранит стандартную корневую диреткорию

script.py.mako - скрипт с шаблоно генерации скриптов миграций



env.py - файл, в котором тоже важные настройки хранятся

Для подлкючения к базе данных:
from config import settings

config.set_main_option('sqlalchemy.url', settings.DATABASE_URL_asyncpg + "?async_fallback=True")


("?async_fallback=True" - обязательно для асинхронного движка)

Для подключения моделей:
from database import Base (или from models import Base)
from models import WorkersORM  !!!Обязательно импортируй модели, иначе их метадата не притянется!!!

target_matedata = Base.metadata # здесь хранится инфа о таблицах



alembic revision --autogenerate - создание миграций локально, где флаг указывает на необходимость сравнения
моделей в коде с моделями в базе данных

алембик сгенерирует файл ревизии со скриптами апгрейд и даунгрейд для обновления/отката версии моделей в бд

alembic upgrade head - обновление до последней версии

alembic upgrade <№ ревизиии> обновит до конкретной ревизии

compare_server_default=True в env.py в параметрах context.configure() метода run_migrations_online - позволяет
безопасно удалять ревизии, которые не были прогнаны

alembic downgrade <№ ревизиии> откат до конкретной ревизии

alembic downgrade base откат до начального состояния

