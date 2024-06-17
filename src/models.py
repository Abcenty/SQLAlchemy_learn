import datetime
import enum
from typing import Optional, Annotated
from sqlalchemy import CheckConstraint, Index, Table, Column, Integer, String, MetaData, ForeignKey, text
from sqlalchemy.orm import Mapped, mapped_column,relationship # позволяет задавать тип столбца
from sqlalchemy.ext.declarative import declarative_base

intpk = Annotated[int, mapped_column(primary_key=True)]
created_at = Annotated[datetime.datetime, mapped_column(server_default=text("TIMEZONE('utc', now())"))]
updated_at = Annotated[datetime.datetime, mapped_column(
        server_default=text("TIMEZONE('utc', now())"),
        onupdate=datetime.UTC
        )]  

Base = declarative_base()

# декларативный стиль описания моделей
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
    
    
class Workload(enum.Enum):
    parttime = 'parttime'
    fulltime = 'fulltime'
    
    
class ResumesOrm(Base):
    __tablename__ = "resumes"
    
    id: Mapped[intpk]
    title: Mapped[str]
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
    
    
class VacanciesOrm(Base):
    __tablename__ = "vacancies"
    id: Mapped[intpk]
    title: Mapped[str]
    compensation: Mapped[Optional[int]]
    resumes_replied: Mapped[list["ResumesOrm"]] = relationship(
        back_populates="vacancies_replied", # ссылка на поле в модели с которой связываемся
        secondary="vacancies_replies", # ссылка на модель зависимостей
    )

    
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









metadata_obj = MetaData() # данные о таблицах, для мигарций и взаимодействия с таблицами

# императивный стиль описания моделей
workers_table = Table(
    "workers",
    metadata_obj,
    Column('id', Integer, primary_key=True),
    Column('username', String),
)