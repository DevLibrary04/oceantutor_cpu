from sqlmodel import SQLModel
from database import run_engine
import models

engine = run_engine()
if engine:
    SQLModel.metadata.drop_all(engine)
    SQLModel.metadata.create_all(engine)
