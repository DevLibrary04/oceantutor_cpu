from dotenv import load_dotenv
import os
from sqlmodel import create_engine, SQLModel

load_dotenv()


# SQLAlchemy 엔진

mysql_url = os.getenv("DATABASE_URL")
if mysql_url is not None:
    engine = create_engine(mysql_url, echo=True)


def run_engine():
    if engine is not None:
        return engine
        # SQLModel.metadata.create_all(engine)
    else:
        print("engine not created")
