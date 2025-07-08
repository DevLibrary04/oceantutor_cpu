from dotenv import load_dotenv
import os
from sqlmodel import create_engine, Session

load_dotenv()


# SQLAlchemy 엔진

mysql_url = os.getenv("DATABASE_URL")
if not mysql_url:
    raise ValueError("Environment variable not loaded")
engine = create_engine(mysql_url, echo=True)


def get_db():
    db = Session(engine)
    try:
        yield db
    finally:
        db.close()
