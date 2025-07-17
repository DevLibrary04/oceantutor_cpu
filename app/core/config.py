from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    DATABASE_URL: str
    GEMINI_APIKEY: str
    BASE_PATH: Path
    SECRET_KEY: str


settings = Settings()
