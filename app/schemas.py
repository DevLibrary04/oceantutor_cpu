from pydantic import BaseModel


# main.py
class RootResponse(BaseModel):
    message: str
    endpoints: str
