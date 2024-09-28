from pydantic import BaseModel

# Define the Email schema using Pydantic
class Email(BaseModel):
    email: str
    label: str