from langchain_experimental.tabular_synthetic_data.openai import create_openai_data_generator
from langchain.chat_models import ChatOpenAI
from models.email_model import Email
from prompts.prompt_template import prompt_template


# Create the synthetic data generator
def create_email_data_generator():
    return create_openai_data_generator(
        output_schema=Email,
        llm=ChatOpenAI(model="gpt-4o"),
        prompt=prompt_template
    )
