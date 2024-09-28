from langchain.prompts import FewShotPromptTemplate, PromptTemplate
from prompts.email_examples import examples
from langchain_experimental.tabular_synthetic_data.prompts import (
    SYNTHETIC_FEW_SHOT_PREFIX,
    SYNTHETIC_FEW_SHOT_SUFFIX
)


# Define the prompt template for OpenAI
OPENAI_TEMPLATE = PromptTemplate(input_variables=["example"], template="{example}")

# Define the few-shot prompt
prompt_template = FewShotPromptTemplate(
    prefix=SYNTHETIC_FEW_SHOT_PREFIX,
    examples=examples,
    suffix=SYNTHETIC_FEW_SHOT_SUFFIX,
    input_variables=["subject", "extra"],
    example_prompt=OPENAI_TEMPLATE
)