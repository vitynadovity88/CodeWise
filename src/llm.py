from langchain_community.chat_models.openai import ChatOpenAI
from dotenv import load_dotenv
import config
import os

load_dotenv()

HF_ACCESS_TOKEN = os.environ.get("HUGGINGFACEHUB_API_TOKEN")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
OPENAI_ORG = os.environ.get("OPENAI_ORG")


def get_llm(selected_model="gpt-3.5-turbo", max_tokens=1000, temperature=0.3):
    if selected_model == "gpt-3.5-turbo":
        model = ChatOpenAI(
            model_name="gpt-3.5-turbo",
            openai_api_key=OPENAI_API_KEY,
            openai_organization=OPENAI_ORG,
            max_tokens=max_tokens,
            temperature=temperature,
        )
    else:
        model = ChatOpenAI(
            model_name="tgi",
            openai_api_key=HF_ACCESS_TOKEN,
            openai_api_base=config.LLAMA_MODEL_NAME + "/v1/",
            max_tokens=max_tokens,
            temperature=temperature,
        )

    return model
