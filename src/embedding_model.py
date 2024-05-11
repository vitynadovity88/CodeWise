from langchain_community.embeddings import OllamaEmbeddings
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
import config
import os


load_dotenv()


def get_embedding_model_and_index(model=config.SELECTED_MODEL):
    if model == "gpt-3.5-turbo":
        embeddings = OpenAIEmbeddings(model=config.OPENAI_EMBEDDING_MODEL)
        vector_index = config.CODE_REPO_INDEX_OPENAI
    else:
        embeddings = OllamaEmbeddings(model=config.LLAMA_EMBEDDING_MODEL)
        vector_index = config.CODE_REPO_INDEX_LLAMA
    return embeddings, vector_index
