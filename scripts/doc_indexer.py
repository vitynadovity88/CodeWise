from dotenv import load_dotenv
import os
import config
from langchain.document_loaders import GithubFileLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import (
    Language,
    RecursiveCharacterTextSplitter,
)
from src.embedding_model import get_embedding_model_and_index

load_dotenv()
GITHUB_ACCESS_TOKEN = os.environ.get("GITHUB_ACCESS_TOKEN")

language_map = {".html": Language.HTML, ".py": Language.PYTHON}


def load_relevant_files(repo, api_link, file_extension):
    # fetch repository from gitHub
    loader = GithubFileLoader(
        repo=repo,  # the repo name
        access_token=GITHUB_ACCESS_TOKEN,  # github api access token
        github_api_url=api_link,
        file_filter=lambda file_path: file_path.endswith(
            file_extension
        ),  # filter files with a specific extension
    )
    return loader.load()


# split documents into chunks
def split_content(documents, chunk_size, chunk_overlap, file_extension):
    splitter = RecursiveCharacterTextSplitter.from_language(
        language=language_map[file_extension],  # specify the language
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return splitter.split_documents(documents)


def format_docs_metadata(documents):
    for doc in documents:
        new_url = doc.metadata["source"]
        new_url = new_url.replace("api.", "")
        doc.metadata.update({"source": new_url})  # fixing the path for the source files
    return documents


def index_and_upload():
    # load and split documents
    repo = config.GITHUB_REPO
    github_api = config.GITHUB_API_URL
    all_documents = []
    for each_type in config.REPO_FILETYPES:
        documents = load_relevant_files(
            repo, github_api, each_type
        )  # load files with a single extension
        split_documents = split_content(
            documents, config.CHUNK_SIZE, config.OVERLAP_SIZE, each_type
        )  # split documents into chunks
        all_documents.extend(split_documents)

    # format documents to fix source links
    all_documents = format_docs_metadata(all_documents)

    # embed and upload into vector db
    embeddings, index = get_embedding_model_and_index()

    print("Documents Indexing :: Initializing...")
    Chroma.from_documents(all_documents, embeddings, persist_directory=index)
    print(f"Uploaded {len(all_documents)} into VectorDB")


if __name__ == "__main__":
    index_and_upload()
