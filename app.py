import gradio as gr
from langchain_chroma import Chroma
from dotenv import load_dotenv

import config
from src.prompts import get_chat_template, get_contexual_prompt
from src.llm import get_llm
from src.embedding_model import get_embedding_model_and_index
from langchain.chains import create_history_aware_retriever
from langchain_core.messages import HumanMessage
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
import os

load_dotenv()
HF_ACCESS_TOKEN = os.environ.get("HUGGINGFACEHUB_API_TOKEN")
selected_model = config.SELECTED_MODEL
chat_history = []

greeting_message = """
Hey! I'm here to assist you with any programming or software development questions related to the code repository at https://github.com/vitynadovity88/staking. How can I help you today?
"""


def get_retriever(embeddings, index):
    db = Chroma(persist_directory=index, embedding_function=embeddings)
    return db.as_retriever()


async def respond(
    question,
    history: list[tuple[str, str]],
    system_message="You are a friendly Chatbot.",
    max_tokens=1000,
    temperature=0.3,
):

    model = get_llm(selected_model, max_tokens, temperature)
    embeddings, index = get_embedding_model_and_index()
    retriever = get_retriever(embeddings, index)

    context_retriever_prompt = get_contexual_prompt()
    history_aware_retriever = create_history_aware_retriever(
        model, retriever, context_retriever_prompt
    )

    chat_prompt = get_chat_template()
    qa_chain = create_stuff_documents_chain(model, chat_prompt)

    rag_chain = create_retrieval_chain(history_aware_retriever, qa_chain)

    supported_documents = []
    response = ""

    async for chunk in rag_chain.astream(
        {"input": question, "chat_history": chat_history}
    ):
        try:
            if "context" in chunk.keys():
                supported_documents = list(
                    set([each.metadata["source"] for each in chunk["context"]])
                )

        except:
            continue
        if len(chunk.get("answer", "")) > 0:
            answer = chunk["answer"]
            response += answer
            yield response

    chat_history.extend([HumanMessage(content=question), response])
    if len(supported_documents) > 0:
        response += (
            "\n\n"
            + "Sources Link: "
            + "\n"
            + "\n".join([each for each in supported_documents])
        )
        yield response


# gradio app
demo = gr.ChatInterface(
    respond,
    chatbot=gr.Chatbot(value=[[None, greeting_message]], height=300),
    textbox=gr.Textbox(placeholder="Type your query here..", container=False, scale=7),
    title="Staking Solana",
    theme="soft",
    retry_btn=None,
    undo_btn="Delete Previous",
    clear_btn="Clear",
    examples=[
        ["Explain the main functionality of my Staking Solana program"],
        [
            "Help writting integration tests using anchor and examples provided in the repository?"
        ],
    ],
    cache_examples=True,
    additional_inputs=[
        gr.Textbox(value="You are a friendly Chatbot.", label="System message"),
        gr.Slider(minimum=1, maximum=2048, value=1000, step=1, label="Max new tokens"),
        gr.Slider(minimum=0.1, maximum=1.0, value=0.7, step=0.1, label="Temperature"),
    ],
)


if __name__ == "__main__":
    demo.launch(share=True)
