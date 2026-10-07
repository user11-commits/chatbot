"""Check RAG dependencies locally, without making API requests."""

import sys
from importlib.metadata import version
from io import StringIO

import langchain
import streamlit
from dotenv import dotenv_values
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


def main() -> None:
    assert sys.version_info[:2] == (3, 11), sys.version
    print(f"Python {sys.version.split()[0]}")
    for package in (
        "langchain", "langchain-openai", "langchain-text-splitters",
        "streamlit", "python-dotenv",
    ):
        print(f"{package}=={version(package)}")

    splitter = RecursiveCharacterTextSplitter(chunk_size=80, chunk_overlap=10)
    chunks = splitter.split_documents(
        [Document(page_content="RAG environment check. " * 20)]
    )
    assert len(chunks) > 1
    assert all(len(chunk.page_content) <= 80 for chunk in chunks)
    prompt = ChatPromptTemplate.from_messages([("human", "{question}")])
    assert prompt.invoke({"question": "Hello"}).to_messages()[0].content == "Hello"
    assert dotenv_values(stream=StringIO("CHECK_VALUE=ready"))["CHECK_VALUE"] == "ready"

    # Dummy key only checks client construction; no network requests are made.
    chat = ChatOpenAI(api_key="sk-local-import-test")
    embeddings = OpenAIEmbeddings(api_key="sk-local-import-test")
    assert chat.client is not None
    assert embeddings.client is not None
    print("PASS: imports, splitting, prompts, dotenv, and OpenAI client construction")


if __name__ == "__main__":
    main()
