"""
This file is created to handle all AI logic of the project, like:

reading documents
understanding text
storing knowledge
answering questions using AI

👉 In short:

“This file converts your document into a smart searchable AI knowledge system.”
"""
import tempfile
import os
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq


# -----------------------------
# LOAD .env FILE
# -----------------------------
load_dotenv()

API_KEY = os.getenv("GROQ_API_KEY")


# safety check (optional but smart)
if not API_KEY:
    raise ValueError("❌ GROQ_API_KEY not found in .env file")


# embeddings model
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


# -----------------------------
# 1. DOCUMENT PROCESSING
# -----------------------------
def process_document(file):
    suffix = ".pdf" if file.name.endswith(".pdf") else ".txt"

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(file.read())
        path = tmp.name

    if suffix == ".pdf":
        loader = PyPDFLoader(path)
    else:
        loader = TextLoader(path, encoding="utf-8")

    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    chunks = splitter.split_documents(docs)

    embeddings = get_embeddings()

    vectorstore = FAISS.from_documents(chunks, embeddings)

    os.remove(path)

    return vectorstore


# -----------------------------
# 2. ASK QUESTION (RAG)
# -----------------------------
def ask_question(vectorstore, question):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    docs = retriever.invoke(question)

    context = "\n\n".join([d.page_content for d in docs])

    llm = ChatGroq(
        api_key=API_KEY,
        model="llama-3.1-8b-instant",
        temperature=0.3
    )

    prompt = f"""
You are a helpful assistant.
Use ONLY the context below.

Context:
{context}

Question:
{question}

Answer in simple language:
"""

    response = llm.invoke(prompt)

    return response.content