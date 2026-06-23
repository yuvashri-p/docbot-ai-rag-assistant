# rag_pipeline.py

import os
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma

load_dotenv()

# Support both local .env AND Streamlit Cloud secrets
import streamlit as st
if not os.getenv("GOOGLE_API_KEY"):
    try:
        os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
    except Exception:
        pass


def process_pdf(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = splitter.split_text(text)
    print(f"✅ Created {len(chunks)} chunks")
    return chunks


def create_vectorstore(chunks):
    embeddings = GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-001",  # ← CORRECT model name
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )
    vectorstore = Chroma.from_texts(
        texts=chunks,
        embedding=embeddings,
        persist_directory="chroma_db"
    )
    return vectorstore


def get_qa_chain(vectorstore):
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",  # ← CORRECT model name
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=0.3
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    return llm, retriever


def ask(chain_tuple, question):
    llm, retriever = chain_tuple
    docs = retriever.invoke(question)
    context = "\n\n".join([doc.page_content for doc in docs])

    prompt = f"""You are a helpful study assistant.

Use the content below from the student's document to answer the question.
If the document doesn't fully explain it, use your own knowledge to fill gaps
with simple language, formulas, and examples.

Document content:
{context}

Question: {question}

Answer clearly with simple words, and include formulas or examples if helpful:"""

    response = llm.invoke(prompt)
    return response.content, docs