import os
import streamlit as st

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq

# ==========================================
# PAGE CONFIG
# ==========================================

st.set_page_config(
    page_title="Banking AI Assistant",
    page_icon="🏦",
    layout="wide"
)

st.title("🏦 Banking AI Assistant")
st.write("Ask questions from the SBI Banking PDF.")

# ==========================================
# API KEY
# ==========================================
from dotenv import load_dotenv
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    st.error("Groq API Key not found. Please check your .env file.")
    st.stop()


# ==========================================
# LOAD PDF ONLY ONCE
# ==========================================

@st.cache_resource
def load_vector_database():

    pdf_path = r"C:\Users\jyoti singh\Data Science\gen ai\SBI-BANKING\data\SBI - CPCR Booklet (English)_14.07.2025.pdf"

    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vector_db = FAISS.from_documents(
        chunks,
        embeddings
    )

    return vector_db

with st.spinner("Loading Banking Documents..."):

    vector_db = load_vector_database()

# ==========================================
# LOAD GROQ MODEL
# ==========================================

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0
)

# ==========================================
# USER QUESTION
# ==========================================

question = st.text_input(
    "Ask your Banking Question"
)

if st.button("Get Answer"):

    if question == "":
        st.warning("Please enter a question.")

    else:

        docs = vector_db.similarity_search(
            question,
            k=3
        )

        context = "\n\n".join(
            doc.page_content
            for doc in docs
        )

        prompt = f"""
You are an SBI Banking AI Assistant.

Answer ONLY from the given context.

If the answer is not available,
reply exactly:

I couldn't find that information in the uploaded banking documents.

Context:
{context}

Question:
{question}

Answer:
"""

        with st.spinner("Generating Answer..."):

            response = llm.invoke(prompt)

        st.success("Answer")

        st.write(response.content)