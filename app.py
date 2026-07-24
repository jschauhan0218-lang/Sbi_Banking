import os
import streamlit as st

from dotenv import load_dotenv

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
# LOAD ENVIRONMENT VARIABLES
# ==========================================

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    st.error("GROQ_API_KEY not found.")
    st.stop()


# ==========================================
# LOAD VECTOR DATABASE
# ==========================================

@st.cache_resource
def load_vector_database():

    # Project folder
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    # PDF path
    pdf_path = os.path.join(
        BASE_DIR,
        "data",
        "SBI_CPCR_Booklet.pdf"
    )

    print("PDF Path:", pdf_path)
    print("Exists:", os.path.exists(pdf_path))


    if not os.path.exists(pdf_path):
        raise FileNotFoundError(
            f"PDF not found: {pdf_path}"
        )


    # Load PDF

    loader = PyPDFLoader(pdf_path)

    documents = loader.load()


    # Split text into chunks

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.split_documents(documents)


    # Create embeddings

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


    # Create vector database

    vector_db = FAISS.from_documents(
        chunks,
        embeddings
    )


    return vector_db



# Load database

with st.spinner("Loading Banking Documents..."):

    vector_db = load_vector_database()



# ==========================================
# LOAD GROQ MODEL
# ==========================================

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0,
    api_key=GROQ_API_KEY
)



# ==========================================
# USER INPUT
# ==========================================

question = st.text_input(
    "Ask your Banking Question"
)



if st.button("Get Answer"):


    if question.strip() == "":

        st.warning(
            "Please enter a question."
        )


    else:


        # Retrieve relevant documents

        docs = vector_db.similarity_search(
            question,
            k=3
        )


        # Create context

        context = "\n\n".join(
            doc.page_content
            for doc in docs
        )


        # Prompt

        prompt = f"""
You are an SBI Banking AI Assistant.

Answer ONLY from the given context.

If the answer is not available, reply exactly:

I couldn't find that information in the uploaded banking documents.


Context:

{context}


Question:

{question}


Answer:
"""


        # Generate answer

        with st.spinner("Generating Answer..."):

            response = llm.invoke(prompt)


        st.success("Answer")

        st.write(response.content)