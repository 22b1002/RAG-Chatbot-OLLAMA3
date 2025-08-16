import streamlit as st
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import ConversationalRetrievalChain

load_dotenv()

st.set_page_config(page_title="RAG Chatbot", page_icon="🤖")

st.title("📄 RAG Chatbot with Streamlit")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    st.error("Please set your OPENAI_API_KEY in the .env file")
    st.stop()

uploaded_files = st.file_uploader("Upload PDF(s) for knowledge base", type=["pdf"], accept_multiple_files=True)

if uploaded_files:
    documents = []
    for uploaded_file in uploaded_files:
        temp_path = os.path.join("temp", uploaded_file.name)
        os.makedirs("temp", exist_ok=True)
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        loader = PyPDFLoader(temp_path)
        documents.extend(loader.load())

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = text_splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    vectorstore = FAISS.from_documents(chunks, embeddings)

    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=ChatOpenAI(temperature=0, openai_api_key=OPENAI_API_KEY),
        retriever=vectorstore.as_retriever(),
    )

    chat_history = []
    user_query = st.text_input("Ask a question:")
    if user_query:
        response = qa_chain({"question": user_query, "chat_history": chat_history})
        chat_history.append((user_query, response["answer"]))
        st.markdown(f"**Answer:** {response['answer']}")
