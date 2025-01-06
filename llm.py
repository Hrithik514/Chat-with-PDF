import os
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

# Title
st.title("Chat with Your PDF")

# File Uploader
uploaded_file = st.file_uploader("Upload your PDF file", type=["pdf"])
if uploaded_file:
    # Save uploaded file temporarily
    temp_file_path = f"temp_{uploaded_file.name}"
    with open(temp_file_path, "wb") as f:
        f.write(uploaded_file.read())

    try:
        # Debug: Loading PDF
        st.write("Loading PDF...")
        loader = PyPDFLoader(temp_file_path)
        docs = loader.load()
        st.write("PDF loaded into documents successfully.")

        # Debug: Splitting text
        st.write("Splitting PDF into chunks...")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
        documents = text_splitter.split_documents(docs)
        st.write(f"PDF split into {len(documents)} chunks.")

        # Debug: Creating vector database
        st.write("Creating vector database...")
        db = FAISS.from_documents(documents, OllamaEmbeddings(model="gemma"))
        st.write("Vector database created successfully.")

        # Debug: Setting up chain
        st.write("Setting up retrieval chain...")
        retriever = db.as_retriever()
        llm = Ollama(model="gemma")
        prompt = ChatPromptTemplate.from_template("""
        Answer the following question based only on the provided context. 
        Think step by step before providing a detailed answer. 
        <context>
        {context}
        </context>
        Question: {input}""")
        document_chain = create_stuff_documents_chain(llm, prompt)
        retrieval_chain = create_retrieval_chain(retriever, document_chain)
        st.success("Your PDF is ready for the chat!")

    except Exception as e:
        st.error(f"Error during PDF processing: {e}")
    finally:
        # Cleanup temporary file
        os.remove(temp_file_path)

    # Chat Interface
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    user_input = st.text_input("Ask a question about the PDF:")

    if user_input:
        try:
            st.write("Processing your question...")
            response = retrieval_chain.invoke({"input": user_input})
            answer = response.get("answer", "Sorry, no answer found.")

            # Update chat history
            st.session_state.chat_history.append((user_input, answer))

        except Exception as e:
            st.error(f"Error while answering your question: {e}")

    # Display Chat History
    if st.session_state.chat_history:
        st.write("### Chat History")
        for question, answer in st.session_state.chat_history:
            st.markdown(f"**You:** {question}")
            st.markdown(f"**AI:** {answer}")

else:
    st.info("Please upload a PDF to start.")
