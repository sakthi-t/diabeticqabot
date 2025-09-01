# app.py
import os
import streamlit as st
from dotenv import load_dotenv
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_community.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI

# ------------------ Load environment variables ------------------
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# ------------------ Paths ------------------
VECTORSTORE_PATH = os.path.join("storage", "faiss_index")  # folder containing index.faiss and index.pkl

# ------------------ Load vectorstore ------------------
@st.cache_resource
def load_vectorstore(path):
    if not os.path.exists(path):
        st.error(f"FAISS index not found at {path}. Please run ingest.py first.")
        return None
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    vectorstore = FAISS.load_local(
        path,
        embeddings,
        allow_dangerous_deserialization=True  # necessary for pickle-based index
    )
    return vectorstore

vectorstore = load_vectorstore(VECTORSTORE_PATH)
if vectorstore is None:
    st.stop()

# ------------------ Load LLM ------------------
@st.cache_resource
def load_llm():
    llm = ChatOpenAI(
        model_name="gpt-3.5-turbo",  # or "gpt-4" if you have access
        temperature=0,
        openai_api_key=OPENAI_API_KEY
    )
    return llm

llm = load_llm()

# ------------------ Memory ------------------
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

# ------------------ Conversational Retrieval Chain ------------------
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
    memory=memory,
    output_key="answer"
)

# ------------------ Streamlit UI ------------------
st.title("💉 Diabetes Chatbot")
st.write("Ask anything about diabetes from the PDF document.")

if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

user_input = st.text_input("Your question:")
if user_input:
    with st.spinner("Generating answer..."):
        try:
            result = qa_chain({"question": user_input})
            answer = result["answer"]
            st.session_state["chat_history"].append((user_input, answer))
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.stop()

# Display chat history
if st.session_state["chat_history"]:
    for i, (q, a) in enumerate(st.session_state["chat_history"]):
        st.markdown(f"**You:** {q}")
        st.markdown(f"**Bot:** {a}")
        st.markdown("---")
