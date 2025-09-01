# app.py
import os


# Make Streamlit write config locally (avoids PermissionError in Spaces)
os.environ["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] = "false"  # disable usage stats
os.environ["STREAMLIT_CONFIG_DIR"] = os.getcwd()  # store Streamlit configs locally

# os.environ["STREAMLIT_CONFIG_DIR"] = ".streamlit"
os.environ["STREAMLIT_LOG_FOLDER"] = ".streamlit"
os.environ["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] = "false"


import streamlit as st
# from dotenv import load_dotenv
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_community.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI

# ------------------ Load environment variables ------------------
# load_dotenv()
OPENAI_API_KEY = os.environ.getenv("OPENAI_API_KEY")

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
        allow_dangerous_deserialization=True
    )
    return vectorstore

vectorstore = load_vectorstore(VECTORSTORE_PATH)
if vectorstore is None:
    st.stop()

# ------------------ Load LLM ------------------
@st.cache_resource
def load_llm():
    llm = ChatOpenAI(
        model_name="gpt-3.5-turbo",
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
st.write("Chat with the bot about diabetes. It remembers your questions during this session!")

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

# ------------------ Chat Interface ------------------
user_input = st.chat_input("Type your question here...")

if user_input:
    # Display user message instantly
    st.session_state["chat_history"].append((user_input, None))

    # Run QA chain and generate answer
    with st.spinner("Bot is thinking..."):
        result = qa_chain({"question": user_input, "chat_history": st.session_state["chat_history"]})
        answer = result["answer"]
        # Update the last user message with the bot response
        st.session_state["chat_history"][-1] = (user_input, answer)

# Display chat history using Streamlit chat messages
for q, a in st.session_state["chat_history"]:
    st.chat_message("user").write(q)
    st.chat_message("assistant").write(a)
