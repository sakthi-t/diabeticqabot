# app.py
import os
from dotenv import load_dotenv
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.chat_models import ChatOpenAI
import gradio as gr

# ------------------ Load environment variables ------------------
# load_dotenv()
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

# ------------------ Paths ------------------
VECTORSTORE_PATH = os.path.join("storage", "faiss_index")  # folder containing index.faiss and index.pkl

# ------------------ Load vectorstore ------------------
def load_vectorstore(path):
    if not os.path.exists(path):
        raise ValueError(f"FAISS index not found at {path}. Please run ingest.py first.")
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    vectorstore = FAISS.load_local(
        path,
        embeddings,
        allow_dangerous_deserialization=True
    )
    return vectorstore

vectorstore = load_vectorstore(VECTORSTORE_PATH)

# ------------------ Load LLM ------------------
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

# ------------------ Gradio Chat ------------------
def respond(user_message, chat_history):
    if user_message:
        try:
            result = qa_chain({"question": user_message, "chat_history": memory.chat_memory.messages})
            answer = result["answer"]
        except Exception as e:
            answer = f"Error: {str(e)}"
        chat_history.append((user_message, answer))
    return chat_history, chat_history

with gr.Blocks() as demo:
    gr.Markdown("## 💉 Diabetes Chatbot\nChat with the bot about diabetes. It remembers your questions during this session!")
    chatbot = gr.Chatbot()
    user_input = gr.Textbox(label="Type your question here...", placeholder="Ask anything about diabetes...", lines=1)
    user_input.submit(respond, [user_input, chatbot], [chatbot, chatbot])

demo.launch()
