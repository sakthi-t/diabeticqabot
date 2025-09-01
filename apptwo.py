# app.py
import os
from dotenv import load_dotenv
import gradio as gr
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_community.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI

# ------------------ Load environment variables ------------------
load_dotenv()
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

if OPENAI_API_KEY is None:
    raise ValueError("Please set your OPENAI_API_KEY in the .env file")

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

# ------------------ Gradio Chat Function ------------------
def chat(user_message):
    result = qa_chain({"question": user_message, "chat_history": memory.chat_memory.messages})
    return result["answer"]

# ------------------ Launch Gradio Interface ------------------
with gr.Blocks() as demo:
    gr.Markdown("## 💉 Diabetes Chatbot")
    gr.Markdown("Ask anything about diabetes. The bot remembers your session questions!")
    
    chatbot = gr.Chatbot()
    msg = gr.Textbox(placeholder="Type your question here...", show_label=False)
    submit = gr.Button("Send")

    def respond(message):
        answer = chat(message)
        chatbot.append((message, answer))
        return chatbot, ""

    submit.click(respond, inputs=msg, outputs=[chatbot, msg])
    msg.submit(respond, inputs=msg, outputs=[chatbot, msg])

if __name__ == "__main__":
    demo.launch()
