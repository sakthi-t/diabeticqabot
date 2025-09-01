import os
from dotenv import load_dotenv

# Correct imports
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS  # use community version

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Paths
pdf_path = os.path.join("data", "diabetes.pdf")
out_dir = os.path.join("storage", "faiss_index")
os.makedirs(out_dir, exist_ok=True)

print("📄 Loading PDF...")
loader = PyPDFLoader(pdf_path)
docs = loader.load()

print("✂️ Splitting into chunks...")
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_documents(docs)

print("🧠 Creating embeddings with OpenAI...")
embeddings = OpenAIEmbeddings()

print("📦 Building FAISS index...")
vectorstore = FAISS.from_documents(chunks, embeddings)

vectorstore.save_local(out_dir)
print(f"✅ FAISS index created at {out_dir}")

