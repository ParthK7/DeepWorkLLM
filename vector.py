from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os

# Load and split PDF into documents
loader = PyPDFLoader("Deep-Work.pdf")
pages = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
documents = splitter.split_documents(pages)

# Create embeddings
embeddings = OllamaEmbeddings(model="mxbai-embed-large")
db_location = "./chrome_langchain_db"
add_documents = not os.path.exists(db_location)

# Initialize Chroma vector store
vector_store = Chroma(
    collection_name="restaurant_reviews",
    persist_directory=db_location,
    embedding_function=embeddings
)

if add_documents:
    # Insert in chunks to avoid exceeding Chroma's max batch size
    batch_size = 150  # Keep this < 166
    for i in range(0, len(documents), batch_size):
        batch_docs = documents[i:i+batch_size]
        batch_ids = [str(j) for j in range(i, i + len(batch_docs))]
        vector_store.add_documents(documents=batch_docs, ids=batch_ids)


retriever = vector_store.as_retriever(
    search_kwargs={"k": 5}
)