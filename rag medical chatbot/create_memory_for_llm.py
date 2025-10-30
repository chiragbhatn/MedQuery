import os
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


# Step 1: Load raw PDF(s)
DATA_PATH = "data/"


def load_pdf_files(data):
    loader = DirectoryLoader(data,
                             glob='*.pdf',
                             loader_cls=PyPDFLoader)

    documents = loader.load()
    return documents


print("Loading documents...")
documents = load_pdf_files(data=DATA_PATH)
print(f"Loaded {len(documents)} documents.")


# Step 2: Create Chunks
def create_chunks(extracted_data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800,
                                                   chunk_overlap=200)
    text_chunks = text_splitter.split_documents(extracted_data)
    return text_chunks


print("Splitting documents into chunks...")
text_chunks = create_chunks(extracted_data=documents)
print(f"Split documents into {len(text_chunks)} chunks.")


# Step 3: Create Vector Embeddings

def get_embedding_model():
    # This will download the model "sentence-transformers/all-MiniLM-L6-v2"
    # to your local machine the first time you run it.
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return embedding_model


print("Loading embedding model...")
embedding_model = get_embedding_model()
print("Embedding model loaded.")

# Step 4: Store embeddings in FAISS
DB_FAISS_PATH = "vectorstore/db_faiss"

# Ensure the directory exists
os.makedirs(DB_FAISS_PATH, exist_ok=True)

print("Creating and saving FAISS vector store...")
db = FAISS.from_documents(text_chunks, embedding_model)
db.save_local(DB_FAISS_PATH)
print(f"Vector store saved to {DB_FAISS_PATH}")