from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma


PERSIST_DIR = "./chroma_db"

def web_document_loader(web_link):
    """
    Load a Web Page file and convert it into LangChain Document objects.
    """
    print(f"[INFO] Loading Web Page: {web_link}")
    loader = WebBaseLoader(web_link)
    documents = loader.load()
    print(f"[INFO] Completed loading WebPage: {web_link}")
    return documents

def split_chunks(docs, chunk_size=1000, chunk_overlap=100):
    """
    Split documents into smaller overlapping chunks
    to improve embedding and retrieval accuracy.
    """
    print("[INFO] Splitting documents into chunks")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True,
    )
    return splitter.split_documents(docs)

def create_embeddings():
    """
    Initialize Google Gemini embedding model.
    """
    print("[INFO] Initializing Gemini embedding model")
    return GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001"
    )

def create_vector_store(embeddings):
    """
    Create or load a Chroma vector store for persisting embeddings.
    """
    print("[INFO] Creating vector store")
    return Chroma(
        collection_name="handbook",
        embedding_function=embeddings,
        persist_directory=PERSIST_DIR,
    )

def main():
    """
    End-to-end semantic search pipeline:
    WebPage → Chunking → Embedding → Vector Store → Similarity Search
    """
    web_link = 'https://docs.cloud.google.com/vpc/docs/private-access-options'

    # Load and preprocess documents
    documents = web_document_loader(web_link)
    chunks = split_chunks(documents, chunk_overlap=200)

    # Initialize embeddings and vector store
    embeddings = create_embeddings()
    vector_store = create_vector_store(embeddings)

    # Store document embeddings
    print("[INFO] Adding documents to vector store")
    vector_store.add_documents(chunks)



if __name__ == "__main__":
    main()