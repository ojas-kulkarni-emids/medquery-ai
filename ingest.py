from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# function to create vector store from pdf using hugging face embeddings
def create_huggingface_vector_store(pdf_path, persist_directory="./chroma_db"):
    # load pdf
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    # split into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_documents(documents)

    # create hugging face embeddings
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': False}
    
    hf_embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )

    # create vector store
    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=hf_embeddings,
        persist_directory=persist_directory
    )
    
    print(f"Success! Vector store saved to {persist_directory}")
    return vector_db

if __name__ == "__main__":
    my_local_db = create_huggingface_vector_store("1725951640_cardiology.pdf")