from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import	PyPDFLoader,DirectoryLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

DATA_PATH = "data/"
DB_FAISS_PATH = "vectorstores/db_faiss"

#Vector database creation 
def createVectorDb():
    loader = DirectoryLoader(DATA_PATH,glob='*.pdf',loader_cls=PyPDFLoader)
    documents = loader.load()
    textSplitter = RecursiveCharacterTextSplitter(chunk_size= 500, chunk_overlap= 50)
    texts = textSplitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2',
    model_kwargs={'device':'cpu'})
    
    db = FAISS.from_documents(texts,embeddings)
    db.save_local(DB_FAISS_PATH)

if __name__ == '__main__':
    createVectorDb()
    
