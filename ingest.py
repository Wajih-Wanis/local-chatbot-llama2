from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import	PyPDFLoader,DirectoryLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

dataPath = "data/"
dbFaissPath = "vectorstores/db_faiss"

#Vector database creation 
def createVectorDb():
    loader = DirectoryLoader(dataPath,glob='*.pdf',loader_cls=PyPDFLoader)
    documents = loader.load()
    textSplitter = RecursiveCharacterTextSplitter(chunk_size= 500, chunk_overlap= 50)
    texts = textSplitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2',
    model_kwargs={'device':'cpu'})
    
    db = FAISS.from_documents(texts,embeddings)
    db.save_local(dbFaissPath)

if __name__ == '__main__':
    createVectorDb()
    
