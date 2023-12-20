from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.prompts import PromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import CTransformers
from langchain.chains import RetrievalQA
import chainlit as cl


DB_FAISS_PATH = "vectorstores/db_faiss/"

CUSTOM_PROMPT_TEMPLATE = """Use the following pieces of information to answer the user's question.
If you don't know the answer, please just say that you don't know the answer, don't try to make up an answer.

Context = {context}
Question = {question}

Only return the helpful answer below and nothing else.
Helpful answer: 
"""

def setCustomPrompt():
    #Prompt template for QA retrieval for each vector stores 

    prompt = PromptTemplate(template=CUSTOM_PROMPT_TEMPLATE,
                            input_variables=['context','question'])

    return prompt

def loadLLM():
    llm = CTransformers(
        #Change this model to the one you are using
        model="llama-2-13b-chat.ggmlv3.q8_0.bin",
        model_type="llama",
        max_new_tokens=1024,
        temperature= 0.5
    )
    return llm

def retrievalQaChain(llm,prompt,db):
    qaChain = RetrievalQA.from_chain_type(
        llm = llm,
        chain_type= "stuff",
        retriever = db.as_retriever(search_kwargs={'k':2}),
        return_source_documents = True,
        chain_type_kwargs= {'prompt':prompt}
    )
    return qaChain

def qaBot():
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2',
    model_kwargs={'device':'cpu'})

    db = FAISS.load_local(DB_FAISS_PATH, embeddings)
    llm = loadLLM()
    qaPrompt = setCustomPrompt()
    qa = retrievalQaChain(llm, qaPrompt, db)
    
    return qa

def finalResult(query):
    qaResult = qaBot()
    response = qaResult({'query':query})
    return response

## ** Chainlit ** ##

@cl.on_chat_start
async def start():
    chain = qaBot()
    msg = cl.Message(content="Starting the bot...")
    await msg.send()
    msg.content = "Hello Wajih, How can I help you today"
    await msg.update()
    cl.user_session.set("chain", chain)

@cl.on_message
async def main(message):
    chain = cl.user_session.get("chain")
    cb = cl.AsyncLangchainCallbackHandler(
        stream_final_answer=True,
        answer_prefix_tokens= ["FINAL","ANSWER"]
    )
    cb.answer_reached = True
    res = await chain.acall(message,callbacks=[cb])
    answer = res["result"]
    sources = res["source_documents"]
    
    if sources:
        answer += f"\nSources:" + str(sources)
    else:
        answer += f"\nNo Sources Found"
    
    await cl.Message(content=answer).send()


