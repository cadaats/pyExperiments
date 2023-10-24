import os
from langchain.llms import OpenAI
from langchain.chat_models.openai import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import Chroma
from langchain.agents.agent_toolkits import create_vectorstore_agent, VectorStoreToolkit, VectorStoreInfo
from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv())
# use gpt-4 and connect to open ai
llm = ChatOpenAI(model="gpt-4", temperature=0.1, verbose=True)
pdf_loader = PyPDFLoader("..\Data\SSRN-id3247865.pdf")
pages = pdf_loader.load_and_split()

# load documents in chromaDB
embeddings = OpenAIEmbeddings()
store = Chroma.from_documents(pages, embeddings, collection_name="151_Trading_Strategies")
vector_store_info = VectorStoreInfo(name="151_Trading_Strategies",
                                    description="151 Trading Strategies By Zura Kakushadze",
                                    vectorstore=store)
tool_kit = VectorStoreToolkit(vectorstore_info=vector_store_info)
agent_executor = create_vectorstore_agent(llm=llm,
                                          toolkit=tool_kit,
                                          verbose=True)

prompt = input("Enter a prompt: ")

if prompt:
    response = agent_executor.run(prompt)
    print(response)
    

