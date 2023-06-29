import os 
from api_key import apikey 
import streamlit as st 
import os
import pickle
import faiss
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chains.question_answering import load_qa_chain
from langchain import OpenAI
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationSummaryMemory



st.title('Chatbot')
prompt = st.text_input('Ask me anything!') 



os.environ["OPENAI_API_KEY"] = "xxxxxxxxxx"
urls = [
    'https://www.mosaicml.com/blog/mpt-7b',
    'https://stability.ai/blog/stability-ai-launches-the-first-of-its-stablelm-suite-of-language-models',
    'https://lmsys.org/blog/2023-03-30-vicuna/'
]
from langchain.document_loaders import UnstructuredURLLoader
loaders = UnstructuredURLLoader(urls=urls)
data = loaders.load()
from langchain.text_splitter import CharacterTextSplitter

text_splitter = CharacterTextSplitter(separator='\n',
                                      chunk_size=1000,
                                      chunk_overlap=200)


docs = text_splitter.split_documents(data)



embeddings = OpenAIEmbeddings()
#vectorStore_openAI = FAISS.from_documents(docs, embeddings)

#with open("faiss_store_openai.pkl", "wb") as f:
  #pickle.dump(vectorStore_openAI, f)


with open("faiss_store_openai.pkl", "rb") as f:
    VectorStore = pickle.load(f)

llm = OpenAI(temperature=0)




conversation_chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=VectorStore.as_retriever()) 




if prompt:
    response = conversation_chain({"question": prompt}, return_only_outputs=True)
    st.write(response['answer'])
    st.write(response['sources'])
