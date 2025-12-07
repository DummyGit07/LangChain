from langchain_chroma import Chroma
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain, create_history_aware_retriever
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader

import streamlit as st
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_core.runnables import RunnableWithMessageHistory


llm = ChatOllama(model = 'llama3.2')
embeddings = OllamaEmbeddings(model = 'llama3.2')

prompt_template = ChatPromptTemplate.from_messages(
[
    ("system","""You are an assistant for answering questions.
    Use the provided context to respond.If the answer 
    isn't clear, acknowledge that you don't know. 
    Limit your response to three concise sentences.
    {context}
    """),
    MessagesPlaceholder(variable_name='chat_history'),
    ("human", "{input}")
]
)

document = TextLoader("RAG/product-data.txt").load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 200)
chunks = text_splitter.split_documents(document)

vector_store = Chroma.from_documents(chunks, embeddings)
retriver = vector_store.as_retriever()

history_aware_retriver = create_history_aware_retriever(llm, retriver, prompt_template)
qa_chain = create_stuff_documents_chain(llm, prompt_template)
rag_chain = create_retrieval_chain(history_aware_retriver, qa_chain)

history_for_chain = StreamlitChatMessageHistory()
chain_with_history = RunnableWithMessageHistory(
    rag_chain,
    lambda session_id:history_for_chain,
    input_messages_key = "input",
    history_messages_key = "chat_history"
)

question = st.text_input("Enter Question: ")
if question:
    response = chain_with_history.invoke({"input": question}, {'configurable':{'session_id':"abc123"}})
    st.write(response['answer'])