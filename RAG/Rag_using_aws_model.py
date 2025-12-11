# before running config the aws and download awscli, boto3, langchian-aws
from langchain_chroma import Chroma
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain, create_history_aware_retriever
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_aws import ChatBedrockConverse, BedrockEmbeddings

import streamlit as st
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_core.runnables import RunnableWithMessageHistory

st.set_page_config(page_title="RAG Assistant", page_icon="🤖")

# LLM: Nova Pro (us.amazon.nova-pro-v1:0) - Chat/Output Generation
llm = ChatBedrockConverse(
    model_id="us.amazon.nova-pro-v1:0",      # ← Chat model (Nova Pro)
    region_name="us-east-2",
    temperature=0,
    max_tokens=512,
)

# Embeddings: Amazon Titan Text Embeddings v2 (amazon.titan-embed-text-v2:0)
embeddings = BedrockEmbeddings(
    model_id="amazon.titan-embed-text-v2:0",  # ← Amazon embedding model (not Cohere)
    region_name="us-east-1",
)


# PROMPTS
# 1) Prompt for history-aware retriever (NO {context} here)
contextualize_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant that reformulates user questions "
            "based on the chat history so they can be used for retrieval. "
            "Do NOT answer the question, only rewrite it as a standalone query."
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ]
)

# 2) Prompt for QA with retrieved context (HAS {context})
qa_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a helpful product assistant.
            Use the provided context to answer questions accurately.
            If the answer isn't clear from context, say "I don't have that information."
            Limit responses to 3 concise sentences.
            Context: {context}"""
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ]
)


# LOAD & PROCESS DOCUMENTS

document = TextLoader("RAG/product-data.txt").load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.split_documents(document)


# VECTOR STORE & RETRIEVER (with score threshold)

vector_store = Chroma.from_documents(chunks, embeddings)

retriever = vector_store.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={
        "score_threshold": 0.3,  # tune this (0.2–0.5 is a common range)
        "k": 4,                  # max docs if they pass the threshold
    },
)


# BUILD RAG CHAIN (history-aware)

# History-aware retriever: rewrites the question using chat history
history_aware_retriever = create_history_aware_retriever(
    llm,
    retriever,
    contextualize_prompt,   # no {context} here
)

# QA chain: uses retrieved context + question to answer
qa_chain = create_stuff_documents_chain(
    llm,
    qa_prompt,              # has {context}
)

# Full RAG chain: history-aware retriever → QA
rag_chain = create_retrieval_chain(
    history_aware_retriever,
    qa_chain,
)


# LANGCHAIN CHAT HISTORY WRAPPER

history_for_chain = StreamlitChatMessageHistory()

chain_with_history = RunnableWithMessageHistory(
    rag_chain,
    lambda session_id: history_for_chain,
    input_messages_key="input",       # matches {input}
    history_messages_key="chat_history",
    output_messages_key="answer",     # matches create_retrieval_chain output key
)



st.title("🤖 RAG Assistant")
st.caption("Powered by Amazon Bedrock (Nova Pro + Amazon Titan Embeddings)")

# Initialize UI chat history (for visual chat bubbles)
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Hi! Ask me anything about your products.",
        }
    ]

# Render full chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input at bottom (ChatGPT-style)
user_input = st.chat_input("Ask about your products...")

if user_input:
    # 1) Show user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # 2) Run RAG chain
    with st.chat_message("assistant"):
        with st.spinner("🤖 Nova Pro is thinking..."):
            try:
                response = chain_with_history.invoke(
                    {"input": user_input},
                    {"configurable": {"session_id": "abc123"}},
                )
                answer = response["answer"]
            except Exception as e:
                answer = f"Error: {str(e)}"

        st.markdown(answer)

    # 3) Store assistant answer in UI history
    st.session_state.messages.append(
        {"role": "assistant", "content": answer}
    )
