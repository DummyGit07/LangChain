import streamlit as st
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
import os

# Configure page for dark mode
st.set_page_config(
    page_title="Speech Writer AI",
    page_icon="🎤",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for dark mode and ChatGPT-like styling
st.markdown("""
    <style>
    /* Dark theme base */
    .stApp {
        background-color: #0e1117;
        color: #fafafa;
    }
    .stMarkdown {
        color: #fafafa;
    }
    
    /* Chat container */
    .chat-container {
        max-height: 70vh;
        overflow-y: auto;
        padding: 1rem;
        background-color: #161b22;
        border-radius: 10px;
        border: 1px solid #30363d;
        margin-bottom: 1rem;
    }
    
    /* Chat messages */
    .user-message {
        background: linear-gradient(135deg, #1f6feb 0%, #007acc 100%);
        margin: 0.5rem 0;
        padding: 1rem;
        border-radius: 18px 18px 4px 18px;
        max-width: 80%;
        color: white;
        margin-left: auto;
    }
    
    .assistant-message {
        background: linear-gradient(135deg, #21262d 0%, #161b22 100%);
        margin: 0.5rem 0;
        padding: 1rem;
        border-radius: 18px 18px 18px 4px;
        max-width: 80%;
        color: #c9d1d9;
        border: 1px solid #30363d;
    }
    
    /* Input styling */
    .stTextInput > div > div > input {
        background-color: #21262d;
        border: 1px solid #30363d;
        color: #fafafa;
        border-radius: 25px;
        padding: 0.75rem 1rem;
    }
    
    /* Chat input focus */
    .stTextInput > div > div:focus {
        border-color: #1f6feb !important;
        box-shadow: 0 0 0 2px rgba(31, 111, 235, 0.2) !important;
    }
    
    /* Scrollbar styling */
    .chat-container::-webkit-scrollbar {
        width: 8px;
    }
    .chat-container::-webkit-scrollbar-track {
        background: #161b22;
    }
    .chat-container::-webkit-scrollbar-thumb {
        background: #30363d;
        border-radius: 4px;
    }
    
    /* Header styling */
    .main-header {
        font-size: 2.5rem;
        background: linear-gradient(135deg, #1f6feb 0%, #007acc 50%, #1e40af 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1rem;
        text-align: center;
    }
    
    /* Clear button */
    .stButton > button {
        background: linear-gradient(135deg, #6e40c9 0%, #1e40af 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.5rem 1.5rem;
        font-weight: 600;
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #5b32a3 0%, #1e3a8a 100%);
        box-shadow: 0 4px 12px rgba(110, 64, 201, 0.4);
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "topic_processed" not in st.session_state:
    st.session_state.topic_processed = False
if "current_topic" not in st.session_state:
    st.session_state.current_topic = ""

# Initialize LLMs and chains
@st.cache_resource
def load_models():
    llm1 = ChatOllama(model="mistral")
    llm2 = ChatOllama(model="llama3.2")
    
    title_prompt = PromptTemplate(
        input_variables=["topic"],
        template="""You are an experienced speech writer.
        You need to craft an impactful title for a speech 
        on the following topic: {topic}
        Answer exactly with one title only, no extra text."""
    )
    
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", """You are a powerful speech writer. Write 350-word speeches based on the provided title and conversation context.
        Maintain a professional, engaging tone suitable for public speaking."""),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])
    
    history = StreamlitChatMessageHistory(key="speech_writer_history")
    chain1 = title_prompt | llm1
    chain2 = prompt_template | llm2
    
    chain_with_history = RunnableWithMessageHistory(
        chain2,
        lambda session_id: history,
        input_messages_key="input",
        history_messages_key="chat_history"
    )
    
    return chain1, chain_with_history, title_prompt

chain1, chain_with_history, title_prompt = load_models()

# Header
st.markdown('<h1 class="main-header">🎤 Speech Writer AI</h1>', unsafe_allow_html=True)
st.markdown("Write impactful speeches with AI-powered titles and content generation:  \nFor New Topics Clear Chat History.")

# Chat container
chat_container = st.container()
with chat_container:
    # st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    
    # Display chat messages
    for message in st.session_state.messages:
        if message["role"] == "user":
            st.markdown(f'<div class="user-message">{message["content"]}</div>', unsafe_allow_html=True)
        elif message["role"] == "assistant":
            st.markdown(f'<div class="assistant-message">{message["content"]}</div>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# Sidebar for controls
with st.sidebar:
    st.header("⚙️ Controls")
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.session_state.topic_processed = False
        st.session_state.current_topic = ""
        st.rerun()
    
    st.markdown("---")
    st.info("💡 **How it works:**\n1. Enter a speech topic\n2. AI generates an optimized title\n3. AI writes a 350-word speech\n4. Continue conversation naturally\n5. For New Topic Clear History")

# Main chat input
if prompt := st.chat_input("Continue the conversation...", key="main_input"):
    # Add user message to chat
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # First message logic: Generate title
    if not st.session_state.topic_processed:
        try:
            with st.chat_message("user"):
                st.markdown(prompt)
            
            with st.chat_message("assistant"):
                # Place Holder till the Response come
                placeholder = st.empty()
                placeholder.markdown("⏳ Generating response, please wait...")

                title_response = chain1.invoke({"topic": prompt})
                refined_title = title_response.content.strip()

                st.session_state.current_topic = refined_title
                st.session_state.topic_processed = True
                
                placeholder.markdown(f"**📝 Generated Title:** {refined_title}  \n**✍️ Now writing your speech...**")
                
                # Generate speech using refined title
                speech_response = chain_with_history.invoke(
                    {"input": refined_title},
                    {"configurable": {"session_id": "speech_writer"}}
                )
                
                speech_content = f"📝 Generated Title: {refined_title}  \n{speech_response.content}"
                st.session_state.messages.append({"role": "assistant", "content": speech_content})
                # st.markdown(speech_content)
                placeholder.markdown(speech_content)
                
        except Exception as e:
            st.error(f"Error generating speech: {str(e)}")
            st.session_state.messages.append({"role": "assistant", "content": "Sorry, I encountered an error. Please try again."})
    else:
        # Continue conversation
        try:
            with st.chat_message("user"):
                st.markdown(prompt)
            
            with st.chat_message("assistant"):
                # place holder till the response come
                placeholder = st.empty()
                placeholder.markdown("⏳ Generating response, please wait...")
                
                response = chain_with_history.invoke(
                    {"input": prompt},
                    {"configurable": {"session_id": "speech_writer"}}
                )
                #change the place holder
                
                
                content = response.content
                st.session_state.messages.append({"role": "assistant", "content": content})
                # st.markdown(content)
                placeholder.markdown(content)
                
        except Exception as e:
            st.error(f"Error: {str(e)}")
            st.session_state.messages.append({"role": "assistant", "content": "Sorry, there was an error processing your request."})

# Auto-scroll to bottom
st.markdown("""
    <script>
    parent = window.parent.document;
    chat_container = parent.querySelector('.chat-container');
    if (chat_container) {
        chat_container.scrollTop = chat_container.scrollHeight;
    }
    </script>
""", unsafe_allow_html=True)
