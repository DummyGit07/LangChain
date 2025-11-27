from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import streamlit as st

title_prompt = PromptTemplate(
    input_variables=['topic'],
    template="""You are an experienced speech writer.
                You need to craft an impactful title for a speech
                on the following topic: {topic}
                Answer exactly with one title.
            """
)

speech_prompt = PromptTemplate(
    input_variables=['title', 'emotion'],
    template = """You need to write a powerful {emotion} speech of 350 words
                for the following title: {title}"""
)

llm1 = ChatOllama(model='llama3.2')
llm2 = ChatOllama(model='mistral')

# chainning
first_chain = title_prompt | llm1 | StrOutputParser() | (lambda title: (st.write(title), title)[1])
second_chain = speech_prompt | llm2

final_chain = first_chain | (lambda title: {"title" : title, "emotion" : emotion}) | second_chain

st.title("Speech Generator")
topic = st.text_input("Enter Topic: ")
emotion = st.text_input("Enter Emotion: ")

if topic and emotion:
    response = final_chain.invoke({"topic":topic})
    st.write(response.content)