from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import PromptTemplate
import streamlit as st

llm = ChatOllama(model = "gemma:2b")

prompt_template = PromptTemplate(
    input_variables=['position', 'company', 'strengths', 'weaknesses'],
    template= """You are a career coach. Provide tailored interview tips for the
                position of {position} at {company}.
                Highlight your strengths in {strengths} and prepare for questions
                about your weaknesses such as {weaknesses}.
            """
)

st.title("InterView Tips Generator")
position = st.text_input("Enter Job Position:")
company = st.text_input("Enter Company:")
strengths = st.text_area("Enter your Strengths:", height=100)
weaknesses = st.text_area("Enter Your Weaknesses: ", height=100)

if position and company and strengths and weaknesses:
    response = llm.invoke(prompt_template.format(position=position, company=company, strengths=strengths, weaknesses=weaknesses))
    st.write(response.content)