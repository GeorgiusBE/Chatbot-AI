import streamlit as st
import openai
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
import os

# load variables in .env locally (if present)
from dotenv import load_dotenv
load_dotenv()

# ensure that OpenAI and LangChain API key is provided, otherwise the app stops
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY", None)
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY") or st.secrets.get("LANGCHAIN_API_KEY", None)

if not OPENAI_API_KEY:
    st.error("Missing OPENAI_API_KEY")
    st.stop()

if not LANGCHAIN_API_KEY:
    st.error("Missing LANGCHAIN_API_KEY")
    st.stop()

# Title of the app
st.title("Q&A Chatbot With OpenAI")

# Sidebar to set model settings
with st.sidebar:
    st.header("Model Settings")

    # llm model selection
    model = st.selectbox(
        "Model",
        options=["gpt-4o-mini", "gpt-4o", "gpt-4.1-mini","gpt-4.1",],
        index=0,         # default selection
        help="Choose the OpenAI chat model to use"
    )

    # model temperature selection
    temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=2.0,
        value=0.7,
        step=0.1,
        help="Higher = more creative, lower = more deterministic."
        )

    #

# create function to generate query response
def generate_response(question:str, system_prompt:str, model:str, temperature:float):
    llm = ChatOpenAI(model=model, temperature=temperature)
    prompt = ChatPromptTemplate(
        [
            ("system", system_prompt),
            ("human", "{input}")
        ]
    )
    chain = prompt | llm | StrOutputParser()
    response = chain.invoke({"input": question})
    return response