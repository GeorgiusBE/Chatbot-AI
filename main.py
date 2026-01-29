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



# define prompt template
prompt = ChatPromptTemplate(
    [
        ("system", "You are a helpful assistant. Please response to the user queries."),
        ("human", "{input}")
    ]
)

#

llm = ChatOpenAI(model="gpt-5-mini")
chain = prompt | llm | StrOutputParser()
# response = chain.invoke()

#Title of the app
st.title("Q&A Chatbot With OpenAI")




