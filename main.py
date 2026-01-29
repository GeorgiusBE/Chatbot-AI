import streamlit as st
import openai
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
import os

# load variables in .env
from dotenv import load_dotenv
load_dotenv()

# define prompt template
prompt = ChatPromptTemplate(
    [
        ("system", "You are a helpful assistant. Please response to the user queries."),
        ("human", "{input}")
    ]
)

#

llm = ChatOpenAI("gpt-5-mini")
chain = prompt | llm | StrOutputParser()
response = chain.invoke()





