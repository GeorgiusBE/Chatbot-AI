import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.messages import trim_messages

import os, tempfile, shutil
from pathlib import Path
from operator import itemgetter

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



# --- Back-end ---
## Create object to store chat message history
    ## We use this "if" statement, because in Streamlit, when we click on a widget, etc. it will rerun
    ## ... the whole Python file. And so, this "if" statement prevents the storage from getting overwritten
if 'store' not in st.session_state:
    st.session_state.store={}

# define a function to retrieve chat message history
def get_session_history(session_id:str) -> BaseChatMessageHistory:
    if session_id not in st.session_state.store:
        st.session_state.store[session_id] = ChatMessageHistory()
    return st.session_state.store[session_id]

# set trimmer to manage the conversation history
def trim_history(inputs: dict) -> dict:
    # create llm chain
    llm = ChatOpenAI(model=model)
    inputs["history"] = trim_messages(
        inputs.get("history", []),
        max_tokens=10000,
        strategy="last",
        token_counter=llm,
        include_system=True,
        allow_partial=False,
        start_on="human"
    )
    return inputs

## create function to generate query reponse without RAG
def generate_response_plain(question: str, system_prompt: str, model: str, temperature: float):
    llm = ChatOpenAI(model=model, temperature=temperature)
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{input}")
        ]
    )
    chain = RunnableLambda(lambda x: trim_history(x)) | prompt | llm | StrOutputParser()

    # wrap runnable with message history
    with_message_history = RunnableWithMessageHistory(
        chain,
        get_session_history=get_session_history,
        input_messages_key="input",
        history_messages_key="history"
    )

    config={"configurable":{"session_id":"chat1"}}

    return with_message_history.stream({"input": question}, config=config)

## Save user-uploaded files to disk and return saved paths
def save_uploads_to_dir(uploaded_files, target_dir:Path) -> list[Path]:
    '''Save Streamlit UplaodedFile objects to disk and return saved paths'''
    target_dir.mkdir(parents=True, exist_ok=True) # create folder to store the uploaded files
    saved = []
    for uf in uploaded_files:
        out = target_dir / uf.name # define the file path
        out.write_bytes(uf.getbuffer())
            # "uf.getbuffer()" gets the raw bytes of the uploaded file.
            # "write_bytes(...)" writes those bytes to a real file on disk at "out".
        saved.append(out) # store file path of saved files
    return saved

## create history-aware RAG retriever
def create_rag_retriever(pdf_dir: Path):
    # data ingestion: On all pdf files in a folder
    loader = DirectoryLoader(
        path=str(pdf_dir), # open this folder path
        glob="**/*.pdf", # "*.pdf" opens any pdf file ; "**/" search this folder and subfolders (e.g. ./research_papers/paper1.pdf , ./research_papers/subfolder/paper2.pdf)
        loader_cls=PyPDFLoader # choosing the type of loader
        )
    docs = loader.load()
    
    # data spliting
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(docs)

    # embed the chunks and store in Vector Database
    embedding_model = OpenAIEmbeddings(model="text-embedding-3-large")
    vectorstoredb = Chroma.from_documents(
        chunks,
        embedding_model,
        persist_directory="./chroma_db", # save the vector store locally
        collection_name="pdf_embedding" # specify a new name for this collection, bcs a persisted Chroma DB can hold more than one collection
        )

    # create the retriever (not history-aware)
    basic_retriever = vectorstoredb.as_retriever() # store retriever in state memory

    # create prompt to rewrite the user input + history
    rewrite_prompt = ChatPromptTemplate.from_messages(
        [
            ("system",
            "Given a chat history and the latest user question "
            "which might reference context in the chat history, "
            "formulate a standalone question which can be understood "
            "without the chat history. Do NOT answer the question, just "
            "reformulate it if needed and otherwise return it as is."
            ),
            MessagesPlaceholder("history"),
            ("human", "{input}")
        ]
    )

    # Choose the llm model to rewrite the user input + history
    llm_rewrite = ChatOpenAI(model="gpt-5-mini")
    
    # create chain that rewrites the current prompt based on chat history
    st.session_state.retriever = (
        {
            "input": itemgetter("input"),
            "history": itemgetter("history")
        }
        | rewrite_prompt
        | llm_rewrite
        | StrOutputParser()         # -> standalone query (string)
        | basic_retriever           # -> list[Document]
    )

## wrapper function on create_rag_retriever() -> avoid rerunning the RAG chain
def get_retriever(pdf_dir: Path):
    if "retriever" not in st.session_state:
        # create retriever
        create_rag_retriever(pdf_dir)

## create function to generate query response with RAG
def format_docs(docs):
    '''Join the page content of multiple Document objects'''
    return "\n\n".join(d.page_content for d in docs)

## function to generate RAG-based llm response
def generate_response_rag(retriever, question:str, system_prompt:str, model:str, temperature:float):
    # create llm chain
    llm = ChatOpenAI(model=model, temperature=temperature)
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="history"),
            ("human", '''
             <context>
             {context}
             </context>

             <question>
             {input}
             </question>'''
             )
        ]
    )
    chain = (
        RunnableLambda(lambda x: trim_history(x))
        | {
            "context": retriever | RunnableLambda(lambda x: format_docs(x)),
            "input": itemgetter("input"),
            "history": itemgetter("history")
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    with_message_history = RunnableWithMessageHistory(
        chain,
        get_session_history=get_session_history,
        input_messages_key="input",
        history_messages_key="history"
    )

    config = {"configurable": {"session_id":"chat1"}}
    response = with_message_history.stream({"input":question}, config=config)
    return response

## create a wrapper to choose whether to repond with or without RAG
def generate_response(question: str, system_prompt: str, model: str, temperature: float, use_rag: bool):
    if use_rag and "retriever" in st.session_state:
        return generate_response_rag(st.session_state.retriever, question, system_prompt, model, temperature)

    # either user chose no RAG, or retriever not ready
    return generate_response_plain(question, system_prompt, model, temperature)








# --- Sidebar to set model settings ---
with st.sidebar:
    ## pdf file uploader UI
    st.header("File Uploader")

    ## file uploader: PDFs
    if "uploader_key" not in st.session_state:
        st.session_state.uploader_key = 0

    uploaded_pdfs = st.file_uploader(
        "Upload PDF files",
        type=['pdf'],
        accept_multiple_files=True,
        key=f"key_pdf_uploader{st.session_state.uploader_key}" # this will be needed to reset the file_uploader
        )

    col1, col2 = st.columns(2)
    with col1:
        build = st.button("Build Retriever", type="primary", use_container_width=True)
    with col2:
        clear = st.button("Clear Retriever", type="secondary", use_container_width=True)

    ## Only build RAG when user clicks the button
    if build:
        if not uploaded_pdfs:
            st.error("Upload PDFs first before creating a retriever!")
        else:
            with st.spinner("Indexing PDFs (loasing -> Splitting -> embedding)..."):
                try:
                    # create a temporary directory to store temporary files (in disk/permanent)
                    tmp_root = Path(tempfile.mkdtemp()) # create temporary folder with unique name, e.g. C:\Users\You\AppData\Local\Temp\tmpabc123
                    pdf_dir = tmp_root / "pdfs" # e.g. C:\Users\You\AppData\Local\Temp\tmpabc123\pdfs

                    # save the uploaded file in the temporary directory
                    save_uploads_to_dir(uploaded_pdfs, pdf_dir)

                    # Build the retriever
                    get_retriever(pdf_dir)
                    st.success("The Retriever is ready!")
                
                finally:
                    # remove the folder/files after uploading, as those files have been stored in the Vector Store
                    shutil.rmtree(tmp_root, ignore_errors=True) # "rmtree()" stands for "remove tree". This will delete the entire diretory tree (i.e. the folder and everything inside it)

    ## Clear the retriever
    if clear:
        st.session_state.pop("retriever", None)
        st.session_state.uploader_key += 1 # change the key to reset the file_uploader
        st.success("Cleared retriever!")
        st.rerun()

    ## Toggle to either use/not use the created retriever 
    use_rag = st.checkbox("Use uploaded files (RAG)", value=True)

    ## LLM model settings
    st.divider()
    st.header("Model Settings")

    # llm model selection
    model = st.selectbox(
        "Model",
        options=["gpt-5-mini","gpt-4o-mini", "gpt-4o", "gpt-4.1-mini","gpt-4.1"],
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

    # toggle for system message
    st.divider()
    show_system_prompt = st.checkbox("Show system prompt", value=False)

    system_prompt = st.text_area(
        "System prompt",
        value="You are a helpful assistant. Please respond to the user queries.",    # default value
        height=120,
        disabled= not show_system_prompt
        )

# --- App UI ---
## Title of the app
st.title("Q&A Chatbot With OpenAI")

## store message history list in session state
if "messages" not in st.session_state:
    st.session_state.messages = [] # each item: {"role": "user"/"ai", "content": "..."}

## display full chat history messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

## user to input question
question = st.chat_input("What can I help you today?")

## run the model
if question:
    # store new user input message
    st.session_state.messages.append({"role":"user", "content":question})

    # render/show/print user input message
    with st.chat_message("user"):
        st.markdown(question)
    
    # render/print AI response with a placeholder to update while streaming
    with st.chat_message("ai"):
        placeholder = st.empty() # intially shows nothing, but it's replaced with contents later
        full = ""

        # generate response
        stream = generate_response(
            question=question,
            model=model,
            temperature=temperature,
            system_prompt=system_prompt,
            use_rag=use_rag
        )

        for chunk in stream:
            full += chunk
            placeholder.markdown(full) #re-render the placeholder with the latest full text

    if use_rag and "retriever" not in st.session_state:
        st.warning("RAG is enabled, but no retriever is built yet. Answering without uploaded files...")

    # store new AI response message
    st.session_state.messages.append({"role":"ai", "content":full})
