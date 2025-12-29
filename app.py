from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.tools import tool
from langchain_groq import ChatGroq
from langchain.agents import create_agent
import os
from dotenv import load_dotenv
import streamlit as st
from langchain.agents.middleware import SummarizationMiddleware
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.runnables import RunnableConfig
import json
from typing import Any, List
from langchain.agents import create_agent
from langchain.agents.middleware import (
    AgentMiddleware,
    AgentState,
    hook_config,
    PIIMiddleware,
)
from langchain.messages import AIMessage
from langchain.chat_models import init_chat_model
from langgraph.runtime import Runtime

# file paths for persistence
ROLES_FILE = "roles.json"
USER_CONFIG_FILE = "user_config.json"


def load_roles():
    # load roles from disk
    if not os.path.exists(ROLES_FILE):
        return {}
    
    if os.path.getsize(ROLES_FILE) == 0:
        return {}
    
    with open(ROLES_FILE, "r") as f:
        return json.load(f)


def save_roles(roles):
    # save roles to disk
    with open(ROLES_FILE, "w") as f:
        json.dump(roles, f, indent=2)


def load_user_role():
    # load last selected role
    if not os.path.exists(USER_CONFIG_FILE):
        return None
    with open(USER_CONFIG_FILE, "r") as f:
        return json.load(f).get("selected_role")


def save_user_role(role_name):
    # persist selected role
    with open(USER_CONFIG_FILE, "w") as f:
        json.dump({"selected_role": role_name}, f)




# streamlit page setup
st.set_page_config(page_title="Healthcare AI", layout="wide")
st.title("Healthcare RAG Assistant")


# load environment variables
load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")


# sidebar ui
roles = load_roles()
saved_role = load_user_role()

# initialize ui state flags
if "show_history" not in st.session_state:
    st.session_state.show_history = True

with st.sidebar:
    st.header("user role settings")

    role_names = list(roles.keys())

    selected_role = st.selectbox(
        "select role",
        role_names,
        index=role_names.index(saved_role) if saved_role in role_names else 0
    )

    # persist role selection
    save_user_role(selected_role)

    st.markdown("### role prompt")

    role_prompt = st.text_area(
        "edit role prompt",
        value=roles[selected_role]["prompt"],
        height=200
    )

    if st.button("save role prompt"):
        roles[selected_role]["prompt"] = role_prompt
        save_roles(roles)
        st.success("role prompt saved")

    if st.button("Clear conversation"):
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.markdown("### add new role")

    new_role_name = st.text_input("role name")
    new_role_prompt = st.text_area("role prompt", height=150)

    if st.button("add role"):
        if new_role_name.strip():
            roles[new_role_name.lower()] = {
                "description": "Custom role",
                "prompt": new_role_prompt
            }
            save_roles(roles)
            st.success(f"role '{new_role_name}' added")
            st.rerun()

    st.header("User Chat History")

    toggle_label = "hide chat history" if st.session_state.show_history else "show chat history"

    if st.button(toggle_label):
        st.session_state.show_history = not st.session_state.show_history




# initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# 1.define hf embeddings
model_name = "sentence-transformers/all-MiniLM-L6-v2"
hf_embeddings = HuggingFaceEmbeddings(model_name=model_name, model_kwargs={'device': 'cpu'})

# 2.load the existing Vector Store
vector_db = Chroma(persist_directory="./chroma_db", embedding_function=hf_embeddings)

# function to create vector store from pdf using hugging face embeddings
def ingest_pdf(pdf_path, vector_db=vector_db):
    # 3.check if the file has already been ingested
    # we look for the source in the metadata
    existing_docs = vector_db.get(where={"source": pdf_path})
    
    if len(existing_docs['ids']) > 0:
        print(f"Skipping: {pdf_path} is already in the vector store.")
        return vector_db

    # 4. if file not found, proceed with loading and splitting
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_documents(documents)

    # 5. add only the new chunks
    vector_db.add_documents(chunks)
    print(f"Successfully added {pdf_path} to the store.")
    return vector_db

# st file uploader
uploaded_files = st.file_uploader(
    "Upload PDF files", accept_multiple_files=True, type="pdf"
)
# upload and ingest each file
for file in uploaded_files:
    with open(file.name, "wb") as f:
        f.write(file.getbuffer())
    vector_db = ingest_pdf(file.name)
    st.success(f"Uploaded and ingested: {file.name}")


# initialize agent only once ****

# streamlit reruns the script on every interaction
# so we must store the agent in session_state
if "active_role" not in st.session_state:
    st.session_state.active_role = selected_role

if ("agent" not in st.session_state or st.session_state.active_role != selected_role):

    st.session_state.active_role = selected_role

    # in-memory saver stores conversation state
    checkpointer = InMemorySaver()

    # main llm
    model = ChatGroq(
        model_name="openai/gpt-oss-120b",
        temperature=0.7,
    )
    
    # define tool to retrieve context from local DB
    @tool(response_format="content_and_artifact")
    def retrieve_context(query: str):
        """
        Retrieves relevant medical and healthcare context from the local database.
        Input should be a search query string.
        """
        retrieved_docs = vector_db.similarity_search(query, k=4)
        serialized = "\n\n".join(
            (f"Source: {doc.metadata}\nContent: {doc.page_content}")
            for doc in retrieved_docs
        )
        print(f"Retrieved {len(retrieved_docs)} documents for query: {query}")
        print(serialized)
        return serialized, retrieved_docs

    class RoleToneGuardrail(AgentMiddleware):
        # ensures response tone matches active role

        def __init__(self, active_role: str, active_role_prompt: str):
            super().__init__()
            self.active_role = active_role
            self.active_role_prompt = active_role_prompt
            self.judge = model = ChatGroq(model_name="openai/gpt-oss-120b", temperature=0.7,)

        @hook_config(can_jump_to=["end"])
        def after_agent(
            self,
            state: AgentState,
            runtime: Runtime,
        ) -> dict[str, Any] | None:

            ai_msg = state["messages"][-1]

            if not isinstance(ai_msg, AIMessage):
                return None

            prompt = f"""
            CHECK IF THE AI RESPONSE MATCHES THE EXPECTED TONE FOR THE ROLE.

            ROLE:
            {self.active_role}

            ROLE RULES WHICH ARE MANDATORY TO FOLLOW:
            {self.active_role_prompt}
            
            AI RESPONSE:
            {ai_msg.content}

            RESPOND WITH ONLY ONE WORD:
            MATCH
            OR
            MISMATCH
            """
            verdict = self.judge.invoke(
                [{"role": "user", "content": prompt}]
            ).content.strip().upper()

            if verdict == "MISMATCH":
                ai_msg.content = (
                    f"Let me rephrase this in a way that better suits a {self.active_role}:\n\n"
                    + ai_msg.content
                )

            return None

    # agent setup with guardrails

    # get active role prompt
    active_role_prompt = roles[selected_role]["prompt"]

    # build final system prompt
    system_prompt = (
        f"Current user role is as follows: {st.session_state.active_role}\n\n"
        f"{active_role_prompt}\n\n"
        "You have access to a tool called retrieve_context.\n"
        "For any healthcare or document-based question, you MUST call retrieve_context first.\n"
        "Use ONLY the retrieved context to answer. If context is insufficient, say so clearly."
    )


    # create agent with summarization middleware
    st.session_state.agent = create_agent(
        model=model,
        tools=[retrieve_context],
        system_prompt=system_prompt,
        middleware=[
            SummarizationMiddleware(
                model=ChatGroq(model_name="llama-3.1-8b-instant"),
                trigger=("tokens", 4000),
                keep=("messages", 20)
            ),
            RoleToneGuardrail(active_role=st.session_state.active_role, active_role_prompt=active_role_prompt),
        ],
        checkpointer=checkpointer,
    )


# langgraph thread config

# same thread_id = same memory
config: RunnableConfig = {
    "configurable": {
        "thread_id": "healthcare-thread-1"
    }
}

# display previous messages only if enabled
if st.session_state.show_history:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

# chat input
if query := st.chat_input("say something"):

    # store user message
    st.session_state.messages.append({
        "role": "user",
        "content": query
    })

    # show user message
    with st.chat_message("user"):
        st.markdown(query)

    # run agent
    with st.chat_message("assistant"):
        with st.spinner("thinking..."):

            final_state = st.session_state.agent.invoke(
                {"messages": st.session_state.messages},
                config
            )

            # extract assistant message
            last_message = final_state["messages"][-1]
            response_text = (
                last_message.content[0]["text"]
                if isinstance(last_message.content, list)
                else last_message.content
            )

        st.markdown(response_text)

    # store assistant message
    st.session_state.messages.append({
        "role": "assistant",
        "content": response_text
    })
