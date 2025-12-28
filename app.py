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
st.title("Healthcare RAG Chatbot")


# load environment variables
load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")



# sidebar ui
roles = load_roles()
saved_role = load_user_role()

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
     
    # # before-agent medical guardrail
    # class MedicalInputGuardrail(AgentMiddleware):
    #     # deterministic guardrail to block diagnosis and treatment requests

    #     def __init__(self):
    #         super().__init__()
    #         self.blocked_keywords = [
    #             "diagnose",
    #             "diagnosis",
    #             "treat",
    #             "treatment",
    #             "prescribe",
    #             "prescription",
    #             "dosage",
    #             "dose",
    #             "medicine for me",
    #             "should i take",
    #         ]

    #     @hook_config(can_jump_to=["end"])
    #     def before_agent(
    #         self,
    #         state: AgentState,
    #         runtime: Runtime,
    #     ) -> dict[str, Any] | None:

    #         # ensure there is at least one message
    #         if not state["messages"]:
    #             return None

    #         first_message = state["messages"][0]

    #         # only inspect human messages
    #         if first_message.type != "human":
    #             return None

    #         content = first_message.content.lower()

    #         # block unsafe medical intent
    #         for keyword in self.blocked_keywords:
    #             if keyword in content:
    #                 return {
    #                     "messages": [
    #                         {
    #                             "role": "assistant",
    #                             "content": (
    #                                 "i am not a medical professional and cannot provide diagnoses, "
    #                                 "treatments, or prescriptions. please consult a licensed healthcare provider."
    #                             ),
    #                         }
    #                     ],
    #                     "jump_to": "end",
    #                 }

    #         return None

    # # after-agent medical guardrail
    # class MedicalOutputGuardrail(AgentMiddleware):
    #     # model-based safety scan for medical advice leakage

    #     def __init__(self):
    #         super().__init__()
    #         self.safety_model =  ChatGroq(model_name="openai/gpt-oss-120b",temperature=0.7,)

    #     @hook_config(can_jump_to=["end"])
    #     def after_agent(
    #         self,
    #         state: AgentState,
    #         runtime: Runtime,
    #     ) -> dict[str, Any] | None:

    #         # ensure there is a final message
    #         if not state["messages"]:
    #             return None

    #         last_message = state["messages"][-1]

    #         if not isinstance(last_message, AIMessage):
    #             return None

    #         # llm-based medical safety check
    #         safety_prompt = f"""
    #         you are a healthcare compliance checker.
    #         respond only with SAFE or UNSAFE.

    #         unsafe if the response:
    #         - gives diagnosis
    #         - recommends treatment
    #         - suggests medication or dosage
    #         - replaces a medical professional

    #         response:
    #         {last_message.content}
    #         """

    #         result = self.safety_model.invoke(
    #             [{"role": "user", "content": safety_prompt}]
    #         )

    #         if "UNSAFE" in result.content:
    #             last_message.content = (
    #                 "i cannot provide medical advice. "
    #                 "this information is for educational purposes only. "
    #                 "please consult a qualified healthcare professional."
    #             )

    #         # enforce medical disclaimer
    #         if "educational purposes only" not in last_message.content.lower():
    #             last_message.content += (
    #                 "\n\nthis information is for educational purposes only "
    #                 "and is not a substitute for professional medical advice."
    #             )

    #         return None

    class RelevanceGuardrail(AgentMiddleware):
    # model-based guardrail to ensure answer matches user query intent
        def __init__(self):
            super().__init__()
            self.judge_model = init_chat_model("gpt-4o-mini")

        @hook_config(can_jump_to=["end"])
        def after_agent(
            self,
            state: AgentState,
            runtime: Runtime,
        ) -> dict[str, Any] | None:

            # ensure we have enough messages
            if len(state["messages"]) < 2:
                return None

            user_message = state["messages"][0]
            last_message = state["messages"][-1]

            # ensure correct message types
            if user_message.type != "human":
                return None

            if not isinstance(last_message, AIMessage):
                return None

            # relevance evaluation prompt
            relevance_prompt = f"""
    determine whether the assistant response is relevant to the user query.

    user query:
    {user_message.content}

    assistant response:
    {last_message.content}

    respond with only one word:
    RELEVANT or IRRELEVANT
    """

            judgment = self.judge_model.invoke(
                [{"role": "user", "content": relevance_prompt}]
            )

            # handle irrelevant responses
            if "IRRELEVANT" in judgment.content.upper():
                last_message.content = (
                    "i may have misunderstood your question. "
                    "could you please clarify what information you are looking for?"
                )

            return None
    # agent setup with guardrails

    # get active role prompt
    active_role_prompt = roles[selected_role]["prompt"]

    # build final system prompt
    system_prompt = (
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
            # MedicalInputGuardrail(),
            # layer 2: pii protection (healthcare-safe default)
            PIIMiddleware("email", strategy="redact", apply_to_input=True),
            PIIMiddleware("ip", strategy="block", apply_to_input=True),
            # MedicalOutputGuardrail(),
        ],
        checkpointer=checkpointer,
    )

    st.session_state.active_role = selected_role


# langgraph thread config

# same thread_id = same memory
config: RunnableConfig = {
    "configurable": {
        "thread_id": "healthcare-thread-1"
    }
}

# display previous messages
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
