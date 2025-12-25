from langchain.tools import tool
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain.agents import create_agent
import os
from dotenv import load_dotenv
import streamlit as st

st.set_page_config(page_title="Healthcare AI", layout="wide")
st.title("Healthcare RAG Chatbot")

load_dotenv()

os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

# defining groq model
model = ChatGroq(
    model_name="llama-3.3-70b-versatile", 
    temperature=0.7,
)

# defining embedding model and initializing vector store
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2", model_kwargs={'device': 'cpu'})

vector_db = Chroma(
    persist_directory="./chroma_db", 
    embedding_function=embeddings
)

# define tool to retrieve context from local DB
@tool(response_format="content_and_artifact")
def retrieve_context(query: str):
    """
    Retrieves relevant medical and healthcare context from the local database.
    Input should be a search query string.
    """
    retrieved_docs = vector_db.similarity_search(query, k=2)
    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\nContent: {doc.page_content}")
        for doc in retrieved_docs
    )
    return serialized, retrieved_docs

# define tools for agent
tools = [retrieve_context]

# sidebar for user role
with st.sidebar:
    st.header("User Settings")
    user_role = st.selectbox(
        "Select your role:",
        ["Patient", "Doctor", "Medical Student"],
        index=0
    )
    if st.button("Clear Conversation"):
        st.session_state.messages = []
        st.rerun()
    
    st.divider()
    st.info(f"Current Persona: **{user_role}**")

# define system prompt with role-based instructions
prompt = (
    f"You are a versatile healthcare assistant. Current User Role: {user_role}.\n\n"
    "### INSTRUCTIONS:\n"
    "1. Always search the local database using `retrieve_context` first.\n"
    "2. If the local database has no information, use the web search tool to find reputable medical sources.\n"
    "3. Respond according to the User Role:\n"
    "- **Doctor**: Use clinical jargon, focus on pathophysiology and data.\n"
    "- **Patient**: Use simple, empathetic 'layman' terms. No jargon.\n"
    "- **Medical Student**: Educational, explain the 'why' and biochemical links.\n\n"
    "Strictly follow medical ethics and keep responses grounded in the retrieved facts."
)

# create the agent
agent = create_agent(model, tools, system_prompt=prompt)

# initialize session state for messages
if "messages" not in st.session_state:
    st.session_state.messages = []

# display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# chat input
if query := st.chat_input("Say something"):
    # display user message
    with st.chat_message("user"):
        st.markdown(query)
    
    # append user message to session history
    st.session_state.messages.append({"role": "user", "content": query})

    # process with agent and guardrail
    with st.chat_message("assistant"):
        with st.spinner("Analyzing and Verifying..."):
            # generate initial response
            final_state = agent.invoke({"messages": st.session_state.messages})
            
            # extract last ai message and  context 
            last_msg = final_state["messages"][-1]
            # identify context used ie the source document
            context_docs = [m.content for m in final_state["messages"] if m.type == "tool"]
            if context_docs:
                context_text = "\n".join(context_docs) 
            else:
                context_text = "No context retrieved."
                print("No context was retrieved for verification.")

            if isinstance(last_msg.content, list):
                raw_response = last_msg.content[0].get("text", "")
            else:
                raw_response = last_msg.content

            # guardrail verification step
            # we ask the model to verify its own answer against the context
            guard_prompt = f"""
            You are a Medical Fact-Checker. 
            CONTEXT: {context_text}
            DRAFT ANSWER: {raw_response}

            Your goal: Rewrite the DRAFT ANSWER so it ONLY contains information supported by the CONTEXT. 
            - If the draft mentions drugs or treatments not in the context, remove them.
            - Ensure the tone is appropriate for a {user_role}.
            - If the context is missing info, say "The provided records do not contain information on [topic]."
            - Return only the final, verified text.
            """
            # using the same model for verification
            verification = model.invoke(guard_prompt).content
            
            if "FAILED" in verification:
                full_response = verification.replace("FAILED", "⚠️ *Self-Corrected for Accuracy:*")
                print("Response contained inaccuracies and was corrected.")
            else:
                full_response = raw_response
                print("Response verified as accurate.")

        # display final response
        st.markdown(full_response)

    # append assistant message to session history
    st.session_state.messages.append({"role": "assistant", "content": full_response})
    