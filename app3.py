import streamlit as st
from llama_index.llms.ollama import Ollama
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings
from llama_index.embeddings.fastembed import FastEmbedEmbedding
from llama_index.core.llms import ChatMessage
from llama_index.core.storage.chat_store import SimpleChatStore
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.chat_engine import CondensePlusContextChatEngine
import uuid  # For generating unique session IDs


# Define the Chatbot class
class Chatbot:
    def __init__(self, llm="llama3.1:latest", embedding_model="intfloat/multilingual-e5-large", vector_store=None):
        self.Settings = self.set_setting(llm, embedding_model)

        # Indexing
        self.index = self.load_data()

        # Memory
        self.memory = self.create_memory()

        # Chat Engine
        self.chat_engine = self.create_chat_engine(self.index)

    def set_setting(_arg, llm, embedding_model):
        Settings.llm = Ollama(model=llm, base_url="http://127.0.0.1:11434")
        Settings.embed_model = FastEmbedEmbedding(
            model_name=embedding_model, cache_dir="./fastembed_cache")
        Settings.system_prompt = """
                                You are a multi-lingual expert system who has knowledge, based on 
                                real-time data. You will always try to be helpful and try to help them 
                                answering their question. If you don't know the answer, say that you DON'T
                                KNOW.
                                """
        return Settings

    @st.cache_resource(show_spinner=False)
    def load_data(_arg, vector_store=None):
        with st.spinner(text="Loading and indexing – hang tight! This should take a few minutes."):
            # Read & load document from folder
            reader = SimpleDirectoryReader(input_dir="./docs", recursive=True)
            documents = reader.load_data()

        if vector_store is None:
            index = VectorStoreIndex.from_documents(documents)
        return index

    def set_chat_history(self, messages):
        self.chat_history = [ChatMessage(role=message["role"], content=message["content"]) for message in messages]
        self.chat_store.store = {"chat_history": self.chat_history}

    def create_memory(self):
        self.chat_store = SimpleChatStore()
        return ChatMemoryBuffer.from_defaults(chat_store=self.chat_store, chat_store_key="chat_history", token_limit=16000)

    def create_chat_engine(self, index):
        return CondensePlusContextChatEngine(
            verbose=True,
            memory=self.memory,
            retriever=index.as_retriever(),
            llm=Settings.llm
        )


# Initialize chat session if not available
if "chat_sessions" not in st.session_state:
    st.session_state.chat_sessions = {}

# Initialize selected session
if "selected_session" not in st.session_state:
    st.session_state.selected_session = None

# Function to create a new chat session with the default name
# Function to create a new session
def create_new_session():
    session_id = f"session_{len(st.session_state.chat_sessions) + 1}"
    st.session_state.chat_sessions[session_id] = {"name": "New Chat", "messages": []}
    st.session_state.selected_session = session_id

# Sidebar for managing chat sessions
with st.sidebar:
    st.header("Chat Sessions")

    # Check if there are no chat sessions and automatically create one
    if "chat_sessions" not in st.session_state:
        st.session_state.chat_sessions = {}

    if not st.session_state.chat_sessions:
        create_new_session()  # Automatically create a new session if none exist

    # Show list of sessions with session names being the last user message
    session_names = {session_id: session_data["name"] for session_id, session_data in st.session_state.chat_sessions.items()}
    selected_name = st.selectbox("Select a session", list(session_names.values()))

    # Get selected session ID based on name
    if selected_name:
        selected_session_id = list(st.session_state.chat_sessions.keys())[list(session_names.values()).index(selected_name)]
        st.session_state.selected_session = selected_session_id

    # Option to create a new session
    if st.button("Start New Session"):
        create_new_session()


# Main Program
st.title("Simple RAG Chatbot with Streamlit")
chatbot = Chatbot()

# Display chat history of selected session
if st.session_state.selected_session:
    session_data = st.session_state.chat_sessions[st.session_state.selected_session]
    session_messages = session_data["messages"]

    for message in session_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    chatbot.set_chat_history(session_messages)

    # React to user input in the selected session
    if prompt := st.chat_input("What is up?"):
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)

        # Add user message to the session history
        session_messages.append({"role": "user", "content": prompt})

        # Update session name to be the last question asked by the user
        session_data["name"] = prompt[:50]  # Limit session name length to 50 characters

        # Get assistant's response and display it
        with st.chat_message("assistant"):
            response = chatbot.chat_engine.chat(prompt)
            st.markdown(response.response)

        # Add assistant's message to the session history
        session_messages.append({"role": "assistant", "content": response.response})
else:
    st.write("Please select or start a new chat session.")