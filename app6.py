import streamlit as st
from llama_index.llms.ollama import Ollama
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings
from llama_index.embeddings.fastembed import FastEmbedEmbedding
from llama_index.core.llms import ChatMessage
from llama_index.core.storage.chat_store import SimpleChatStore
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.chat_engine import CondensePlusContextChatEngine
from difflib import SequenceMatcher
import fitz  # PyMuPDF for handling PDF content (pages, images)
import uuid  # For generating unique session IDs


# Define the Chatbot class
class Chatbot:
    def __init__(self, llm="llama3.1:latest", embedding_model="intfloat/multilingual-e5-large", vector_store=None, pdf_path=None):
        self.Settings = self.set_setting(llm, embedding_model)

        # Indexing
        self.index = self.load_data()

        # Memory
        self.memory = self.create_memory()

        # Chat Engine
        self.chat_engine = self.create_chat_engine(self.index)

        # Load the PDF document
        self.pdf_doc = self.load_pdf(pdf_path)

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

        # PDF handling methods
    def load_pdf(self, pdf_path):
        if pdf_path:
            return fitz.open(pdf_path)  # Load the PDF document
        return None

    def find_relevant_pages(self, query):
        relevant_pages = []
        # Convert query to lowercase
        query_lower = query.lower()
        # List of known exercise names in the PDF (you could extract these from the document)
        known_exercises = [
            "180 jump squat", "bulgarian split squat", "alternating dumbbell swing", "bear squat",
            "Alternating Side Lunge", "Alternating Side Lunge Touch", "Ankle Circles", "Ankle Hops",
            "Arms Cross Side Lunge", "Boxer Squat Punch", "Burpees", "Butterfly Stretch", "Calf Raises", 
            "Calf Stretch", "Circles In The Sky", "Core Control Rear Lunge", "Cossack Squat", "Cross Jacks", 
            "Curtsy Lunge", "Curtsy Lunge Side Kick Raise", "Curtsy Lunge Side Kick", "Diamond Kicks", 
            "Double Pulse Squat Jump", "Dumbbell Thrusters", "Figure 8 Squat", "Fingertip To Toe Jacks", 
            "Flutter Kick Squats", "Forward Jump Shuffle Back", "Frog Jumps", "Front And Back Lunges", 
            "Gate Swings", "Good Mornings", "Pilates Grasshopper", "Half Squat Jab Cross", "Hamstring Stretch", 
            "Advanced Bridge", "Back Leg Lifts", "Band Donkey Kicks", "Band Kickback", "Band Reverse Plank", 
            "Basketball Shots", "Booty Squeeze", "Bridge And Twist", "Butt Kicks", "Chest Fly Glute Bridge", 
            "Clamshell", "Deadlift Upright Row", "Donkey Kicks", "Downward Dog Crunch", "Dumbbell Skier Swing", 
            "Fire Hydrant", "Abdominal Bridge", "Ankle Tap Push Ups", "Balance Chop", "Band Leg Abduction Crunch", 
            "Bent Leg Jack Knife", "Bent Over Twist", "Bicycle Crunches", "Bird Dogs", "Boat Twist", 
            "Breakdancer Kick", "Chest Press With Legs Extended", "Crab Kicks", "Crab Toe Touches", "Cross Crunches", 
            "Crunch Chop", "Crunches", "Dead Bug", "Donkey Kick Twist", "Double Leg Stretch", "Dumbbell Leg Loop", 
            "Dumbbell Side Bend", "Flutter Kicks", "Frog Crunches", "Glute Bridge Overhead Reach", 
            "Alternate Heel Touchers", "Inchworm", "Inner Thigh Squeeze And Lift", "Inverted V Plank", 
            "Kick Crunch", "Bicep Curls", "Biceps Stretch", "Concentration Curl", "Standing Cross Chest Curl", 
            "Hammer Curls", "Side Lunge Curl", "Split Squat Curl", "Tabletop Reverse Pike", "Up Down Plank", 
            "V Sit Curl Press", "Butterfly Dips", "Lying Tricep Extension", "One Arm Tricep Push Up", 
            "One Arm Triceps Kickback", "Single Leg Tricep Dips", "Squat With Overhead Tricep Extension", 
            "Tricep Dips", "Dumbbell Triceps Extension", "Dumbbell Triceps Kickback", "Triceps Stretch", 
            "Arm Circles", "Arm Swings", "Arnold Shoulder Press", "Bear Walk", "Bent Over Lateral Raise", 
            "Bent Over Row", "Bent Over Row Press", "Big Arm Circles", "Dumbbell Front Raise", "Dumbbell Lateral Raise", 
            "Dumbbell Overhead Rainbow", "Dumbbell Punches", "Single Arm Dumbbell Snatch", 
            "Elbow Squeeze Shoulder Press", "Half Squat Jab Cross", "Hindu Push Ups", "Knee And Elbow Press Up", 
            "Alternating Lunge Front Raise", "Lunge Punch", "Medicine Ball Overhead Circles", "Pike Push Up", 
            "Dumbbell Push Press", "Reverse Lunge Shoulder Press", "Dumbbell Shoulder Press", "Shoulder Rolls", 
            "Shoulder Stretch", "Dumbbell Shoulder To Shoulder Press", "Side Lunge Band Lateral Raise", 
            "Speed Bag Punches", "Squat Band Front Raise", "Standing Neck Stretch", "Standing Y Raise", 
            "Wall Shoulder Stretch", "Around the Worlds", "Asymmetrical Push Up", "Chest Fly", 
            "Chest Press Punch Up", "Chest Stretch", "Decline Push Up", "Dumbbell Pullover", 
            "Open Arm Chest Stretch", "Plie Squat Scoop Up", "Push Up", "Spiderman Push Ups", 
            "Stability Ball Chest Press", "Staggered Arm Push Up", "Standing Chest Fly", "Alternating Superman", 
            "Back Extensions", "Back Stretch", "Bow and Arrow Squat Pull", "Cobra Lat Pulldown", 
            "Lawnmower Band Pull", "LawnMower Pull", "Lower Back Stretch", "Mid Back Band Pull", "Neck Rolls", 
            "Pilates Swimming", "Prone Back Extension", "Rolling Like A Ball", "Superman", "Upper Back Stretch", 
            "Waist Slimmer Squat", "Wide Row", "Wood Chop"]  # Add more as needed
        # Dynamically find which exercise the user is asking about by matching known exercise names
        matching_exercise = None
        for exercise in known_exercises:
            if exercise in query_lower:
                matching_exercise = exercise
                break
        if matching_exercise and self.pdf_doc:
            for page_num in range(len(self.pdf_doc)):
                page = self.pdf_doc.load_page(page_num)
                text = page.get_text("text").lower()
                # Match the exercise directly within the page text
                if matching_exercise in text:
                    relevant_pages.append(page_num)
        return relevant_pages


    def display_page_as_image(self, page_num):
        page = self.pdf_doc.load_page(page_num)
        pix = page.get_pixmap()
        img_bytes = pix.tobytes("png")  # Convert page to image bytes
        st.image(img_bytes, caption=f"Page {page_num + 1}", use_column_width=True)

    # Respond with text and relevant PDF page images
    def respond_with_pdf_pages(self, query):
        # Search relevant pages
        relevant_pages = self.find_relevant_pages(query)

        if relevant_pages:
            for page_num in relevant_pages:
                self.display_page_as_image(page_num)
        else:
            st.write("No relevant pages found in the document.")


# Initialize chat session if not available
if "chat_sessions" not in st.session_state:
    st.session_state.chat_sessions = {}

# Initialize selected session
if "selected_session" not in st.session_state:
    st.session_state.selected_session = None

# Function to create a new chat session with the default name
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
st.title("Gym Exercise Chatbot")

# Initialize the chatbot with your PDF file
chatbot = Chatbot(pdf_path="photos/Photos.pdf")  # Use the path to your uploaded PDF

# Display chat history of selected session
if st.session_state.selected_session:
    session_data = st.session_state.chat_sessions[st.session_state.selected_session]
    session_messages = session_data["messages"]

    for message in session_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    chatbot.set_chat_history(session_messages)

    # React to user input in the selected session
    if prompt := st.chat_input("Ask something"):
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

        # Display relevant PDF pages as images based on the prompt
        chatbot.respond_with_pdf_pages(prompt)
else:
    st.write("Please select or start a new chat session.")