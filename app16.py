import os
from dotenv import load_dotenv
from openai import OpenAI  # For Ollama client
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from swarm import Swarm, Agent
import fitz  # PyMuPDF for PDF handling
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load environment variables
load_dotenv()

# Ollama client setup
ollama_client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")

# LlamaIndex settings (use Ollama models)
Settings.llm = Ollama(model="llama3.2:latest", base_url="http://127.0.0.1:11434")
Settings.embed_model = OllamaEmbedding(base_url="http://127.0.0.1:11434", model_name="nomic-embed-text:latest")

multiplier = 1.0


# Function to create the RAG index
def create_rag_index(pdf_filepath="docs"):
    try:
        documents = SimpleDirectoryReader(pdf_filepath, recursive=True).load_data()
        index = VectorStoreIndex.from_documents(documents)
        return index
    except Exception as e:
        print(f"Error creating RAG index: {e}")
        return None


# Load the RAG index
rag_index = create_rag_index()


# PDF Agent logic
class PDFHandler:
    def __init__(self, pdf_path):
        self.pdf_doc = fitz.open(pdf_path) if pdf_path else None

    def extract_keywords(self, text, max_keywords=5):
        """Extract keywords using TF-IDF for dynamic keyword extraction."""
        vectorizer = TfidfVectorizer(stop_words="english", max_features=max_keywords)
        response = vectorizer.fit_transform([text])
        feature_array = np.array(vectorizer.get_feature_names_out())
        tfidf_sorting = np.argsort(response.toarray()).flatten()[::-1]
        return feature_array[tfidf_sorting][:max_keywords].tolist()

    def find_relevant_pages(self, query):
        """Find relevant pages using TF-IDF and cosine similarity."""
        if not self.pdf_doc:
            return []

        relevant_pages = []
        all_pages_text = [self.pdf_doc[i].get_text("text") for i in range(len(self.pdf_doc))]
        vectorizer = TfidfVectorizer(stop_words="english")
        vectorizer.fit(all_pages_text)
        query_vector = vectorizer.transform([query]).toarray()

        for i, page_text in enumerate(all_pages_text):
            page_vector = vectorizer.transform([page_text]).toarray()
            similarity = cosine_similarity(query_vector, page_vector)[0][0]
            print(f"Page {i+1} similarity: {similarity}")
            if similarity > 0.5:
                relevant_pages.append(i)


        return relevant_pages

    def display_page_as_image(self, page_num):
        """Convert a PDF page to image bytes for rendering in Streamlit."""
        if not self.pdf_doc:
            return None
        try:
            page = self.pdf_doc.load_page(page_num)
            pix = page.get_pixmap()
            return pix.tobytes("png")  # Return image bytes
        except Exception as e:
            print(f"Error displaying PDF page: {e}")
            return None


# Create a PDF handler instance
pdf_handler = PDFHandler(pdf_path="photos/photos.pdf")


# Function for querying RAG
def query_rag(query_str):
    try:
        query_engine = rag_index.as_query_engine()
        response = query_engine.query(query_str)
        return str(response) if response else "No relevant information found."
    except Exception as e:
        print(f"Error in query_rag: {e}")
        return "I encountered an error while processing your query."


# Define Triage Agent instructions
def triage_agent_instructions(context_variables):
    history = context_variables.get("conversation_history", "")
    modifier = context_variables.get("multiplier", "")
    return f"""You are a triage agent. Respond to the user in clear, plain text without including technical metadata such as tool_calls or function calls.

    Here is your instruction:
    1. If the user asks a question related to the document or exercises, use the handoff function to retreive data from the rag agent.
    2. If there is a number of sets/reps  in your answer, modify it to {modifier} times of the original.
    3. If the user asks for a training regimen or anything of sorts, use the handoff function to retreive data and modify the amount of training to {modifier} times of the
     original. You DO NOT need to change or add anything. simply adjust the amount to {modifier} times of the original
    4. If the user asks about BMI (body mass index) or is mentioning anything related to someone's (user or somone else) weight and/or height, handoff to the bmi agent
     which will calculate the BMI of said person.
    5. If the query does not relate to the document, exercises, or BMI, respond that you do not know.

    Conversation history, for context only. DO NOT answer any question that is not asked right now. (if it is empty, that means the conversation just started):
    {history}"""


# Define RAG Agent instructions
def rag_agent_instructions(context_variables):
    history = context_variables.get("conversation_history", "")
    return """You are a RAG agent. Use the query_rag function to retrieve information."""

    # Conversation history for context (if it is empty, that means the conversation just started):
    # {history}"""


# Define PDF Agent instructions
def pdf_agent_instructions(context_variables):
    history = context_variables.get("conversation_history", "")
    return """You are a PDF agent. Your task is to identify what exercise is mentioned by rag agent, find related pages from RAG agent's answer and display the relevant
    pages as images."""
    # Conversation history for context (if it is empty, that means the conversation just started):
    # {history}"""


def bmi_agent_instructions(context_variables):
    history = context_variables.get("conversation_history", "")
    multiplier = context_variables.get("multiplier", "")
    return """You are a BMI agent, your task is to calculate the user's bmi and the user's classification based on the user's BMI.
    you can calculate the BMI using the calculate_bmi function.
    Use the calculated BMI to clasify the user and return both the BMI and the classification to triage agent.

    this is the bmi classification
    - bmi < 18.5: "Underweight"
    - 18.5 <= bmi <= 24.9: "Ideal Weight"
    - 25 <= bmi <= 29.9: "Overweight"
    - 30 <= bmi: "Obese" """

    # Conversation history for context (if it is empty, that means the conversation just started):
    # {history}"""


# PDF query handler
def query_pdf(query_str):
    relevant_pages = pdf_handler.find_relevant_pages(query_str)
    if relevant_pages:
        response = ""
        for page_num in relevant_pages:
            page_image = pdf_handler.display_page_as_image(page_num)
            if page_image:
                st.image(page_image, caption=f"Page {page_num + 1}", use_column_width=True)
                response += f"Displaying relevant page: {page_num + 1}\n"
        return response.strip()
    return "No relevant pages found."


def handoff(query, context_variables=None):
    """Unified handoff function for RAG, PDF, and BMI agents."""
    context_variables = context_variables or {}

    # Query the RAG index
    try:
        rag_response = query_rag(query)
        pdf_response = query_pdf(query)
        final_response = f"RAG Agent Response:\n{rag_response}\n\nPDF Agent Response:\n{pdf_response or 'No relevant pages found.'}"
        print(f"RAG Response for query '{query}': {rag_response}")
        print(f"PDF Agent query (processed): {pdf_handler.extract_keywords(rag_response)}")

        return final_response
    except Exception as e:
        return f"An error occurred while processing your query: {e}"


def handoff_to_bmi_agent(query):
    return bmi_agent


def calculate_bmi(weight, height, context_variables):
    """Calculate BMI and determine the multiplier."""
    try:
        bmi = weight / ((height/100) ** 2)
        if bmi < 18.5:
            multiplier = 0.8
        elif 18.5 <= bmi <= 24.9:
            multiplier = 1.0
        elif 25 <= bmi <= 29.9:
            multiplier = 1.2
        else:
            multiplier = 1.3
        context_variables["multiplier"] = multiplier
        return round(bmi, 2)
    except ZeroDivisionError:
        return None  # Return default multiplier if height is 0
    except Exception as e:
        print(f"Error calculating BMI: {e}")
        return None


# Define the agents
triage_agent = Agent(
    name="Triage Agent",
    instructions=triage_agent_instructions,
    functions=[handoff, handoff_to_bmi_agent],
    model="llama3.2:latest"
)

rag_agent = Agent(
    name="RAG Agent",
    instructions=rag_agent_instructions,
    functions=[query_rag],
    model="llama3.2:latest"
)

pdf_agent = Agent(
    name="PDF Agent",
    instructions=pdf_agent_instructions,
    functions=[query_pdf],
    model="llama3.2:latest"
)

bmi_agent = Agent(
    name="BMI Agent",
    instructions=bmi_agent_instructions,
    functions=[calculate_bmi],
    model="llama3.2:latest"
)

# Initialize the swarm
swarm_client = Swarm(client=ollama_client)
swarm_client.agents = [triage_agent, rag_agent, pdf_agent, bmi_agent]


# Streamlit interface
st.title("Multi-Agent Swarm Gym Chatbot")

if "chat_sessions" not in st.session_state:
    st.session_state.chat_sessions = {}

if "selected_session" not in st.session_state:
    st.session_state.selected_session = None


# Function to create a new session
def create_new_session():
    session_id = f"session_{len(st.session_state.chat_sessions) + 1}"
    st.session_state.chat_sessions[session_id] = {
        "name": f"Session {len(st.session_state.chat_sessions) + 1}",
        "messages": [{"role": "assistant", "content": "Hello! How can I assist you today?"}]
    }
    st.session_state.selected_session = session_id


# Sidebar session management
with st.sidebar:
    st.header("Chat Sessions")
    if not st.session_state.chat_sessions:
        create_new_session()

    session_names = {sid: data["name"] for sid, data in st.session_state.chat_sessions.items()}
    selected_name = st.selectbox("Select a session", list(session_names.values()))
    if selected_name:
        st.session_state.selected_session = list(session_names.keys())[list(session_names.values()).index(selected_name)]

    if st.button("Start New Session"):
        create_new_session()

# Load the current session
if st.session_state.selected_session:
    current_session = st.session_state.chat_sessions[st.session_state.selected_session]

    # Display messages from the current session
    for message in current_session["messages"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Handle user input
    if prompt := st.chat_input("Ask me anything!"):
        current_session["messages"].append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.markdown(prompt)

        conversation_history = "\n".join(
            [f"{m['role']}: {m['content']}" for m in current_session["messages"]]
        )

        try:
            response = swarm_client.run(
                agent=triage_agent,
                messages=[{"role": "user", "content": prompt}],
                context_variables={"conversation_history": conversation_history, "multiplier": multiplier}
            )

            # Extract and display the assistant's reply
            assistant_reply = response.messages[-1].get("content", "I'm sorry, I couldn't process your request.")
            with st.chat_message("assistant"):
                st.markdown(assistant_reply)

            print(assistant_reply)

            # Check if RAG agent response mentions relevant pages and display them
            if "relevant page" in assistant_reply.lower():
                relevant_pages = pdf_handler.find_relevant_pages(assistant_reply)
                for page_num in relevant_pages:
                    page_image = pdf_handler.display_page_as_image(page_num)
                    if page_image:
                        st.image(page_image, caption=f"Relevant Page {page_num + 1}", use_column_width=True)

            # Log additional details for debugging (optional)
            tool_calls = response.messages[-1].get("tool_calls", [])
            if tool_calls:
                print("Tool calls:", tool_calls)

            current_session["messages"].append({"role": "assistant", "content": assistant_reply})

        except Exception as e:
            error_message = f"An error occurred: {e}"
            with st.chat_message("assistant"):
                st.markdown(error_message)
            current_session["messages"].append({"role": "assistant", "content": error_message})

else:
    st.write("Please select or start a new session.")