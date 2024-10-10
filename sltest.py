import streamlit as st
from ollama import Client
 
st.title("LLM with Streamlit")
 
# Set Ollama Client
client = Client(host="http://127.0.0.1:11434")
 
# handles stream response back from LLM
def stream_parser(stream):
    for chunk in stream:
        yield chunk['message']['content']
 
 
# Set a default model
if "ollama_model" not in st.session_state:
    st.session_state["ollama_model"] = "llama3.1:latest"
 
# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
 
# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
 
# Accept user input
if prompt := st.chat_input("What is up?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
 
# Display assistant response in chat message container
    with st.chat_message("assistant"):
        stream = client.chat(
            model=st.session_state["ollama_model"],
            messages=[
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state.messages
                ],
            stream=True,
        )
        response = st.write_stream(stream_parser(stream))
    st.session_state.messages.append({"role": "assistant", "content": response})