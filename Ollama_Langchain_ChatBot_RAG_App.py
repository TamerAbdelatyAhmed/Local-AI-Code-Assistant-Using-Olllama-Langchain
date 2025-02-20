import streamlit as st
from dotenv import load_dotenv
from langchain_ollama import ChatOllama

from langchain_core.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
    MessagesPlaceholder
)

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import SQLChatMessageHistory

load_dotenv('.env')

st.title("AI Code Assistant")
st.write("Ask me anything about coding!")

base_url = "http://localhost:11434"
model = "codellama"
user_id = st.text_input("Enter your user id")

def get_session_history(session_id):
    return SQLChatMessageHistory(session_id, "sqlite:///chat_history.db")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if st.button("Start Chatting"):
    st.session_state.chat_history = []
    history = get_session_history(user_id)
    history.clear()

# Display chat history in the sidebar
st.sidebar.title("Chat History")
for message in st.session_state.chat_history:
    st.sidebar.write(f"{message['role'].capitalize()}: {message['content']}")

# llm Setup
llm = ChatOllama(model=model, base_url=base_url)

system = SystemMessagePromptTemplate.from_template("You are a helpful AI code assistant.")
human = HumanMessagePromptTemplate.from_template("{input}")
messages = [system, MessagesPlaceholder(variable_name='history'), human]
prompt = ChatPromptTemplate(messages=messages)
chain = prompt | llm | StrOutputParser()
runnable_with_history = RunnableWithMessageHistory(chain, get_session_history, input_messages_key='input', history_messages_key='history')

def chat_with_llm(message):
    output = runnable_with_history.invoke(
        {'input': message},
        config={'configurable': {'session_id': user_id}}
    )
    return output

prompt = st.chat_input("How can I help you with your code?")

if prompt:
    st.write("You: ","\n", prompt)
    output = chat_with_llm(prompt)
    st.session_state.chat_history.append({'role': 'user', 'content': prompt})
    st.session_state.chat_history.append({'role': 'bot', 'content': output})
    st.write("Bot: ", "\n", output)

# Button to start a new chat
if st.sidebar.button("New Chat"):
    st.session_state.chat_history = []
    history = get_session_history(user_id)
    history.clear()
    st.query_params.clear()
