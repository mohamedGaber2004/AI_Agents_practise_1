from typing import TypedDict
from langgraph.graph import StateGraph , START , END
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage , AIMessage , HumanMessage , AnyMessage
from pydantic import BaseModel , Field
import os 
import streamlit as st



if "messages" not in st.session_state : 
    st.session_state.messages = []



st.title("chatbot")


for message in st.session_state.messages : 
    if isinstance(message,HumanMessage):
        with st.chat_message("user"):
            st.write(message.content)
    elif isinstance(message,AIMessage) : 
        with st.chat_message("bot") : 
            st.write(message.content)



llm = ChatOllama(
    model= "qwen3:4b",
    temperature=0
)

sys_msg = SystemMessage(
    content="""You are a friendly chatbot . 
    Engage in a pleasant conversation with the user"""
    )


class chat_state(TypedDict) : 
    messages : list[AnyMessage]


def chat(chat_state) : 
    response = llm.invoke([sys_msg]+chat_state['messages'])
    return {"messages":chat_state['messages'] + [AIMessage(content=response.content)]}


builder = StateGraph(chat_state) 
builder.add_node("chat",chat)

builder.add_edge(START,"chat")
builder.add_edge("chat",END)

graph = builder.compile()



if prompt := st.chat_input("How are you ?"):
    user_msg = HumanMessage(content=prompt)
    st.session_state.messages.append(user_msg)

    with st.chat_message("user") : 
        st.write(prompt)


    with st.chat_message("assistant") : 
        response = graph.invoke({"messages" : st.session_state.messages})

        st.write(response['messages'][-1].content)
        st.session_state.messages.append(AIMessage(content=response['messages'][-1].content))