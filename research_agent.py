from typing import TypedDict , List
from langgraph.graph import StateGraph , START , END 
from langchain_core.messages import HumanMessage , AnyMessage , SystemMessage , AIMessage
from langchain_ollama import ChatOllama 
from pydantic import BaseModel , Field
import streamlit as st
from functions import google_search


if "messages" not in st.session_state:
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


class ResearchRequired(BaseModel) : 
    is_required : bool = Field(
        description="whether the agent needs to perform research (google search) to answer user question or not."
    )


class SearchQuery(BaseModel) : 
    query : str = Field(
        description="The google search query to be used for research."
    )



class ResearchResults (BaseModel) : 
    results : list[str] = Field(
        description="A list of relevant search results obtained from google search."
    )


class AnswerSatisfying (BaseModel) : 
    is_satisfying : bool = Field(
        description="whether the current answer is answering the user's query or not."
    )



class search_agent_state(TypedDict) : 
    messages : list[AnyMessage]
    research_query : str 
    research_results : str 
    research_conclusion : str 
    next : str


def decide_resarch_or_not(search_agent_state) : 
    sys_message = SystemMessage(content="Based on the conversation you have with the user , decide whether research is required or not.")
    schema_llm = llm.with_structured_output(schema=ResearchRequired.model_json_schema(),strict = True)
    response = schema_llm.invoke(
        [sys_message]+search_agent_state['messages']
    )

    if response['is_required'] : 
        return {"next" : "write_search_query"}
    else : 
        return {"next" : "answer the user"}


def write_search_query(search_agent_state) : 
    sys_message = SystemMessage(content="Your task is to define a search query for google search based on the conversation you have with the user.")
    schema_llm = llm.with_structured_output(schema=SearchQuery.model_json_schema(),strict = True)
    response = schema_llm.invoke(
        [sys_message]+search_agent_state['messages']
    )
    return {"research_query" : response['query']}

def perform_research(chat_agent_state) : 
    search_res = google_search(chat_agent_state['research_query'])
    sys_message = SystemMessage(content="you recieved the google search's top result.Create good summary")
    user_msg = HumanMessage(content=f"the following are search results obtained from google searching")
    response = llm.invoke(
        [sys_message]+[user_msg]
    )

    return {"research_results" : response.content}


def write_conclusion(search_agent_state):
    sys_message = SystemMessage(content="Your task is to write a concise conclusion based on the research results provided, in the form of a response to the user. The conclusion should directly address the user's query and summarize the key findings from the research. Keep the conclusion under 500 characters.")

    usr_message = HumanMessage(content=f"The following are the research results obtained from Google search: {search_agent_state['research_results']}.")

    response = llm.invoke([sys_message] + [usr_message])

    return {"research_conclusion": response.content}

def decide_answer_satisfying(search_agent_state):
    sys_message = SystemMessage(content="Based on the question from the user, decide whether the following answer is answering the user's query satisfactory.")
    usr_message = HumanMessage(content=f"User's question: {search_agent_state['messages'][-1]} \nAnswer based on research: {search_agent_state['research_conclusion']}")

    schema_llm = llm.with_structured_output(schema=AnswerSatisfying.model_json_schema(), strict=True)

    response = schema_llm.invoke([sys_message] + [usr_message])


    if response['is_satisfying']:
        return {"next": "end_node","messages": search_agent_state['messages'] + [AIMessage(content=search_agent_state['research_conclusion'])]}
    else:
        return {"next": "write_search_query"}
    


def answer_user(search_agent_state):

    sys_message = SystemMessage(content="Answer the user's query based on the conversation.")

    response = llm.invoke([sys_message] + search_agent_state['messages'])

    return {"messages": search_agent_state['messages'] + [AIMessage(content=response.content)]}


def end_node(search_agent_state):
    return {}



builder = StateGraph(search_agent_state)
builder.add_node("decide_research_required", decide_resarch_or_not)
builder.add_node("write_search_query", write_search_query)
builder.add_node("perform_research", perform_research)
builder.add_node("write_conclusion", write_conclusion)
builder.add_node("decide_answer_satisfying", decide_answer_satisfying)
builder.add_node("answer_user", answer_user)
builder.add_node("end_node", end_node)

builder.add_edge(START, "decide_research_required")
builder.add_edge("write_search_query", "perform_research")
builder.add_edge("perform_research", "write_conclusion")
builder.add_edge("write_conclusion", "decide_answer_satisfying")
builder.add_edge("end_node", END)

builder.add_conditional_edges(
    "decide_research_required",
    lambda state: search_agent_state.get("next"),
    {"write_search_query": "write_search_query", "answer_user": "answer_user"}
)

builder.add_conditional_edges(
    "decide_answer_satisfying",
    lambda state: search_agent_state.get("next"),
    {"end_node": "end_node", "write_search_query": "write_search_query"}
)

graph = builder.compile()



if prompt := st.chat_input("How are you?"):
    user_message = HumanMessage(content=prompt)
    st.session_state.messages.append(user_message)

    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        response = graph.invoke({"messages": st.session_state.messages})

        st.write(response['messages'][-1].content)
        st.session_state.messages.append(AIMessage(content=response['messages'][-1].content))