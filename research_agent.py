from typing import TypedDict, List
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage, AnyMessage, SystemMessage, AIMessage
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field
import streamlit as st
from functions import google_search


# ===========================
# Streamlit UI
# ===========================
st.set_page_config(page_title="Research Chatbot", page_icon="🔍")

if "messages" not in st.session_state:
    st.session_state.messages = []

st.title("🔍 Research Chatbot")

# Display chat history
for msg in st.session_state.messages:
    role = "user" if isinstance(msg, HumanMessage) else "assistant"
    with st.chat_message(role):
        st.write(msg.content)


# ===========================
# LLM Configuration
# ===========================
llm = ChatOllama(
    model="qwen3:4b",
    temperature=0
)


# ===========================
# SCHEMAS
# ===========================
class ResearchRequired(BaseModel):
    is_required: bool = Field(
        description="Whether Google research is required to answer the user's question."
    )


class SearchQuery(BaseModel):
    query: str = Field(
        description="Google search query to use."
    )


class AnswerSatisfying(BaseModel):
    is_satisfying: bool = Field(
        description="Whether the produced answer is satisfying."
    )


# ===========================
# STATE TYPE
# ===========================
class SearchAgentState(TypedDict):
    messages: List[AnyMessage]
    research_query: str
    research_results: str
    research_conclusion: str
    next: str


# ===========================
# NODE FUNCTIONS
# ===========================
def decide_research_required(state: SearchAgentState):
    sys = SystemMessage(content="Decide whether Google research is needed.")
    schema_llm = llm.with_structured_output(
        schema=ResearchRequired.model_json_schema(), strict=True
    )

    response = schema_llm.invoke([sys] + state["messages"])

    if response["is_required"]:
        return {"next": "write_search_query"}
    else:
        return {"next": "answer_user"}


def write_search_query(state: SearchAgentState):
    sys = SystemMessage(content="Write a Google search query based on user conversation.")
    schema_llm = llm.with_structured_output(
        schema=SearchQuery.model_json_schema(), strict=True
    )

    response = schema_llm.invoke([sys] + state["messages"])
    return {"research_query": response["query"]}


def perform_research(state: SearchAgentState):
    results = google_search(state["research_query"])

    sys = SystemMessage(content="Summarize the following Google search results.")
    usr = HumanMessage(content=str(results))

    response = llm.invoke([sys, usr])

    return {"research_results": response.content}


def write_conclusion(state: SearchAgentState):
    sys = SystemMessage(content=
        "Write a short conclusion summarizing the research to answer the user. "
        "Keep it under 500 characters."
    )
    usr = HumanMessage(content=f"Research results: {state['research_results']}")

    response = llm.invoke([sys, usr])

    return {"research_conclusion": response.content}


def decide_answer_satisfying(state: SearchAgentState):
    sys = SystemMessage(content="Evaluate whether the answer is satisfying.")
    usr = HumanMessage(
        content=f"User question: {state['messages'][-1].content}\n"
                f"Answer: {state['research_conclusion']}"
    )

    schema_llm = llm.with_structured_output(
        schema=AnswerSatisfying.model_json_schema(), strict=True
    )
    response = schema_llm.invoke([sys, usr])

    if response["is_satisfying"]:
        return {
            "next": "end_node",
            "messages": state["messages"] + [AIMessage(content=state["research_conclusion"])]
        }
    else:
        return {"next": "write_search_query"}


def answer_user(state: SearchAgentState):
    sys = SystemMessage(content="Answer the user's query using only the conversation.")
    response = llm.invoke([sys] + state["messages"])

    return {"messages": state["messages"] + [AIMessage(content=response.content)]}


def end_node(state: SearchAgentState):
    return {}


# ===========================
# GRAPH BUILDING
# ===========================
builder = StateGraph(SearchAgentState)

builder.add_node("decide_research_required", decide_research_required)
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
builder.add_edge("answer_user", END)
builder.add_edge("end_node", END)

# Conditional edges
builder.add_conditional_edges(
    "decide_research_required",
    lambda state: state.get("next"),
    {
        "write_search_query": "write_search_query",
        "answer_user": "answer_user"
    }
)

builder.add_conditional_edges(
    "decide_answer_satisfying",
    lambda state: state.get("next"),
    {
        "end_node": "end_node",
        "write_search_query": "write_search_query"
    }
)

graph = builder.compile()


# ===========================
# MAIN CHAT INPUT
# ===========================
if user_prompt := st.chat_input("Ask me anything..."):
    user_message = HumanMessage(content=user_prompt)
    st.session_state.messages.append(user_message)

    with st.chat_message("user"):
        st.write(user_prompt)

    # Build initial state with defaults
    state = {
        "messages": st.session_state.messages,
        "research_query": "",
        "research_results": "",
        "research_conclusion": "",
        "next": ""
    }

    result = graph.invoke(state)

    ai_msg = result["messages"][-1].content

    with st.chat_message("assistant"):
        st.write(ai_msg)

    st.session_state.messages.append(AIMessage(content=ai_msg))
