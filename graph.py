import os
from dotenv import load_dotenv

from typing import Literal, TypedDict, Annotated
from pydantic import BaseModel

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

from rag_setup import retriever
from tool import lookup_order

load_dotenv()

llm = ChatOpenAI(
    model="openrouter/free",
    openai_api_key=os.environ.get("OPENROUTER_API_KEY"),
    openai_api_base="https://openrouter.ai/api/v1",
    max_tokens=700,
)
order_llm = llm.bind_tools([lookup_order])


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    just_answered: bool


class RouteDecision(BaseModel):
    next: Literal["faq_agent", "order_agent", "escalation_agent", "FINISH"]
    reasoning: str


SUPERVISOR_PROMPT = """You are a routing supervisor for an e-commerce support chatbot.
Decide which specialist should handle the LATEST user message:
- faq_agent: policy questions (returns, exchanges, shipping, warranty, payment, cancellation)
- order_agent: questions about a SPECIFIC order's status/tracking (needs an order ID)
- escalation_agent: customer is angry, asks for a human, or issue is outside policy/order scope
- FINISH: a specialist already answered this turn

Only choose FINISH if the most recent message is already a specialist's answer."""


def supervisor_node(state: AgentState):
    if state.get("just_answered"):
        return {"next": "FINISH", "just_answered": False}

    structured_llm = llm.with_structured_output(RouteDecision)
    decision = structured_llm.invoke(
        [SystemMessage(content=SUPERVISOR_PROMPT)] + state["messages"]
    )
    return {"next": decision.next}


def faq_agent_node(state: AgentState):
    last_user_msg = state["messages"][-1].content
    docs = retriever.invoke(last_user_msg)
    context = "\n\n".join(d.page_content for d in docs)

    prompt = f"""You are a helpful e-commerce support assistant.
Answer using ONLY the policy context below. If it doesn't cover the question,
say you'll escalate to a specialist.

Context:
{context}

Customer question: {last_user_msg}"""

    response = llm.invoke([HumanMessage(content=prompt)])
    return {"messages": [AIMessage(content=response.content)], "just_answered": True}


def order_agent_node(state: AgentState):
    response = order_llm.invoke(state["messages"])

    if response.tool_calls:
        tool_call = response.tool_calls[0]
        result = lookup_order.invoke(tool_call["args"])
        follow_up = llm.invoke(
            state["messages"]
            + [response, HumanMessage(content=f"Tool result: {result}. Answer the customer clearly.")]
        )
        return {"messages": [AIMessage(content=follow_up.content)], "just_answered": True}

    return {"messages": [response], "just_answered": True}


def escalation_agent_node(state: AgentState):
    msg = ("I understand this needs closer attention. I've flagged this conversation "
           "for a human agent, who will follow up within 24 hours.")
    return {"messages": [AIMessage(content=msg)], "just_answered": True}


def build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("supervisor", supervisor_node)
    graph.add_node("faq_agent", faq_agent_node)
    graph.add_node("order_agent", order_agent_node)
    graph.add_node("escalation_agent", escalation_agent_node)

    graph.set_entry_point("supervisor")
    graph.add_conditional_edges(
        "supervisor",
        lambda state: state["next"],
        {
            "faq_agent": "faq_agent",
            "order_agent": "order_agent",
            "escalation_agent": "escalation_agent",
            "FINISH": END,
        },
    )
    graph.add_edge("faq_agent", "supervisor")
    graph.add_edge("order_agent", "supervisor")
    graph.add_edge("escalation_agent", "supervisor")

    return graph.compile(checkpointer=MemorySaver())


if __name__ == "__main__":
    app = build_graph()
    config = {"configurable": {"thread_id": "test-1"}}

    while True:
        user_input = input("You: ")
        if user_input.lower() in ("quit", "exit"):
            break
        result = app.invoke(
            {"messages": [("user", user_input)], "just_answered": False},
            config=config,
        )
        print("Bot:", result["messages"][-1].content)