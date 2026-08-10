"""Day 2: multi-agent research team using the supervisor pattern."""

import operator
import os
from datetime import datetime
from typing import Annotated, List, Literal

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

load_dotenv()
USE_FAKE = os.getenv("USE_FAKE", "0") == "1"
MAX_REVISIONS = 2
MAX_TURNS = 12


class TeamState(TypedDict):
    task: str
    research_notes: Annotated[List[str], operator.add]
    analysis: str
    draft: str
    critique: str
    revision_count: int
    turn_count: int
    next_agent: str
    execution_logs: Annotated[List[str], operator.add]


class RouterDecision(BaseModel):
    next_agent: Literal["researcher", "analyst", "writer", "critic", "FINISH"]
    reason: str = Field(description="One sentence explaining the choice")


PERSONAS = {
    "researcher": (
        "You are an evidence-focused researcher. Condense supplied search results into factual "
        "notes with source URLs. Do not analyze strategy, recommend actions, or write the report."
    ),
    "analyst": (
        "You are an enterprise analyst. Turn research notes into balanced findings about value, "
        "risk, cost, governance, and adoption. Do not search the web or write the final report."
    ),
    "writer": (
        "You are an executive writer. Produce a concise, source-grounded report with an executive "
        "summary, findings, risks, and recommendations. Do not search or invent evidence."
    ),
    "critic": (
        "You are a strict fact and quality reviewer. Compare the draft with the research. Reply "
        "exactly 'APPROVED' when it is ready, otherwise begin with 'REVISE:' and list concrete fixes. "
        "Do not rewrite the report or search the web."
    ),
}


if USE_FAKE:
    class FakeLLM:
        def invoke(self, messages):
            system = messages[0].content.lower() if isinstance(messages[0], SystemMessage) else ""

            class Response:
                content = ""

            response = Response()
            if "researcher" in system:
                response.content = "Sources show benefits from specialization and risks from coordination overhead."
            elif "analyst" in system:
                response.content = "Adopt selectively where specialization offsets added cost and operational risk."
            elif "executive writer" in system:
                response.content = (
                    "Executive summary: use multi-agent systems selectively.\n\n"
                    "Findings: specialization can improve complex workflows but increases cost and failure modes.\n\n"
                    "Recommendation: pilot a bounded workflow with metrics, guardrails, and human approval."
                )
            else:
                response.content = "APPROVED"
            return response

    class FakeSupervisor:
        def invoke(self, messages):
            text = messages[-1].content
            if "research_ready=False" in text:
                return RouterDecision(next_agent="researcher", reason="Research is missing.")
            if "analysis_ready=False" in text:
                return RouterDecision(next_agent="analyst", reason="Analysis is missing.")
            if "draft_ready=False" in text:
                return RouterDecision(next_agent="writer", reason="A draft is missing.")
            if "critique_ready=False" in text:
                return RouterDecision(next_agent="critic", reason="The draft needs review.")
            if "critique_status=REVISE" in text:
                return RouterDecision(next_agent="writer", reason="The writer must address feedback.")
            return RouterDecision(next_agent="FINISH", reason="The draft is approved.")

    class FakeSearch:
        def invoke(self, payload):
            return {
                "results": [
                    {
                        "title": "Multi-agent enterprise patterns",
                        "url": "https://example.com/patterns",
                        "content": "Supervisor teams offer specialization with coordination overhead.",
                    },
                    {
                        "title": "AI governance guidance",
                        "url": "https://example.com/governance",
                        "content": "Bounded autonomy, observability, and human approval reduce operational risk.",
                    },
                ]
            }

    llm = FakeLLM()
    supervisor_llm = FakeSupervisor()
    search_tool = FakeSearch()
else:
    from langchain_openai import ChatOpenAI
    from langchain_tavily import TavilySearch

    llm = ChatOpenAI(
        model=os.getenv("OPENROUTER_MODEL", "nvidia/nemotron-3-super-120b-a12b:free"),
        temperature=0,
        base_url="https://openrouter.ai/api/v1",
    )
    supervisor_llm = llm.with_structured_output(RouterDecision)
    search_tool = TavilySearch(max_results=4)


def run_persona(role: str, user_content: str) -> str:
    response = llm.invoke(
        [SystemMessage(content=PERSONAS[role]), HumanMessage(content=user_content)]
    )
    return response.content


def log_line(message: str) -> str:
    return f"[{datetime.now():%H:%M:%S}] {message}"


def supervisor_node(state: TeamState):
    turn = state["turn_count"] + 1
    critique_status = (
        "REVISE"
        if state["critique"].startswith("REVISE")
        else "APPROVED"
        if state["critique"].startswith("APPROVED")
        else "NONE"
    )
    status = (
        f"research_ready={bool(state['research_notes'])}; "
        f"analysis_ready={bool(state['analysis'])}; "
        f"draft_ready={bool(state['draft'])}; "
        f"critique_ready={bool(state['critique'])}; "
        f"critique_status={critique_status}; "
        f"revisions={state['revision_count']}; turn={turn}"
    )
    decision = supervisor_llm.invoke(
        [
            SystemMessage(
                content=(
                    "You supervise a research team. Choose the next missing step in order: "
                    "researcher, analyst, writer, critic. Send REVISE feedback to writer and "
                    "finish only after approval."
                )
            ),
            HumanMessage(content=f"Task: {state['task']}\nStatus: {status}"),
        ]
    )
    next_agent = decision.next_agent
    reason = decision.reason
    if turn > MAX_TURNS:
        next_agent, reason = "FINISH", "Maximum supervisor turns reached."
    elif (
        next_agent in {"writer", "critic"}
        and state["draft"]
        and state["revision_count"] >= MAX_REVISIONS
    ):
        next_agent, reason = "FINISH", "Maximum draft revisions reached."
    return {
        "next_agent": next_agent,
        "turn_count": turn,
        "execution_logs": [log_line(f"supervisor: {next_agent} — {reason}")],
    }


def researcher_node(state: TeamState):
    results = search_tool.invoke({"query": state["task"]}).get("results", [])
    raw = "\n\n".join(
        f"Title: {item.get('title', '')}\nURL: {item.get('url', '')}\n{item.get('content', '')}"
        for item in results
    )
    notes = run_persona(
        "researcher", f"Task: {state['task']}\n\nSearch results:\n{raw}"
    )
    return {
        "research_notes": [notes],
        "execution_logs": [log_line(f"researcher: sources={len(results)}")],
    }


def analyst_node(state: TeamState):
    notes = "\n\n".join(state["research_notes"])
    analysis = run_persona(
        "analyst", f"Task: {state['task']}\n\nResearch notes:\n{notes}"
    )
    return {
        "analysis": analysis,
        "execution_logs": [log_line("analyst: analysis completed")],
    }


def writer_node(state: TeamState):
    revising = state["critique"].startswith("REVISE")
    prompt = (
        f"Task: {state['task']}\n\nResearch:\n{' '.join(state['research_notes'])}"
        f"\n\nAnalysis:\n{state['analysis']}"
    )
    if revising:
        prompt += f"\n\nPrevious draft:\n{state['draft']}\n\nRequired fixes:\n{state['critique']}"
    draft = run_persona("writer", prompt)
    return {
        "draft": draft,
        "critique": "",
        "revision_count": state["revision_count"] + (1 if revising else 0),
        "execution_logs": [log_line("writer: revised draft" if revising else "writer: first draft")],
    }


def critic_node(state: TeamState):
    critique = run_persona(
        "critic",
        f"Task: {state['task']}\n\nResearch:\n{' '.join(state['research_notes'])}"
        f"\n\nDraft:\n{state['draft']}",
    )
    normalized = critique.strip()
    if not (normalized.startswith("APPROVED") or normalized.startswith("REVISE")):
        normalized = f"REVISE: {normalized}"
    return {
        "critique": normalized,
        "execution_logs": [log_line(f"critic: {normalized.split(':', 1)[0]}")],
    }


def route_from_supervisor(state: TeamState) -> str:
    return state["next_agent"]


workflow = StateGraph(TeamState)
workflow.add_node("supervisor", supervisor_node)
workflow.add_node("researcher", researcher_node)
workflow.add_node("analyst", analyst_node)
workflow.add_node("writer", writer_node)
workflow.add_node("critic", critic_node)
workflow.add_edge(START, "supervisor")
workflow.add_conditional_edges(
    "supervisor",
    route_from_supervisor,
    {
        "researcher": "researcher",
        "analyst": "analyst",
        "writer": "writer",
        "critic": "critic",
        "FINISH": END,
    },
)
for worker in ("researcher", "analyst", "writer", "critic"):
    workflow.add_edge(worker, "supervisor")


if __name__ == "__main__":
    app = workflow.compile(checkpointer=InMemorySaver())
    print(app.get_graph().draw_mermaid())
    initial_state: TeamState = {
        "task": "Should our company adopt multi-agent AI systems in 2026?",
        "research_notes": [],
        "analysis": "",
        "draft": "",
        "critique": "",
        "revision_count": 0,
        "turn_count": 0,
        "next_agent": "",
        "execution_logs": [],
    }
    config = {"configurable": {"thread_id": "day2-run"}}
    final_state = initial_state
    for final_state in app.stream(initial_state, config, stream_mode="values"):
        if final_state["execution_logs"]:
            print(final_state["execution_logs"][-1])
    print("\nFINAL DRAFT\n", final_state["draft"])
    print(
        f"\nSTATS: turns={final_state['turn_count']}, "
        f"revisions={final_state['revision_count']}, critique={final_state['critique']}"
    )
    print("\nFULL EXECUTION LOG")
    print("\n".join(final_state["execution_logs"]))
