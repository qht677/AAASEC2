"""Day 1: enterprise research agent built as a LangGraph state graph."""

import operator
import os
from datetime import datetime
from typing import Annotated, Dict, List

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_core.vectorstores import InMemoryVectorStore
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

load_dotenv()
USE_FAKE = os.getenv("USE_FAKE", "0") == "1"


class AgentState(TypedDict):
    topic: str
    search_query: str
    collected_data: List[Dict]
    analyzed_data: List[Dict]
    quality_score: int
    iteration_count: int
    final_report: str
    execution_logs: Annotated[List[str], operator.add]


class QualityScore(BaseModel):
    """Validated research-quality result."""

    score: int = Field(ge=1, le=10)
    reasoning: str = Field(description="One-sentence justification")


if USE_FAKE:
    from langchain_core.embeddings import DeterministicFakeEmbedding

    class FakeLLM:
        def invoke(self, messages):
            class Response:
                content = (
                    "Enterprise agentic AI benefits from explicit state, "
                    "guardrails, observability, and human oversight."
                )

            return Response()

    class FakeEvaluator:
        def __init__(self):
            self.calls = 0

        def invoke(self, messages):
            self.calls += 1
            if self.calls == 1:
                return QualityScore(score=5, reasoning="The first pass needs more breadth.")
            return QualityScore(score=8, reasoning="The retry adds sufficient breadth and depth.")

    class FakeSearch:
        def invoke(self, payload):
            query = payload["query"]
            return {
                "results": [
                    {
                        "title": f"Architecture guidance for {query}",
                        "url": "https://example.com/architecture",
                        "content": f"Research about architecture, governance, and controls for {query}.",
                    },
                    {
                        "title": f"Adoption evidence for {query}",
                        "url": "https://example.com/adoption",
                        "content": f"Research about adoption, risk, cost, and ROI for {query}.",
                    },
                ]
            }

    llm = FakeLLM()
    evaluator = FakeEvaluator()
    search_tool = FakeSearch()
    embeddings = DeterministicFakeEmbedding(size=256)
else:
    from langchain_openai import ChatOpenAI
    from langchain_tavily import TavilySearch

    llm = ChatOpenAI(
        model=os.getenv("OPENROUTER_MODEL", "nvidia/nemotron-3-super-120b-a12b:free"),
        temperature=0,
        base_url="https://openrouter.ai/api/v1",
    )
    evaluator = llm.with_structured_output(QualityScore)
    search_tool = TavilySearch(max_results=5)
    try:
        from langchain_huggingface import HuggingFaceEmbeddings

        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
    except ImportError:
        from langchain_core.embeddings import DeterministicFakeEmbedding

        embeddings = DeterministicFakeEmbedding(size=256)

vector_store = InMemoryVectorStore(embeddings)


def log_line(message: str) -> str:
    return f"[{datetime.now():%H:%M:%S}] {message}"


def collect_node(state: AgentState):
    iteration = state["iteration_count"] + 1
    query = state["topic"]
    if iteration > 1:
        query += f" enterprise evidence governance risks ROI updated pass {iteration}"
    results = search_tool.invoke({"query": query}).get("results", [])
    return {
        "search_query": query,
        "collected_data": results,
        "iteration_count": iteration,
        "execution_logs": [log_line(f"collect: pass={iteration}, sources={len(results)}")],
    }


def store_memory_node(state: AgentState):
    texts = [item.get("content", "") for item in state["collected_data"]]
    texts = [text for text in texts if text]
    if texts:
        vector_store.add_texts(texts)
    return {"execution_logs": [log_line(f"store_memory: saved={len(texts)}")]}


def analyze_node(state: AgentState):
    analyzed = []
    for source in state["collected_data"]:
        content = source.get("content", "")
        related = vector_store.similarity_search(content, k=2) if content else []
        memory = "\n".join(document.page_content for document in related)
        response = llm.invoke(
            [
                HumanMessage(
                    content=(
                        f"Analyze this source for an enterprise report about {state['topic']}. "
                        "Extract evidence, risks, benefits, and practical implications.\n\n"
                        f"Source: {content}\n\nRelated memory:\n{memory}"
                    )
                )
            ]
        )
        analyzed.append(
            {
                "title": source.get("title", "Untitled source"),
                "url": source.get("url", ""),
                "insights": response.content,
            }
        )
    return {
        "analyzed_data": analyzed,
        "execution_logs": [log_line(f"analyze: completed={len(analyzed)}")],
    }


def evaluate_node(state: AgentState):
    material = "\n\n".join(item["insights"] for item in state["analyzed_data"])
    result = evaluator.invoke(
        [
            HumanMessage(
                content=(
                    "Score the breadth, evidence, relevance, and actionability of this research "
                    f"from 1 to 10.\n\n{material}"
                )
            )
        ]
    )
    return {
        "quality_score": result.score,
        "execution_logs": [log_line(f"evaluate: score={result.score}; {result.reasoning}")],
    }


def report_node(state: AgentState):
    sources = "\n\n".join(
        f"### {item['title']}\nSource: {item['url']}\n{item['insights']}"
        for item in state["analyzed_data"]
    )
    response = llm.invoke(
        [
            HumanMessage(
                content=(
                    f"Write a concise enterprise report about '{state['topic']}'. Include an "
                    "executive summary, key findings, risks, recommendations, and source links.\n\n"
                    f"Research:\n{sources}"
                )
            )
        ]
    )
    return {
        "final_report": response.content,
        "execution_logs": [log_line("report: generated")],
    }


def audit_node(state: AgentState):
    return {
        "execution_logs": [
            log_line(
                "audit: complete | "
                f"iterations={state['iteration_count']} | score={state['quality_score']} | "
                f"sources={len(state['collected_data'])}"
            )
        ]
    }


def quality_router(state: AgentState) -> str:
    if state["quality_score"] >= 7 or state["iteration_count"] >= 3:
        return "report"
    return "collect"


workflow = StateGraph(AgentState)
workflow.add_node("collect", collect_node)
workflow.add_node("store_memory", store_memory_node)
workflow.add_node("analyze", analyze_node)
workflow.add_node("evaluate", evaluate_node)
workflow.add_node("report", report_node)
workflow.add_node("audit", audit_node)
workflow.add_edge(START, "collect")
workflow.add_edge("collect", "store_memory")
workflow.add_edge("store_memory", "analyze")
workflow.add_edge("analyze", "evaluate")
workflow.add_conditional_edges(
    "evaluate", quality_router, {"collect": "collect", "report": "report"}
)
workflow.add_edge("report", "audit")
workflow.add_edge("audit", END)


if __name__ == "__main__":
    app = workflow.compile(checkpointer=InMemorySaver())
    print(app.get_graph().draw_mermaid())
    initial_state: AgentState = {
        "topic": "Enterprise Agentic AI Systems",
        "search_query": "",
        "collected_data": [],
        "analyzed_data": [],
        "quality_score": 0,
        "iteration_count": 0,
        "final_report": "",
        "execution_logs": [],
    }
    config = {"configurable": {"thread_id": "day1-run"}}
    final_state = initial_state
    for final_state in app.stream(initial_state, config, stream_mode="values"):
        if final_state["execution_logs"]:
            print(final_state["execution_logs"][-1])
    print("\nFINAL REPORT\n", final_state["final_report"])
    print("\nFULL EXECUTION LOG")
    print("\n".join(final_state["execution_logs"]))
