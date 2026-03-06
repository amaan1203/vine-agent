"""
VINE-Agent: ReWOO LangGraph Pipeline (v2)
==========================================
Full rewrite with multi-modal context fusion:
  - AgentState extended with sensor_context, drone_context, context_priority
  - New live_context_node: fetches live sensor + latest drone blocks before planning
  - Planner emits ContextPriority blueprint alongside the ReWOO plan
  - Context Assembler orders PRIMARY/SECONDARY/TERTIARY evidence sections
  - Solver receives assembled multi-modal context block
  - AIMessage (not HumanMessage) wraps the final answer

Graph:
  START → query_rewrite → live_context → make_raptor
        → planner → worker → solver → summary → END
"""

from __future__ import annotations

import logging
import re
from collections import defaultdict, deque
from datetime import datetime
from typing import Annotated, Dict, List, Optional, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from context_assembler import ContextAssembler, ContextPriority, infer_priority_from_query
from prompts import (
    DISTILL_QUERY_PROMPT,
    PLANNER_PROMPT,
    QUERY_REWRITE_PROMPT,
    SOLVER_PROMPT,
    SUMMARY_PROMPT,
)
from raptor import Raptor
from retriever import VINERetriever

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# State
# ─────────────────────────────────────────────────────────────────────────────

class SubQueryResult(TypedDict):
    node_ids: List[str]
    sub_query: str


class AgentState(TypedDict):
    """Full state flowing through the VINE-Agent LangGraph."""
    messages: Annotated[List[BaseMessage], add_messages]

    # Rolling memory
    summary: str

    # ReWOO plan
    plan_string: str
    evidence: Dict[str, str]

    # RAPTOR instance
    raptor: Optional[object]
    node_ids: List[str]

    # Multi-modal live context
    sensor_context: str       # Formatted SensorContextBlock strings (live)
    drone_context: str        # Formatted DroneContextBlock strings (latest flight)
    last_flight_date: str     # ISO date string of most recent drone flight
    data_availability: str    # Human-readable summary of data freshness for Planner

    # Context priority (parsed from planner output)
    context_priority: Optional[Dict]   # {"primary": ..., "secondary": ..., "tertiary": ...}

    # Assembled multi-modal context (built just before Solver)
    assembled_context: str

    # Intermediate
    sub_queries: List[SubQueryResult]
    sub_queries_context: str
    unique_query: str


# ─────────────────────────────────────────────────────────────────────────────
# Helper: leaf-node traversal from RAPTOR adjacency
# ─────────────────────────────────────────────────────────────────────────────

def get_leaf_nodes(node_ids: List[str], adj: Dict[str, List[str]]) -> List[str]:
    """DFS to collect leaf nodes reachable from given node_ids."""
    visited: set = set()
    leaves: set = set()

    def dfs(node: str):
        if node in visited:
            return
        visited.add(node)
        if node not in adj or not adj[node]:
            leaves.add(node)
        else:
            for child in adj[node]:
                dfs(child)

    for n in node_ids:
        dfs(n)
    return list(leaves)


# ─────────────────────────────────────────────────────────────────────────────
# VINEAgent class
# ─────────────────────────────────────────────────────────────────────────────

class VINEAgent:
    """
    Full VINE-Agent pipeline with multi-modal context fusion.

    Usage:
        agent = VINEAgent(
            llm=..., retriever=..., embed_model=...,
            sensor_context_fn=..., drone_blocks=...
        )
        graph = agent.compile()
        result = graph.invoke({"messages": [HumanMessage("When to irrigate?")], "summary": ""})
    """

    def __init__(
        self,
        llm,
        retriever: VINERetriever,
        embed_model,
        raptor_k: int = 100,
        raptor_top_k: int = 10,
        sensor_context_fn=None,      # Callable[[], str] → live sensor context string
        drone_blocks=None,            # List[DroneContextBlock] for latest flight
        last_flight_date: str = "",
    ):
        self.llm = llm
        self.retriever = retriever
        self.embed_model = embed_model
        self.raptor_k = raptor_k
        self.raptor_top_k = raptor_top_k
        self.sensor_context_fn = sensor_context_fn
        self.drone_blocks = drone_blocks or []
        self.last_flight_date = last_flight_date
        self._context_assembler = ContextAssembler()

    # ─── Node: Query Rewrite ──────────────────────────────────────────────────

    def query_rewrite_node(self, state: AgentState) -> AgentState:
        """Resolve follow-up query pronouns/references using conversation summary."""
        summary = state.get("summary", "")
        query = str(state["messages"][-1].content).strip()

        if not summary:
            return {**state}

        prompt = QUERY_REWRITE_PROMPT.format_messages(summary=summary, query=query)
        new_query = self.llm.invoke(prompt)
        if hasattr(new_query, "content"):
            new_query = new_query.content.strip()
        else:
            new_query = str(new_query).strip()

        state["messages"][-1] = HumanMessage(content=new_query)
        logger.info(f"[AGENT] Query rewritten: {new_query[:80]}…")
        return {**state}

    # ─── Node: Live Context Loader ────────────────────────────────────────────

    def live_context_node(self, state: AgentState) -> AgentState:
        """
        Fetch live sensor context and latest drone imagery context.
        Builds data_availability string for the Planner.
        This runs BEFORE make_raptor so the planner knows what data is fresh.
        """
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M UTC")

        # 1. Sensor context (real-time)
        sensor_ctx = ""
        if self.sensor_context_fn is not None:
            try:
                sensor_ctx = self.sensor_context_fn()
                logger.info("[AGENT] Live sensor context loaded.")
                logger.debug(f"[AGENT] Sensor Context: {sensor_ctx}")
            except Exception as e:
                logger.warning(f"Sensor context fetch failed: {e}")
                sensor_ctx = "[SENSOR] Sensor data temporarily unavailable."
        else:
            sensor_ctx = "[SENSOR] No live sensor feed configured (POC mode)."

        # 2. Drone context (latest flight)
        drone_ctx = ""
        if self.drone_blocks:
            from drone_encoder import DroneImageryEncoder
            enc = DroneImageryEncoder()
            drone_ctx = enc.get_latest_flight_summary(self.drone_blocks)
        else:
            drone_ctx = "[DRONE] No drone flight data available."

        # 3. Data availability summary for Planner
        flight_date = self.last_flight_date or "unknown"
        data_avail = (
            f"- Live IoT Sensor Data: AVAILABLE (as of {now_str}, 5-min resolution)\n"
            f"- Drone Multispectral Imagery: "
            + (f"AVAILABLE (last flight: {flight_date})" if self.drone_blocks
               else "UNAVAILABLE (no flight data loaded)")
            + f"\n- Agricultural Knowledge Base (RAPTOR): AVAILABLE (will be retrieved)"
        )

        return {
            **state,
            "sensor_context": sensor_ctx,
            "drone_context": drone_ctx,
            "last_flight_date": flight_date,
            "data_availability": data_avail,
        }

    # ─── Node: Make RAPTOR ────────────────────────────────────────────────────

    def make_raptor_node(self, state: AgentState) -> AgentState:
        """HyDE + FAISS → ColBERT → RAPTOR build."""
        query = str(state["messages"][-1].content.strip())

        logger.info(f"[AGENT] Building RAPTOR index for query: {query[:50]}...")
        final_docs = self.retriever.retrieve(query)
        logger.info(f"[AGENT] RAPTOR building over {len(final_docs)} documents…")

        raptor = Raptor(
            docs=final_docs,
            summariser_model=self.llm,
            embed_model=self.embed_model,
        )
        return {**state, "raptor": raptor}

    # ─── Node: Planner ────────────────────────────────────────────────────────

    def planner_node(self, state: AgentState) -> AgentState:
        """
        Emit a ReWOO-style evidence-gathering plan + ContextPriority blueprint.
        """
        question = state["messages"][-1].content
        summary = state.get("summary", "")
        data_availability = state.get("data_availability", "Not available.")

        prompt = PLANNER_PROMPT.format_messages(
            question=question,
            summary=summary,
            data_availability=data_availability,
        )
        result = self.llm.invoke(prompt)
        if hasattr(result, "content"):
            result = result.content

        logger.info(f"[AGENT] Plan generated:\n{result[:300]}…")

        # Parse ContextPriority from plan
        cp = ContextPriority.from_plan_string(result)
        logger.info(f"[AGENT] Context priority: {cp}")

        return {
            **state,
            "plan_string": result,
            "context_priority": {
                "primary":   cp.primary,
                "secondary": cp.secondary,
                "tertiary":  cp.tertiary,
                "reasoning": cp.reasoning,
            },
        }

    # ─── Node: Worker ─────────────────────────────────────────────────────────

    def worker_node(self, state: AgentState) -> AgentState:
        """
        Execute each Plan step:
          - Raptor[query] → retrieve from RAPTOR collapsed tree
          - LLM[context] → ask LLM to reason/disambiguate
        """
        plan_string = state["plan_string"]
        raptor: Raptor = state["raptor"]
        summary = state.get("summary", "")
        evidence: Dict[str, str] = {}

        regex = r"Plan:\s*(.+?)\s*(#E\d+)\s*=\s*(\w+)\s*\[([^\]]+)\]"
        matches = re.findall(regex, plan_string)
        node_ids: set = set()

        for description, step_id, tool, tool_input in matches:
            logger.info(f"[AGENT] Executing Worker Step {step_id}: {tool}[{tool_input[:60]}]")

            resolved_input = tool_input
            context_dependency = False
            for prev_id, prev_text in evidence.items():
                if prev_id in resolved_input:
                    resolved_input = resolved_input.replace(prev_id, prev_text)
                    context_dependency = True

            if tool == "Raptor":
                search_query = resolved_input

                if context_dependency and len(resolved_input) > 200:
                    distill_prompt = DISTILL_QUERY_PROMPT.format(
                        description=description,
                        resolved_input=resolved_input[:2000],
                    )
                    compressed = self.llm.invoke(distill_prompt)
                    if hasattr(compressed, "content"):
                        search_query = compressed.content.strip().strip('"')
                    else:
                        search_query = str(compressed).strip().strip('"')
                    logger.info(f"[AGENT]   → Distilled query: {search_query}")

                nodes = raptor.retrieve_collapsed(search_query, top_k=self.raptor_top_k)
                matched_ids = [n["id"] for n in nodes]
                node_ids.update(matched_ids)
                context_text = "\n".join([f"- {n['text']}" for n in nodes])
                evidence[step_id] = context_text

            elif tool == "LLM":
                msg = (f"{description}\n\nInput Context:\n{resolved_input}\n\n"
                       f"Conversation context:\n{summary}")
                response = self.llm.invoke(msg)
                if hasattr(response, "content"):
                    evidence[step_id] = response.content
                else:
                    evidence[step_id] = str(response)

            else:
                logger.warning(f"[AGENT] Unknown tool '{tool}' — skipping {step_id}.")
                evidence[step_id] = "No evidence (unknown tool)."

        return {**state, "evidence": evidence, "node_ids": list(node_ids)}

    # ─── Node: Context Assembler ──────────────────────────────────────────────

    def assemble_context_node(self, state: AgentState) -> AgentState:
        """
        Build the ordered multi-modal evidence block from Planner's priority blueprint.
        Runs between Worker and Solver.
        """
        plan_string = state.get("plan_string", "")
        evidence = state.get("evidence", {})
        sensor_ctx = state.get("sensor_context", "")
        drone_ctx  = state.get("drone_context", "")
        last_flight = state.get("last_flight_date", "")

        # Reconstruct flat RAPTOR evidence from worker
        raptor_evidence_parts = []
        regex = r"(#E\d+)\s*=\s*(\w+)\s*\[([^\]]+)\]"
        for line in plan_string.split("\n"):
            if "=" in line:
                match = re.search(regex, line)
                if match:
                    step_id = match.group(1)
                    result_text = evidence.get(step_id, "")
                    if result_text:
                        raptor_evidence_parts.append(f"{line}\n→ {result_text}")
        raptor_evidence = "\n---\n".join(raptor_evidence_parts)

        # Parse or infer priority
        cp_dict = state.get("context_priority") or {}
        if cp_dict:
            cp = ContextPriority(
                primary=cp_dict.get("primary", "raptor_kb"),
                secondary=cp_dict.get("secondary", "sensor_data"),
                tertiary=cp_dict.get("tertiary", "drone_imagery"),
                reasoning=cp_dict.get("reasoning", ""),
            )
        else:
            query = str(state["messages"][-1].content)
            cp = infer_priority_from_query(query)
            logger.info(f"Priority inferred from query: {cp}")

        assembled = self._context_assembler.assemble(
            raptor_evidence=raptor_evidence,
            sensor_block=sensor_ctx,
            drone_block=drone_ctx,
            priority=cp,
            query_time=datetime.now(),
            last_flight_date=last_flight,
        )
        logger.info(f"[AGENT] [ContextPriority] {cp.primary.upper()} → {cp.secondary.upper()} → {cp.tertiary.upper()}")
        logger.debug(f"[AGENT] Assembled Context Length: {len(assembled)} chars")
        return {**state, "assembled_context": assembled}

    # ─── Node: Solver ─────────────────────────────────────────────────────────

    def solver_node(self, state: AgentState) -> AgentState:
        """Synthesise a cited agricultural recommendation from all evidence."""
        plan_string = state["plan_string"]
        evidence = state["evidence"]
        question = state["messages"][-1].content
        summary = state.get("summary", "")
        assembled_context = state.get("assembled_context", "")

        # Reconstruct plan with filled evidence (for plan_evidence field)
        full_context = ""
        regex = r"(#E\d+)\s*=\s*(\w+)\s*\[([^\]]+)\]"
        for line in plan_string.split("\n"):
            if "=" in line:
                match = re.search(regex, line)
                if match:
                    step_id = match.group(1)
                    result_text = evidence.get(step_id, "No evidence found.")
                    full_context += f"{line}\nRESULT: {result_text}\n---\n"
                else:
                    full_context += f"{line}\n"
            else:
                full_context += f"{line}\n"

        prompt = SOLVER_PROMPT.format_messages(
            question=question,
            summary=summary,
            assembled_context=assembled_context,
            plan_evidence=full_context,
        )
        answer = self.llm.invoke(prompt)
        if hasattr(answer, "content"):
            answer_text = answer.content
        else:
            answer_text = str(answer)

        # Use AIMessage for correct conversation role
        return {**state, "messages": [AIMessage(content=answer_text)]}

    # ─── Node: Summary Update ─────────────────────────────────────────────────

    def summary_node(self, state: AgentState) -> AgentState:
        """Update rolling conversation memory after each turn."""
        summary = state.get("summary", "")
        messages = state["messages"]
        if len(messages) < 2:
            return {**state}

        # Last human message is the query, last AI message is the answer
        query = ""
        answer = ""
        for msg in reversed(messages):
            if isinstance(msg, AIMessage) and not answer:
                answer = str(msg.content).strip()
            elif isinstance(msg, HumanMessage) and not query:
                query = str(msg.content).strip()
            if query and answer:
                break

        logger.info(f"[AGENT] Summarizing conversation context...")
        if not query:
            return {**state}
        if not summary:
            summary = query

        prompt = SUMMARY_PROMPT.format_messages(
            summary=summary, query=query, answer=answer
        )
        new_summary = self.llm.invoke(prompt)
        if hasattr(new_summary, "content"):
            new_summary = new_summary.content.strip()
        else:
            new_summary = str(new_summary).strip()

        return {**state, "summary": new_summary}

    # ─── Graph Assembly ───────────────────────────────────────────────────────

    def compile(self):
        """
        Build and compile the LangGraph state machine.

        Graph flow:
          START → query_rewrite → live_context → make_raptor
                → planner → worker → assemble_context → solver
                → summary → END
        """
        graph = StateGraph(AgentState)

        graph.add_node("query_rewrite",    self.query_rewrite_node)
        graph.add_node("live_context",     self.live_context_node)
        graph.add_node("make_raptor",      self.make_raptor_node)
        graph.add_node("planner",          self.planner_node)
        graph.add_node("worker",           self.worker_node)
        graph.add_node("assemble_context", self.assemble_context_node)
        graph.add_node("solver",           self.solver_node)
        graph.add_node("summary",          self.summary_node)

        graph.add_edge(START,              "query_rewrite")
        graph.add_edge("query_rewrite",    "live_context")
        graph.add_edge("live_context",     "make_raptor")
        graph.add_edge("make_raptor",      "planner")
        graph.add_edge("planner",          "worker")
        graph.add_edge("worker",           "assemble_context")
        graph.add_edge("assemble_context", "solver")
        graph.add_edge("solver",           "summary")
        graph.add_edge("summary",          END)

        return graph.compile()


# ─────────────────────────────────────────────────────────────────────────────
# VINEChatBot: conversational wrapper
# ─────────────────────────────────────────────────────────────────────────────

class VINEChatBot:
    """Conversational wrapper around compiled VINEAgent graph."""

    def __init__(self, agent: VINEAgent):
        self.graph = agent.compile()
        self.summary: str = ""

    def chat(self, query: str) -> dict:
        """Execute one conversational turn. Returns answer + metadata."""
        import time
        t0 = time.time()

        out = self.graph.invoke({
            "messages": [HumanMessage(content=query)],
            "summary": self.summary,
            "raptor": None,
            "plan_string": "",
            "evidence": {},
            "node_ids": [],
            "sub_queries": [],
            "sub_queries_context": "",
            "unique_query": "",
            "sensor_context": "",
            "drone_context": "",
            "last_flight_date": "",
            "data_availability": "",
            "context_priority": None,
            "assembled_context": "",
        })

        self.summary = out.get("summary", self.summary)

        # Last AIMessage is the answer
        answer = ""
        for msg in reversed(out["messages"]):
            if isinstance(msg, AIMessage):
                answer = msg.content
                break

        node_ids = out.get("node_ids", [])
        raptor   = out.get("raptor")
        cp       = out.get("context_priority") or {}
        elapsed  = time.time() - t0

        # Collect leaf evidence texts
        leaf_texts = []
        if raptor and node_ids:
            adj = raptor.adjacency
            leaves = get_leaf_nodes(node_ids, adj)
            leaf_texts = [
                raptor.ALL_NODES[lid]["text"]
                for lid in leaves
                if lid in raptor.ALL_NODES
            ]

        return {
            "answer": answer,
            "node_ids": node_ids,
            "leaf_evidence": leaf_texts,
            "summary": self.summary,
            "context_priority": cp,
            "assembled_context": out.get("assembled_context", ""),
            "time_seconds": round(elapsed, 2),
        }
