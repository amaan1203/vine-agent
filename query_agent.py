import logging
import re
from datetime import datetime
from typing import Dict, List, Optional
from langchain_core.messages import AIMessage, HumanMessage

from planner_agent import PlannerAgent
from recommender_agent import RecommenderAgent
from context_assembler import ContextAssembler
from retriever import VINERetriever
from raptor import Raptor
from alert_store import AlertStore

logger = logging.getLogger(__name__)

class AlertAwareQueryAgent:
    def __init__(
        self,
        llm,
        raptor_tree,
        embed_model,
        sensor_context_fn=None,
        drone_blocks=None,
        last_flight_date: str = "",
        raptor_top_k: int = 8,
    ):
        self.llm = llm
        self.raptor_tree = raptor_tree
        self.embed_model = embed_model
        
        self.planner = PlannerAgent(llm)
        self.recommender = RecommenderAgent(llm)
        self.context_assembler = ContextAssembler()
        self.alert_store = AlertStore()
        
        self.sensor_context_fn = sensor_context_fn
        self.drone_blocks = drone_blocks or []
        self.last_flight_date = last_flight_date
        self.raptor_top_k = raptor_top_k
        
        self.summary = "" # Rolling conversation memory

    def chat(self, user_query: str) -> Dict:
        logger.info(f"[QUERY AGENT] Processing user query: {user_query}")
        
        # 1. Fetch current active alerts
        try:
            active_alerts = self.alert_store.get_active_alerts(limit=5)
            alert_ctx = "\n".join([
                f"- {a['severity']} | Block {a['block']} ({a['variety']}): {a['alert_type']} — {a['description']}"
                for a in active_alerts
            ])
            if not alert_ctx:
                alert_ctx = "None currently active."
        except Exception:
            alert_ctx = "Alert Store unavailable."

        # 2. Construct context-aware query
        logger.info(f"[QUERY AGENT] Constructing enhanced query with active alerts...")
        logger.debug(f"[QUERY AGENT] Active Alerts Context:\n{alert_ctx}")
        enriched_query = (
            f"═══ CURRENT SYSTEM ALERT STATE ═══\n"
            f"Active alerts:\n{alert_ctx}\n"
            f"═══════════════════════════════════\n\n"
            f"User Question: {user_query}"
        )

        # 3. Live Modalities
        sensor_ctx = self.sensor_context_fn() if self.sensor_context_fn else "LIVE SENSOR UNAVAILABLE"
        
        drone_ctx = ""
        if self.drone_blocks:
            from drone_encoder import DroneImageryEncoder
            enc = DroneImageryEncoder()
            drone_ctx = enc.get_latest_flight_summary(self.drone_blocks)
        
        data_avail = (
            f"- Live IoT Sensor Data: AVAILABLE\n"
            f"- Drone Multispectral Imagery: " + ("AVAILABLE" if self.drone_blocks else "UNAVAILABLE") + "\n"
            f"- RAPTOR Knowledge Base: AVAILABLE"
        )

        # 4. Planner
        logger.info(f"[QUERY AGENT] Passing query to Planner...")
        logger.debug(f"[QUERY AGENT] Data Availability:\n{data_avail}")
        planner_res = self.planner.generate_plan(enriched_query, data_availability=data_avail)
        plan_string = planner_res["plan_string"]
        priority = planner_res["priority"]

        # 5. Execute Plan
        evidence = {}
        
        for line in plan_string.split("\n"):
            match = re.search(r"(#E\d+)\s*=\s*([A-Za-z0-9_]+)\s*\[([^\]]+)\]", line)
            if not match:
                continue
                
            step_id, tool, tool_input = match.groups()
            description = line.split(step_id)[0].replace("Plan:", "").strip()
            
            resolved_input = tool_input
            logger.info(f"[QUERY AGENT] Executing step {step_id}: {tool}...")
            logger.debug(f"[QUERY AGENT] Tool Input: {resolved_input[:100]}...")
            for prev_id, prev_text in evidence.items():
                if prev_id in resolved_input:
                    resolved_input = resolved_input.replace(prev_id, prev_text)

            if tool == "Raptor":
                # Serialize embedding calls to prevent Apple Silicon segfault
                from proactive_agent import _EMBED_LOCK
                with _EMBED_LOCK:
                    nodes = self.raptor_tree.retrieve_collapsed(resolved_input, top_k=self.raptor_top_k)
                context_text = "\n".join([f"- {n['text']}" for n in nodes])
                evidence[step_id] = context_text
            elif tool == "LLM":
                msg = f"{description}\nContext:\n{resolved_input}"
                rsp = self.llm.invoke(msg)
                evidence[step_id] = rsp.content if hasattr(rsp, "content") else str(rsp)
            else:
                evidence[step_id] = "No evidence."

        # 7. Assembled Context
        logger.info(f"[QUERY AGENT] Assembling multi-modal context...")
        raptor_evidence_parts = []
        for line in plan_string.split("\n"):
            match = re.search(r"(#E\d+)\s*=", line)
            if match and match.group(1) in evidence:
                raptor_evidence_parts.append(f"{line}\n→ {evidence[match.group(1)]}")
        
        assembled_context = self.context_assembler.assemble(
            raptor_evidence="\n---\n".join(raptor_evidence_parts),
            sensor_block=sensor_ctx,
            drone_block=drone_ctx,
            priority=priority,
            query_time=datetime.now(),
            last_flight_date=self.last_flight_date
        )

        # 8. Recommender
        logger.info(f"[QUERY AGENT] Calling Recommender with assembled context...")
        logger.debug(f"[QUERY AGENT] Assembled Context Size: {len(assembled_context)} chars")
        answer_text = self.recommender.generate_recommendation(
            input_question=user_query, # Send original query to solver
            plan_string=plan_string,
            step_evidence=evidence,
            assembled_context=assembled_context,
            summary=self.summary
        )
        
        # 9. Update Conversation Memory (very simple)
        self.summary += f"\nQ: {user_query}\nA: {answer_text[:200]}...\n"

        logger.info(f"[QUERY AGENT] Recommender generated response.")
        logger.debug(f"[QUERY AGENT] Response Text:\n{answer_text[:200]}...")

        return {
            "answer": answer_text,
            "context_priority": priority,
            "assembled_context": assembled_context
        }
