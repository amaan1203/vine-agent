"""
VINE-Agent v3: Proactive Agent Pipeline
Triggered by the Rule Engine. Takes an Alert, constructs a context-aware query,
and runs the Planner -> ContextAssembler -> Recommender pipeline to produce a recommendation.
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional
from langchain_core.messages import AIMessage, HumanMessage
from alert_engine import Alert
from planner_agent import PlannerAgent
from recommender_agent import RecommenderAgent
from context_assembler import ContextAssembler
from retriever import VINERetriever
from raptor import Raptor
import re
import threading

# Global mutex protecting all FAISS / embedding calls.
# FAISS + HuggingFace BGE are NOT thread-safe on Apple Silicon (MKL segfaults).
# All threads that call raptor_tree.retrieve_collapsed() must acquire this lock first.
_EMBED_LOCK = threading.Lock()

logger = logging.getLogger(__name__)

class ProactivePipeline:
    """
    Automated execution of the VINE-Agent. 
    Unlike the conversational chatbot, this skips query rewrite and chat history,
    focusing purely on formulating a plan and executing the recommender.
    """

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
        
        self.sensor_context_fn = sensor_context_fn
        self.drone_blocks = drone_blocks or []
        self.last_flight_date = last_flight_date
        self.raptor_top_k = raptor_top_k

    def generate_recommendation_for_alert(self, alert: Alert) -> Alert:
        """
        Main entry point for autonomous reasoning.
        Enriches an Alert object with a specific recommendation and reasoning.
        """
        logger.info(f"[PROACTIVE PIPELINE] intercepting {alert.severity} alert on Block {alert.block}")
        logger.info(f"[PROACTIVE PIPELINE] Alert Details: {alert.alert_type} | Metric: {alert.metric_name}={alert.metric_value} (Threshold: {alert.threshold_value})")
        
        # 1. Construct the autonomous prompt
        auto_query = (
            f"[AUTOMATED {alert.severity} TRIGGER] Block {alert.block} ({alert.variety}) | "
            f"Type: {alert.alert_type} | Metric: {alert.metric_name} is {alert.metric_value} "
            f"(threshold: {alert.threshold_value}).\n\n"
            f"Assess the immediate risk and provide a specific, actionable recommendation "
            f"including quantity and timing based on historical thresholds and current sensor state."
        )

        # 2. Fetch Live Modalities (Sensors, Drone)
        sensor_ctx = self.sensor_context_fn() if self.sensor_context_fn else f"SENSOR: {alert.metric_name}={alert.metric_value}"
        
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

        # 4. Planner Agent
        logger.info(f"[PROACTIVE PIPELINE] [{alert.alert_id}] Calling Planner...")
        logger.info(f"[PROACTIVE PIPELINE] [{alert.alert_id}] Auto Query to Planner: {auto_query[:100]}...")
        planner_res = self.planner.generate_plan(auto_query, data_availability=data_avail)
        plan_string = planner_res["plan_string"]
        priority = planner_res["priority"]
        logger.info(f"[PROACTIVE PIPELINE] [{alert.alert_id}] Planner finished. Priority: {priority.primary}. Executing Plan...")
        logger.debug(f"[PROACTIVE PIPELINE] [{alert.alert_id}] Plan generated:\n{plan_string}")

        # 5. Execute Plan (Worker Node)
        evidence = {}
        logger.info(f"[{alert.alert_id}] Parsing plan steps...")
        
        for line in plan_string.split("\n"):
            # Safe regex that avoids catastrophic backtracking
            match = re.search(r"(#E\d+)\s*=\s*([A-Za-z0-9_]+)\s*\[([^\]]+)\]", line)
            if not match:
                continue
                
            step_id, tool, tool_input = match.groups()
            description = line.split(step_id)[0].replace("Plan:", "").strip()
            
            resolved_input = tool_input
            for prev_id, prev_text in evidence.items():
                if prev_id in resolved_input:
                    resolved_input = resolved_input.replace(prev_id, prev_text)

            logger.info(f"[PROACTIVE PIPELINE] [{alert.alert_id}] Executing step {step_id}: {tool}...")
            logger.debug(f"[PROACTIVE PIPELINE] [{alert.alert_id}] Tool Input: {resolved_input[:100]}")
            if tool == "Raptor":
                with _EMBED_LOCK:
                    nodes = self.raptor_tree.retrieve_collapsed(resolved_input, top_k=self.raptor_top_k)
                context_text = "\n".join([f"- {n['text']}" for n in nodes])
                evidence[step_id] = context_text
            elif tool == "LLM":
                msg = f"{description}\nContext:\n{resolved_input}"
                response = self.llm.invoke(msg)
                evidence[step_id] = response.content if hasattr(response, "content") else str(response)
            else:
                evidence[step_id] = "No evidence."
            logger.info(f"[{alert.alert_id}] Step {step_id} ({tool}) finished.")

        # 6. Reconstruct Raptor Flat Evidence for Assembler
        logger.info(f"[{alert.alert_id}] Reconstructing RAPTOR evidence...")
        raptor_evidence_parts = []
        for line in plan_string.split("\n"):
            match = re.search(r"(#E\d+)\s*=", line)
            if match:
                step_id = match.group(1)
                if step_id in evidence:
                    raptor_evidence_parts.append(f"{line}\n→ {evidence[step_id]}")
        raptor_evidence = "\n---\n".join(raptor_evidence_parts)

        # 7. Assemble Multi-Modal Context
        logger.info(f"[PROACTIVE PIPELINE] [{alert.alert_id}] Assembling multimodal context...")
        logger.debug(f"[PROACTIVE PIPELINE] [{alert.alert_id}] RAPTOR Evidence Size: {len(raptor_evidence)} chars")
        logger.debug(f"[PROACTIVE PIPELINE] [{alert.alert_id}] Sensor Block: {sensor_ctx}")
        assembled_context = self.context_assembler.assemble(
            raptor_evidence=raptor_evidence,
            sensor_block=sensor_ctx,
            drone_block=drone_ctx,
            priority=priority,
            query_time=datetime.now(),
            last_flight_date=self.last_flight_date
        )

        # 8. Recommender Agent
        logger.info(f"[PROACTIVE PIPELINE] [{alert.alert_id}] Calling Recommender Agent...")
        logger.debug(f"[PROACTIVE PIPELINE] [{alert.alert_id}] Assembled Context Size: {len(assembled_context)} chars")
        recommendation = self.recommender.generate_recommendation(
            input_question=auto_query,
            plan_string=plan_string,
            step_evidence=evidence,
            assembled_context=assembled_context,
            summary="Autonomous trigger execution."
        )
        logger.info(f"[PROACTIVE PIPELINE] [{alert.alert_id}] Recommender finished. Setting alert attributes...")
        logger.debug(f"[PROACTIVE PIPELINE] [{alert.alert_id}] Generated Recommendation:\n{recommendation}")

        # Update the alert
        alert.recommendation = recommendation

        alert.reasoning = f"Prioritization: {priority.primary} -> {priority.secondary}\n" + (priority.reasoning or "")
        return alert

if __name__ == "__main__":
    from main import get_llm, get_embed_model
    from dotenv import load_dotenv
    import time
    
    load_dotenv()
    logging.basicConfig(level=logging.INFO)
    
    llm = get_llm("groq", "llama-3.1-8b-instant")
    embed = get_embed_model("cpu")
    
    class MockRaptor:
        def retrieve_collapsed(self, q, top_k): return [{"text": "Pinot Noir critical threshold for irrigation is 20%. Apply 0.5 inches."}]
    
    pipeline = ProactivePipeline(llm=llm, raptor_tree=MockRaptor(), embed_model=embed, sensor_context_fn=lambda: "SENSOR: VWC=18.5%, Temp=96F")
    test_alert = Alert(alert_id="test-val-01", rule_id="test", block="A", variety="Pinot Noir", severity="CRITICAL", alert_type="IRRIGATION_EMERGENCY", description="Test", metric_name="vwc_min", metric_value=18.5, threshold_value=20.0, operator="lt", cooldown_hours=4)
    
    print("\n--- Running Proactive Pipeline ---")
    t0 = time.time()
    enriched = pipeline.generate_recommendation_for_alert(test_alert)
    print(f"\n[DONE in {time.time()-t0:.2f}s]")
    print(f"Recommendation:\n{enriched.recommendation}")
