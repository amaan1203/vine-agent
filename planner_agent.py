import logging
from typing import Dict, List, Optional
from langchain_core.messages import HumanMessage, SystemMessage
from prompts import PLANNER_PROMPT
from context_assembler import ContextPriority

logger = logging.getLogger(__name__)


class PlannerAgent:

    def __init__(self, llm):
        """Pass a LangChain LLM instance (e.g., ChatGroq)."""
        self.llm = llm

    def generate_plan(self, input_text: str, data_availability: str = "") -> Dict:
        """
        Takes a human query or an automated alert string.
        Returns {"plan_string": str, "priority": ContextPriority}.
        """
        logger.info("Planner Agent evaluating input...")
        if not data_availability:
            data_availability = (
                "Live IoT Sensor Data: AVAILABLE\n"
                "Drone Multispectral Imagery: AVAILABLE\n"
                "Agricultural Knowledge Base (RAPTOR): AVAILABLE"
            )

        prompt = PLANNER_PROMPT.format_messages(
            question=input_text,
            summary="Autonomous execution mode. No prior humans conversation.",
            data_availability=data_availability,
        )
        
        result_msg = self.llm.invoke(prompt)
        plan_string = result_msg.content if hasattr(result_msg, "content") else str(result_msg)

        logger.debug(f"Planner raw output:\n{plan_string[:400]}...")

        # Extract Context Assembly Blueprint (CAB)
        priority = ContextPriority.from_plan_string(plan_string)
        
        # Fallback if the Planner failed to output strict CAB (rare but possible)
        if priority.primary == "unknown":
            logger.warning("Planner failed to output ContextPriority block. Using heuristic fallback.")
            from context_assembler import infer_priority_from_query
            priority = infer_priority_from_query(input_text)
            
        logger.info(f"Planner Context Blueprint routing: {priority.primary.upper()} → {priority.secondary.upper()} → {priority.tertiary.upper()}")

        return {
            "plan_string": plan_string,
            "priority": priority
        }


if __name__ == "__main__":
    from argparse import ArgumentParser
    from main import get_llm
    from dotenv import load_dotenv

    load_dotenv()
    logging.basicConfig(level=logging.INFO)

    llm = get_llm("groq", "llama-3.1-8b-instant")
    planner = PlannerAgent(llm)

    # Test 1: Automated Trigger
    alert_input = (
        "[AUTOMATED TRIGGER] Block A (Chardonnay) | CRITICAL alert | "
        "Type: IRRIGATION_EMERGENCY | Metric: vwc_min drops to 18.5 (threshold: 20.0). "
        "Proactively generating recommendation."
    )
    print("\n--- Testing Planner on Automated Trigger ---")
    res1 = planner.generate_plan(alert_input)
    print(f"Routing Map: {res1['priority']}")

    # Test 2: User Query
    user_input = "Can you check the drone NDVI maps to see where I should apply nitrogen?"
    print("\n--- Testing Planner on User Query ---")
    res2 = planner.generate_plan(user_input)
    print(f"Routing Map: {res2['priority']}")
