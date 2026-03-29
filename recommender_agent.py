import logging
import re
from typing import Dict, List, Optional
from langchain_core.messages import AIMessage, HumanMessage
from prompts import SOLVER_PROMPT

logger = logging.getLogger(__name__)


class RecommenderAgent:

    def __init__(self, llm):
        """Pass a LangChain LLM instance (e.g., vLLM or ChatGroq)."""
        self.llm = llm

    def generate_recommendation(
        self,
        input_question: str,
        plan_string: str,
        step_evidence: Dict[str, str],
        assembled_context: str,
        summary: str = "",
    ) -> str:
     
        logger.info("Recommender Agent synthesizing final answer...")

        # Reconstruct plan with filled evidence (for plan_evidence field)
        full_context = ""
        regex = r"(#E\d+)\s*=\s*(\w+)\s*\[([^\]]+)\]"
        for line in plan_string.split("\n"):
            if "=" in line:
                match = re.search(regex, line)
                if match:
                    step_id = match.group(1)
                    result_text = step_evidence.get(step_id, "No evidence found or tool failed.")
                    # Inject the actual result text
                    full_context += f"{line}\nRESULT: {result_text}\n---\n"
                else:
                    full_context += f"{line}\n"
            else:
                full_context += f"{line}\n"

        # Format the specific solver prompt
        prompt = SOLVER_PROMPT.format_messages(
            question=input_question,
            summary=summary,
            assembled_context=assembled_context,
            plan_evidence=full_context,
        )

        response = self.llm.invoke(prompt)
        answer_text = response.content if hasattr(response, "content") else str(response)

        logger.info(f"Recommender final output: {answer_text[:100]}...")
        return answer_text
