"""Instruction proposer for the native GEPA variant.

Adapts the instruction-proposal flow of ``ProposeNewInstructionModule`` in
``gepa_artifact/gepa/instruction_proposal.py``, calling LangChain's
``BaseChatModel.ainvoke`` with the current instruction and per-example evaluation feedback.
"""

import logging

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

logger = logging.getLogger(__name__)


PROPOSER_SYSTEM = """You are an expert prompt engineer. Your task is to improve the
**system instruction** of a tool-using LLM agent so that it scores higher on a
benchmark.

You will receive:
1. The CURRENT instruction.
2. A handful of EXAMPLE RUNS where the agent was scored. Each example shows the
   inputs, the agent's output, the aggregate score, and per-evaluator scores
   with diagnostic details (often including expected vs actual values, tool
   call counts, judge rationales, etc.).

Your job: write a NEW instruction that fixes the failure modes you observe.

Rules:
- Output ONLY the new instruction text. No preamble, no markdown fences,
  no commentary.
- Preserve any tool names mentioned in the instruction EXACTLY (case-sensitive
  function names the agent must call).
- Preserve any output field names mentioned in the instruction EXACTLY
  (case-sensitive Pydantic schema field names).
- Keep the same task definition; do not invent new tasks or remove
  capabilities.
- Be concrete: if examples show a specific failure (wrong tool, missing
  field, miscounted output), address that failure explicitly.
- Aim for under ~400 words unless the original is longer.
"""


PROPOSER_USER_TEMPLATE = """CURRENT INSTRUCTION:
\"\"\"
{current_instruction}
\"\"\"

EXAMPLE RUNS:
{examples_block}

Now write the improved instruction (text only, no fences):"""


async def propose_new_instruction(
    proposer_lm: BaseChatModel,
    current_instruction: str,
    feedback_blocks: list[str],
) -> str:
    examples_block = "\n\n---\n\n".join(feedback_blocks)
    msgs = [
        SystemMessage(content=PROPOSER_SYSTEM),
        HumanMessage(
            content=PROPOSER_USER_TEMPLATE.format(
                current_instruction=current_instruction,
                examples_block=examples_block,
            )
        ),
    ]
    response = await proposer_lm.ainvoke(msgs)
    text = response.content if hasattr(response, "content") else str(response)
    if isinstance(text, list):
        text = "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in text)
    return text.strip()
