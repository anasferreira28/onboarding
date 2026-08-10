"""
Simple agent script.
"""

from agents import Agent, Runner
from pydantic import BaseModel
from typing import Literal
from local_models import ConsoleTraceProcessor, gemma_model

class PubMedAnswer(BaseModel):
    question: str
    decision: Literal["yes", "no", "maybe"]
    rationale: str
    confidence: int # 0-100

qa_agent = Agent(
    name = "PubMedQA Answering Agent",
    instructions = (
        "You answer biomedical research questions using ONLY the abstract of a PubMed article provided to you. "
        "Do not use outside knowledge."
        "Give a YES/NO/MAYBE decision, a one-sentence rationale grounded in the context, and a 0-100 confidence score for your answer."
    ),
    model = gemma_model,
    output_type = PubMedAnswer,
)

# Example usage
from datasets import load_dataset
ds = load_dataset("qiaojin/PubMedQA", "pqa_labeled")["train"]
example = ds[0]
prompt = f"Question: {example['question']}\n\nContext: {''.join(example['context']['contexts'])}"
result = Runner.run_sync(qa_agent, prompt)
print(result.final_output)
print("Reference:", example["final_decision"])