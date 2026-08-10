"""
Simple agent script.
"""

# Model Startup
from agents import Agent, Runner, function_tool
from pydantic import BaseModel
from typing import Literal
from local_models import gemma_model

# Dataset Startup
from datasets import load_dataset
ds = load_dataset("qiaojin/PubMedQA", "pqa_labeled")["train"]
corpus = {example["pubid"]: " ".join(example["context"]["contexts"]) for example in ds}

class PubMedAnswer(BaseModel):
    question: str
    decision: Literal["yes", "no", "maybe"]
    rationale: str
    confidence: int # 0-100

@function_tool
def fetch_abstract(pubid: str) -> str:
    """
    Fetch the PubMed abstract context for a given article id.
    """
    return corpus.get(pubid, "Not found.")


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

qa_agent_tool = Agent(
    name = "PubMedQA Answering Agent with Abstract Fetching",
    instructions = (
        "You answer biomedical yes/no/maybe questions from the PubMedQA dataset."
        "You are given a pubid."
        "Call fetch_abstract(pubid) to get the relevant context before responding."
        "Only use retrieved evidence, never outside knowledge."
    ),
    model = gemma_model,
    tools = [fetch_abstract],
    output_type = PubMedAnswer,
)

# Example 1
example = ds[0]
# prompt = f"Question: {example['question']}\n\nContext: {''.join(example['context']['contexts'])}"
# result = Runner.run_sync(qa_agent, prompt)
# print(result.final_output)
# print("Reference:", example["final_decision"])

# [TRACE] Agent workflow finished
# question='Do mitochondria play a role in remodelling lace plant leaves during programmed cell death?' 
# decision='yes' 
# rationale='The study investigated mitochondrial dynamics during PCD progression and examined the 
# effect of modulating mitochondrial permeability transition pore formation on leaf perforations.' 
# confidence=100
# Reference: yes


# Example 2
result = Runner.run_sync(qa_agent_tool, f"pubid: {example['pubid']}\nQuestions: {example['question']}")
print(result.final_output)