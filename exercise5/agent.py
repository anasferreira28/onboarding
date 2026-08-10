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
corpus = {str(example["pubid"]): " ".join(example["context"]["contexts"]) for example in ds} # corpus is originally int, so must be converted to string to it matches the requirements of the prompt and function call

# Evaluation Startup
import random
sample = random.sample(list(ds), 30) # 30 examples from the dataset
correct = 0

class PubMedAnswer(BaseModel):
    question: str
    decision: Literal["yes", "no", "maybe"]
    rationale: str
    confidence: Literal["low", "medium", "high"]

@function_tool
def fetch_abstract(pubid: str) -> str:
    """
    Fetch the PubMed abstract context for a given article id.
    """
    return corpus.get(pubid, "Not found.")

# Base agent 
qa_agent = Agent(
    name = "PubMedQA Answering Agent",
    instructions = (
        "You answer biomedical research questions using ONLY the abstract of a PubMed article provided to you. "
        "Do not use outside knowledge."
        "Give a YES/NO/MAYBE decision, a one-sentence rationale grounded in the context, and a confidence level (low, medium, or high) for your answer."
    ),
    model = gemma_model,
    output_type = PubMedAnswer,
)

# Agent with tool call
qa_agent_tool = Agent(
    name = "PubMedQA Answering Agent with Abstract Fetching",
    instructions = (
        "You answer biomedical yes/no/maybe questions from the PubMedQA dataset."
        "You are given a pubid."
        "Call fetch_abstract(pubid) to get the relevant context before responding."
        "Only use retrieved evidence, never outside knowledge."
        "Always quote the key sentence(s) from the abstract that support your answer BEFORE responding."
    ),
    model = gemma_model,
    tools = [fetch_abstract],
    output_type = PubMedAnswer,
)

# Agent with handoffs
specialist_agent = Agent(
    name = "PubMedQA Specialist Agent",
    instructions = (
        "You are a careful biomedical reasoner."
        "Call fetch_abstract for the given pubid and reason"
        "carefully about the quantitative results in the abstract."
        "Then, give a YES/NO/MAYBE decision, a one-sentence rationale grounded in the context, and a confidence level (low, medium, or high) for your answer."
    ),
    model = gemma_model,
    tools = [fetch_abstract],
    output_type = PubMedAnswer,
)

triage_agent = Agent(
    name = "PubMedQA Triage Agent",
    instructions = (
        "Read the questions." 
        "if it requires careful reasoning over numeric or comparative results" 
        "hand off to the Specialist Agent." 
        "Otherwise, answer the question yourself with fetch_abstract." 

    ),
    model = gemma_model,
    tools = [fetch_abstract],
    handoffs = [specialist_agent],
    output_type = PubMedAnswer,
)

# Example 1
example = ds[25]
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
# result = Runner.run_sync(qa_agent_tool, f"pubid: {example['pubid']}\nQuestions: {example['question']}")
# print(result.final_output)

# TRACING CONFIRMS FUNCTION CALL
# [SPAN START] FunctionSpanData: <agents.tracing.span_data.FunctionSpanData object at 0x00000203C9412A80>
# [SPAN END]   FunctionSpanData: <agents.tracing.span_data.FunctionSpanData object at 0x00000203C9412A80>

# First Try
# question='Do mitochondria play a role in remodelling lace plant leaves during programmed cell death?' 
# decision='maybe' 
# rationale='The provided PubMed abstract for pubid: 21645374 was not found, so I cannot determine the answer based on the retrieved evidence.' 
# confidence=0

# After Fixes (much better!!)
# question='Do mitochondria play a role in remodelling lace plant leaves during programmed cell death?' 
# decision='yes' 
# rationale='The abstract discusses elucidating the role of mitochondrial dynamics during developmentally regulated PCD in the lace plant (*A. madagascariensis*), which undergoes PCD resulting in leaf perforations. 
# The study examined mitochondrial dynamics and their relationship with the stages of PCD, directly implicating mitochondria in this process.' 
# confidence='high'

# Example 3
# result = Runner.run_sync(triage_agent, f"pubid: {example['pubid']}\nQuestions: {example['question']}")
# print(result.final_output)

# We verify handoff through tracing 
#   [SPAN START] HandoffSpanData: <agents.tracing.span_data.HandoffSpanData object at 0x000002DD9C770110>
#   [SPAN END]   HandoffSpanData: <agents.tracing.span_data.HandoffSpanData object at 0x000002DD9C770110>

# question='Amblyopia: is visual loss permanent?' 
# decision='maybe' 
# rationale='The abstract discusses visual acuity improvement in cases related to macular degenerationand notes that improvement, 
# when it occurs, generally remained stable over the follow-up period, but it does not explicitly state whether amblyopia leads to permanent visual loss.' 
# confidence='low'

# Example 4

# Model evaluation
for ex in sample:
    result = Runner.run_sync(triage_agent, f"pubid: {ex['pubid']}\nQuestion: {ex['question']}")
    pred = result.final_output.decision
    correct += (pred == ex['final_decision'])
    if pred != ex['final_decision']:
        print("MISMATCH:", ex['question'])
        print("Predicted:", pred, "| Reference:", ex['final_decision'])
        print("Long Answer:", ex['long_answer'][:200]) # long answer up to 200 characters?
print(f"Accuracy:: {correct}/{len(sample)} = {correct/len(sample):.1%}")