"""
Simple agent script.
"""

# Model Startup
from agents import Agent, Runner, function_tool, ModelSettings
from pydantic import BaseModel
from typing import Literal
from local_models import gemma_model

# Stdlib for evaluation/logging
import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

from agents.exceptions import ModelBehaviorError
from pydantic import ValidationError

# Dataset Startup
from datasets import load_dataset
ds = load_dataset("qiaojin/PubMedQA", "pqa_labeled")["train"]
corpus = {
    str(example["pubid"]): "\n".join(
        f"{label}: {text}" for label, text in zip(example["context"]["labels"], example["context"]["contexts"])
    )
    for example in ds
} # corpus is originally int, so must be converted to string to it matches the requirements of the prompt and function call
# NOTE: kept section labels (BACKGROUND/METHODS/RESULTS/CONCLUSIONS) instead of flattening with " ".join(...),
# so the model can locate the outcome-bearing sentence instead of getting an undifferentiated blob.

# Evaluation Startup
EVAL_SEED = 42
EVAL_N = 50
sample = ds.shuffle(seed=EVAL_SEED).select(range(EVAL_N)) # deterministic sample so accuracy is reproducible across runs

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
        "Do not use outside knowledge. "
        "Give a YES/NO/MAYBE decision, a one-sentence rationale grounded in the context, and a confidence level (low, medium, or high) for your answer. "
        "Base the decision on the explicit direction of the finding stated in the RESULTS/CONCLUSIONS section, even if other parts of the abstract contain caveats or hedging language. "
        "Only answer MAYBE if the abstract itself reports a mixed, inconclusive, or conflicting result -- never because you personally feel uncertain. "
        "Example: if RESULTS states 'frequent consumption was associated with a dose-dependent increase in symptoms', and the question asks whether X is a risk factor, the decision is YES, even if CONCLUSIONS also lists limitations of the study."
    ),
    model = gemma_model,
    model_settings = ModelSettings(temperature=0.1, frequency_penalty=0.4), # frequency penalty discourages word repetition by penalizing tokens based on how many times they have already appeared in the generated output; positive values force a varied vocabulary, while negative values encourage repetition.
    output_type = PubMedAnswer,
)

# Agent with tool call
qa_agent_tool = Agent(
    name = "PubMedQA Answering Agent with Abstract Fetching",
    instructions = (
        "You answer biomedical yes/no/maybe questions from the PubMedQA dataset. "
        "You are given a pubid. "
        "Call fetch_abstract(pubid) to get the relevant context before responding. "
        "Only use retrieved evidence, never outside knowledge. "
        "Always quote the key sentence(s) from the abstract that support your answer BEFORE responding. "
        "Base the decision on the explicit direction of the finding stated in the RESULTS/CONCLUSIONS section, even if other parts of the abstract contain caveats or hedging language. "
        "Only answer MAYBE if the abstract itself reports a mixed, inconclusive, or conflicting result -- never because you personally feel uncertain. "
        "Example: if RESULTS states 'frequent consumption was associated with a dose-dependent increase in symptoms', and the question asks whether X is a risk factor, the decision is YES, even if CONCLUSIONS also lists limitations of the study."
    ),
    model = gemma_model,
    model_settings = ModelSettings(temperature=0.1, frequency_penalty=0.4),
    tools = [fetch_abstract],
    output_type = PubMedAnswer,
)

# Agent with handoffs
specialist_agent = Agent(
    name = "PubMedQA Specialist Agent",
    instructions = (
        "You are a careful biomedical reasoner who handles questions requiring numeric or comparative reasoning. "
        "Call fetch_abstract for the given pubid. "
        "Follow these steps before deciding: "
        "1) Identify the two things being compared, or the specific outcome being measured. "
        "2) Find the sentence(s) in the RESULTS/CONCLUSIONS section stating which one the results favor, and by how much. "
        "3) State that direction explicitly in your rationale before giving a decision. "
        "Only answer MAYBE if the numeric results themselves are mixed or inconclusive -- never because the comparison is merely complex. "
        "Report numbers and comparisons in plain text only (e.g. 'increased by 15%', 'group A had 6 vs group B had 3') -- never use LaTeX, markdown math, or symbols like \\text{} or $...$. " # avoid the degenerate JSON error
        "Then give a YES/NO/MAYBE decision, a one-sentence rationale grounded in the context, and a confidence level (low, medium, or high) for your answer."
    ),
    model = gemma_model,
    model_settings = ModelSettings(temperature=0.1, frequency_penalty=0.4),
    tools = [fetch_abstract],
    output_type = PubMedAnswer,
)

triage_agent = Agent(
    name = "PubMedQA Triage Agent",
    instructions = (
        "Read the question. "
        "Hand off to the Specialist Agent if the question involves comparative or quantitative language, e.g. "
        "'better', 'compared to', 'predictor', 'risk factor', 'associated with', 'increase', 'decrease', or mentions numbers/percentages/statistics. "
        "Otherwise, answer the question yourself: call fetch_abstract(pubid) and give a YES/NO/MAYBE decision, a one-sentence rationale grounded in the context, and a confidence level (low, medium, or high)."
    ),
    model = gemma_model,
    model_settings = ModelSettings(temperature=0.1, frequency_penalty=0.4),
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
def run_eval(agent, sample, name):
    correct = 0
    attempted = 0
    confusion = defaultdict(lambda: defaultdict(int)) # confusion[reference][predicted] = count
    mismatches = []
    errors = []
    for ex in sample:
        prompt = f"pubid: {ex['pubid']}\nQuestion: {ex['question']}"
        result = None
        last_error = None
        for attempt in range(2): # one retry -- covers a one-off decoding glitch (e.g. repetition loop breaking the output JSON)
            try:
                result = Runner.run_sync(agent, prompt)
                break
            except (ModelBehaviorError, ValidationError) as e:
                last_error = e
                print(f"WARNING [{name}] generation failed for pubid {ex['pubid']} (attempt {attempt + 1}/2): {e}")
        if result is None:
            errors.append({"question": ex['question'], "pubid": ex['pubid'], "error": str(last_error)})
            continue
        attempted += 1
        pred = result.final_output.decision
        actual = ex['final_decision']
        confusion[actual][pred] += 1
        correct += (pred == actual)
        if pred != actual:
            mismatches.append({"question": ex['question'], "predicted": pred, "reference": actual, "long_answer": ex['long_answer'][:200]})
            print("MISMATCH:", ex['question'])
            print("Predicted:", pred, "| Reference:", actual)
            print("Long Answer:", ex['long_answer'][:200]) # long answer up to 200 characters?
    accuracy = correct / attempted if attempted else 0.0
    labels = ["yes", "no", "maybe"]
    print(f"\n[{name}] Accuracy: {correct}/{attempted} = {accuracy:.1%} ({len(errors)} skipped after failing generation)")
    print(f"[{name}] Confusion matrix (rows=reference, cols=predicted):")
    for actual in labels:
        row = " ".join(f"{label}={confusion[actual][label]}" for label in labels)
        print(f"  {actual:>5}: {row}")
    return {"name": name, "accuracy": accuracy, "attempted": attempted, "confusion": {a: dict(confusion[a]) for a in labels}, "mismatches": mismatches, "errors": errors}

label_counts = Counter(ex['final_decision'] for ex in sample)
majority_label, majority_count = label_counts.most_common(1)[0]
majority_baseline = majority_count / len(sample)
print(f"Majority-class baseline (\"always predict {majority_label}\"): {majority_count}/{len(sample)} = {majority_baseline:.1%}")

tool_results = run_eval(qa_agent_tool, sample, "qa_agent_tool")
triage_results = run_eval(triage_agent, sample, "triage_agent")

log_entry = {
    "timestamp": datetime.now().isoformat(),
    "seed": EVAL_SEED,
    "n": EVAL_N,
    "majority_baseline": majority_baseline,
    "runs": [tool_results, triage_results],
}
with open(Path(__file__).resolve().parent / "eval_log.jsonl", "a") as f:
    f.write(json.dumps(log_entry) + "\n")

# Three mismatch examples
# MISMATCH: The effective orifice area/patient aortic annulus area ratio: a better way to compare different bioprostheses?
# Predicted: no | Reference: yes
# Long Answer: Comparisons of absolute EOA values grouped by the manufacturers' valve sizes are misleading because of specific 
# differences in geometric dimensions. The EOA:patient aortic annulus area ratio provides 

# MISMATCH: PSA repeatedly fluctuating levels are reassuring enough to avoid biopsy?
# Predicted: maybe | Reference: no
# Long Answer: Our study demonstrates no difference in PC detection rate at repeat biopsy between patients with flu or si-PSA levels. 
# PSA Slope, PSAV and PSADT were not found helpful tools in cancer detection. 
# Agent reasoning doesn't make much sense!

# MISMATCH: Fast foods - are they a risk factor for asthma?
# Predicted: no | Reference: yes
# Long Answer: Frequent consumption of hamburgers showed a dose-dependent association with asthma symptoms, 
# and frequent takeaway consumption showed a similar association with BHR.
# Agent reasoning is also wrong.

# MISMATCH: Is year of radical prostatectomy a predictor of outcome in prostate cancer?
# Predicted: no | Reference: yes
# Long Answer: When controlling for preoperative features, the year in which RP was performed is a predictor of outcome on multivariate analysis. 
# This effect could not be explained by stage migration.

# MISMATCH: Are pediatric concussion patients compliant with discharge instructions?
# Predicted: maybe | Reference: yes
# Long Answer: Pediatric patients discharged from the ED are **mostly** compliant with concussion instructions. 
# However, a significant number of patients RTP on the day of injury, while experiencing symptoms or without 
# Agent reasoning makes (some) sense!

# ACCURACY: 


## CHECK PROCESS:
# Get-Content "C:\Users\ana_s\onboarding\exercise5\agent_run.log" -Tail 50
# Get-Process -Id 28652   # confirms it's still running (errors if it finished/died)