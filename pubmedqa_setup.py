"""
Import the PubMedQA dataset from Hugging Face for Agent building.
Includes an example to print the first question and its final decision.
"""

from datasets import load_dataset

pubmedqa = load_dataset("qiaojin/PubMedQA", "pqa_labeled")["train"]

print(pubmedqa[0]["question"])
print(pubmedqa[0]["final_decision"])