"""
Load PubMedQA dataset and inspect one example.
Reference Guide: https://huggingface.co/datasets/qiaojin/PubMedQA 
"""

from datasets import load_dataset

ds = load_dataset("qiaojin/PubMedQA", "pqa_labeled")["train"]

# Inspect one example
example = ds[0]
print("Question:", example["question"])
print("Context:", example["context"]["contexts"]) # list of abstract sections
print("Long Answer:", example["long_answer"]) # The abstract's conclusions
print("Final Decision:", example["final_decision"]) # Yes / No / Maybe