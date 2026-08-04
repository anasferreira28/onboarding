"""
Import the PubMedQA dataset from Hugging Face for Agent building.
Includes an example to print the first question and its final decision.
"""
# from dotenv import load_dotenv
# load_dotenv() # reads .env from the current folder into the environment - if we want to load the HF token

from datasets import load_dataset

pubmedqa = load_dataset("qiaojin/PubMedQA", "pqa_labeled")["train"]

# print(pubmedqa[0]["question"])
# print(pubmedqa[0]["final_decision"])
# # Q: Do mitochondria play a role in remodelling lace plant leaves during programmed cell death?
# # A: yes

## NOTE:
# A token is typically only needed for:
# - Gated/private datasets or models
# - Avoiding rate limits on anonymous requests
# - Pushing data back to the Hub
# In this case, the dataset is public and can be accessed without a token. 
# The token is advantageous for avoiding rate limits.
# If you have a Hugging Face account and want to use your token, you can set it as an environment variable (add in .env file):
# HF_TOKEN=your_token_here

