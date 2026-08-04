"""
Loads the OpenRouter API key from the .env file and initializes the OpenAI client for the Qwen model (since it cannot run locally). 
The ask_qwen function sends a prompt to the Qwen model and returns the response. 
The example at the end demonstrates how to use the function to get an explanation of breast cancer subtypes and their treatments.
"""

from dotenv import load_dotenv
from pypdf import PdfReader
load_dotenv() # reads .env from the current folder into the environment

from openai import OpenAI
import os
from pdf_reader import paper_text

qwen_client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.environ["OPENROUTER_API_KEY"],) #call to OpenRouter's API with your API key

def ask_qwen(prompt: str, model: str = "qwen/qwen3.6-27b", thinking: bool = True) -> str:
    resp = qwen_client.chat.completions.create(model=model, messages=[{"role": "user", "content": prompt}],
                                               max_tokens=3000,
                                               extra_body={"reasoning": {"enabled": thinking}}, # OpenRouter's toggle for reasoning-capable models
)
    return resp.choices[0].message.content


#print(ask_qwen("Extract the following fields as JSON from this pathology snippet: {tumor_size_cm, histologic_grade, ER_status, PR_status, HER2_status, lymph_nodes_positive}. Snippet: 'Invasive ductal carcinoma, 2.3 cm, Nottingham grade 2. ER positive (90%), PR positive (60%), HER2 negative (IHC 1+). 1 of 14 sentinel lymph nodes positive.'", thinking=True))

### FOR EXERCISE 2.4

summarize_prompt = ("Summarize this paper's methods, key result, and one limitation in under 150 words.\n\n" + paper_text)

print(ask_qwen(summarize_prompt))

