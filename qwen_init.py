"""
Loads the OpenRouter API key from the .env file and initializes the OpenAI client for the Qwen model (since it cannot run locally). 
The ask_qwen function sends a prompt to the Qwen model and returns the response. 
The example at the end demonstrates how to use the function to get an explanation of breast cancer subtypes and their treatments.
"""

from dotenv import load_dotenv
load_dotenv() # reads .env from the current folder into the environment

from openai import OpenAI
import os

qwen_client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.environ["OPENROUTER_API_KEY"],) #call to OpenRouter's API with your API key

def ask_qwen(prompt: str, model: str = "qwen/qwen3.6-27b", thinking: bool = True) -> str:
    resp = qwen_client.chat.completions.create(model=model, messages=[{"role": "user", "content": prompt}],
                                               max_tokens=3000,
                                               extra_body={"reasoning": {"enabled": thinking}}, # OpenRouter's toggle for reasoning-capable models
)
    return resp.choices[0].message.content


print(ask_qwen("Explain BI-RADS categories briefly.", thinking=False))
