from dotenv import load_dotenv
load_dotenv() # reads .env from the current folder into the environment

from openai import OpenAI
import os

qwen_client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.environ["OPENROUTER_API_KEY"],) #call to OpenRouter's API with your API key

def ask_qwen(prompt: str, model: str = "qwen/qwen3.6-27b", thinking: bool = True) -> str:
    resp = qwen_client.chat.completions.create(model=model, messages=[{"role": "user", "content": prompt}],
                                               extra_body={"reasoning": {"enabled": thinking}}, # OpenRouter's toggle for reasoning-capable models
)
    return resp.choices[0].message.content


print(ask_qwen("Hello, introduce yourself in one sentence."))
