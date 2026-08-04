"""
Gemma initialization for Exercise 2.4
"""

from openai import OpenAI
from pdf_reader import paper_text

gemma_client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")

def ask_gemma(prompt: str, model: str = "gemma4:e2b") -> str:
    resp = gemma_client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.choices[0].message.content

summarize_prompt = ("Summarize this paper's methods, key result, and one limitation in under 150 words.\n\n" + paper_text)

print(ask_gemma(summarize_prompt))
