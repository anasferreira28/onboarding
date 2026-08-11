"""
Exercise 2.5: LiteLLM call for 2 local Gemma models
"""

import litellm

response_gemma = litellm.completion(
    model = "ollama_chat/gemma4:e2b", # the 'ollama_chat' applies Ollama's chat template correctly for instruction models
    messages = [{"role": "user", "content": "Explain BI-RADSA categories briefly."}],
    api_base="https://localhost:11434" # Ollama's fixed default address
)
print(response_gemma.choices[0].message.content)

response_qwen = litellm.completion(
    model = "ollama_chat/qwen3:8b", # the 'ollama_chat' applies Ollama's chat template correctly for instruction models
    messages = [{"role": "user", "content": "Explain BI-RADSA categories briefly."}],
    api_base="https://localhost:11434" 
)
print(response_qwen.choices[0].message.content)


