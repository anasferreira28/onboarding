from agents import OpenAIChatCompletionsModel
from openai import AsyncOpenAI

ollama_client = AsyncOpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
gemma_model = OpenAIChatCompletionsModel(model="gemma4:e2b", openai_client=ollama_client)

from agents import set_trace_processors
from agents.tracing.processor_interface import TracingProcessor

class ConsoleTraceProcessor(TracingProcessor):
    def on_trace_start(self, trace): print(f"[TRACE] {trace.name} started")
    def on_trace_end(self, trace): print(f"[TRACE] {trace.name} finished")
    def on_span_start(self, span): print(f"  [SPAN START] {type(span.span_data).__name__}: {span.span_data}")
    def on_span_end(self, span): print(f"  [SPAN END]   {type(span.span_data).__name__}: {span.span_data}")
    def shutdown(self): pass
    def force_flush(self): pass

set_trace_processors([ConsoleTraceProcessor()])