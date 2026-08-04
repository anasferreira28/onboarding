"""
Configure the OpenAI Agents SDK to use the local Ollama endpoint for the Gemma model, 
and set up a console-based tracing processor to print trace information to the console 
instead of uploading it to OpenAI's platform (keep all information local).
"""

from agents import OpenAIChatCompletionsModel, set_trace_processors
from agents.tracing.processor_interface import TracingProcessor
from openai import AsyncOpenAI

# Same local Ollama endpoint as Setup 1.1 
ollama_client = AsyncOpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
gemma_model = OpenAIChatCompletionsModel(model="gemma4:e2b", openai_client=ollama_client) # use Gemma for the purpose of experimentation; instead we could use the default OpenAI model (e.g., gpt-4o) if we wanted to use the OpenAI API instead of Ollama.


# The SDK's default tracing uploads to platform.openai.com and needs an OpenAI key.
# This replaces that with a local console printer instead.
class ConsoleTraceProcessor(TracingProcessor):
    # Announce when a whole workflow run begins/end
    def on_trace_start(self, trace):
        print(f"[TRACE] {trace.workflow_name} started")
    def on_trace_end(self, trace):
        print(f"[TRACE] {trace.workflow_name} finished")
    # Logs each individual span (step) in the workflow, including its type and data
    def on_span_start(self, span):
        print(f" [SPAN START] {type(span.span_data).__name__}: {span.span_data}")
    def on_span_end(self, span):
        print(f" [SPAN END] {type(span.span_data).__name__}: {span.span_data}")
    # Announce when a workflow run is canceled
    def on_trace_canceled(self, trace):
        print(f"[TRACE] {trace.workflow_name} canceled")
    # For when the processor is being shut down, e.g., at the end of a run (required by the interface but not needed for this console implementation)
    def shutdown(self): 
        pass
    def force_flush(self):
        pass

set_trace_processors([ConsoleTraceProcessor()]) # Set the SDK to use the console trace processor instead of uploading traces to OpenAI's platform (default).