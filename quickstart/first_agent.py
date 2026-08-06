"""
Create your first agent - OpenAI SDK's Quickstart

Use a plain Agent plus Runner when the task mainly lives
in prompts, tools and conversation state.
"""

import asyncio
from agents import Agent, Runner
from agents.decorators import tool 

@tool
def history_fun_fact() -> str:
    """Return a short history fun fact."""
    return "Did you know that the Roman Empire fell in 476 AD?"

history_tutor_agent = Agent(
    name = "History Tutor",
    instructions = "You answer history questions clearly and concisely.",
    tools = [history_fun_fact],
    handoff_description = "Specialist agent for answering history questions. Use the history_fun_fact tool when appropriate.",
    )
    #instructions = "You are a history tutor. You will answer questions about historical events, figures, and timelines. Provide detailed explanations and context for your answers.",

math_tutor_agent = Agent(
    name = "Math Tutor",
    instructions = "You answer math questions clearly and concisely.",
    tools = [],
    handoff_description = "Specialist agent for answering math questions.",
    )

# Define a handoff agent that can route questions to the appropriate specialist agent
triage_agent = Agent(
    name = "Triage Agent",
    instructions = "Route each homework question to the right specialist.",
    handoffs = [history_tutor_agent, math_tutor_agent],
)

async def main():
    result = await Runner.run(
        triage_agent, 
        "Who was the first emperor of Rome and when did the Roman Empire fall?",
        )
    print(result.final_output)
    print(f"Answered by: {result.final_agent.name}")

if __name__ == "__main__":
    asyncio.run(main())