"""
Manual, opt-in smoke test for the HER2 tool-calling agent against the real
gemma4:e2b model running locally in Ollama. Not part of the pytest suite —
run directly (`python manual_smoke_test.py`) and read the output by eye.

Uses run_her2_agent_verbose so we can see, per snippet: what fields the
live model actually extracted, what score_her2_status deterministically
computed from them, and whether that matches this snippet's expected
classification — not just the model's final paraphrased text. This is
the check the mocked pytest suite (test_agent.py) can't do, since it
assumes a well-formed tool call to begin with.

Synthetic vignettes only — never real patient data. Output carries the
same "educational use only" disclaimer as the agent itself.
"""

from agent import DISCLAIMER, HER2AgentError, run_her2_agent_verbose

SNIPPETS = [
    (
        "Clear IHC 3+, no ISH mentioned",
        "Invasive ductal carcinoma, 3.1 cm, Nottingham grade 3. ER negative, "
        "PR negative, HER2 positive by immunohistochemistry (IHC 3+). "
        "2 of 11 sentinel lymph nodes positive.",
        "Positive",
    ),
    (
        "IHC 2+ with ISH landing in equivocal group 4",
        "Invasive lobular carcinoma, 1.8 cm, Nottingham grade 2. HER2 IHC 2+, "
        "equivocal by immunohistochemistry. Reflex dual-probe ISH performed: "
        "HER2/CEP17 ratio 1.6, average HER2 copy number 5.2 signals per cell.",
        "Equivocal",
    ),
    (
        "IHC 2+ with ISH pending, no values given",
        "Invasive ductal carcinoma, 2.0 cm, Nottingham grade 2. ER positive (80%), "
        "PR positive (40%). HER2 IHC 2+. Dual-probe ISH pending.",
        "Equivocal",
    ),
    (
        "Straightforward IHC 0",
        "Invasive ductal carcinoma, 1.2 cm, Nottingham grade 1. ER positive (95%), "
        "PR positive (70%), HER2 negative by immunohistochemistry (IHC 0).",
        "Negative",
    ),
]


def main() -> None:
    for label, snippet, expected_status in SNIPPETS:
        print("=" * 80)
        print(label)
        print(f"Expected classification: {expected_status}")
        print("-" * 80)
        print(snippet)
        print("-" * 80)

        try:
            pipeline = run_her2_agent_verbose(snippet)
        except HER2AgentError as exc:
            print(f"AGENT ERROR: {exc}")
            print()
            continue

        actual_status = pipeline["tool_result"]["status"]
        verdict = "MATCH" if actual_status == expected_status else "MISMATCH"

        print(f"Extracted tool arguments: {pipeline['tool_arguments']}")
        print(f"Raw tool result: {pipeline['tool_result']}")
        print(f"{verdict}: actual={actual_status!r} expected={expected_status!r}")
        print("-" * 80)
        print(pipeline["final_text"])
        print()
        print(DISCLAIMER)
        print()


if __name__ == "__main__":
    main()
