# Exercise 3 — HER2 Status Agentic Pipeline

## What this is

A learning exercise (part of the `onboarding` repo for Kather Lab conventions) in
building a small agentic pipeline: an LLM reads a free-text pathology report and
extracts structured biomarker fields, then plain Python code applies HER2 staging
rules to those fields.

**This is for learning and experimentation only — never clinical use.** Every
output the pipeline produces should carry a visible "for educational use only,
not a clinical decision tool" disclaimer. Use only synthetic/invented pathology
snippets, never real patient data.

The goal is to practice the full loop: plan → test → implement, for each stage
of the pipeline, not just to produce a working script.

## Architecture

Two stages, kept strictly separate:

1. **Extraction (LLM-backed)** — input: a free-text pathology report snippet.
   Output: structured fields, e.g.
   `{ihc_score: "0"|"1+"|"2+"|"3+", ish_ratio: float|None, ish_copy_number: float|None}`.
   This is the *only* stage allowed to call a model.
2. **Scoring (pure Python, no LLM)** — deterministic classification of the
   structured fields into a HER2 status, per the rules below. Must be fully
   unit-testable with no network or model call involved.

Orchestration is just thin glue: call extraction, call scoring, format the
result with the disclaimer.

**Why split it this way:** the actual staging decision has to stay deterministic
and auditable. LLM unpredictability is confined to parsing free text into
structured fields — it never gets to decide the clinical category itself.

`her2_status.py` currently holds the pre-agentic baseline (`input()`-driven,
simplified FISH handling as a positive/negative/equivocal string). Expect this
to be refactored into the scoring stage rather than preserved as-is — the FISH
input in particular should evolve from a free-text guess into real
ratio/copy-number values once we test against the rules below.

## Reference logic: ASCO/CAP 2018 HER2 testing guideline

This is the ground truth to write test cases against (simplify only if we
explicitly agree to; don't quietly drift from it).

**IHC:**
- `0` or `1+` → Negative
- `3+` → Positive
- `2+` → Equivocal, reflex to ISH/FISH

**ISH (dual-probe), five result groups:**
| Group | HER2/CEP17 ratio | Avg HER2 copies/cell | Call |
|---|---|---|---|
| 1 | ≥ 2.0 | ≥ 4.0 | Positive |
| 2 | ≥ 2.0 | < 4.0 | Needs concurrent IHC review (Positive only if IHC 3+, else Negative) |
| 3 | < 2.0 | ≥ 6.0 | Needs concurrent IHC review (Positive only if IHC 3+, else Negative; recount recommended) |
| 4 | < 2.0 | 4.0–6.0 | Needs concurrent IHC review; equivocal if IHC isn't clearly 3+ |
| 5 | < 2.0 | < 4.0 | Negative |

(Source: Wolff et al., ASCO/CAP 2018 focused update, *J Clin Oncol*.)

## Workflow for this exercise

1. **Plan** — before writing a new stage, state the function's inputs, outputs,
   and edge cases (in chat is fine; doesn't need its own file).
2. **Test** — write pytest cases *before* implementation for the scoring
   function: all 4 IHC categories, all 5 ISH groups, and invalid/missing input.
   Extraction-stage tests should use fixed/recorded model outputs rather than
   live API calls, so the suite stays deterministic and fast.
3. **Implement** — minimal code to pass the tests. Scoring stage stays LLM-free.
4. Re-run pytest after every change. A stage isn't done until its tests pass.

## Tooling

- Repo venv is at `../.venv`; `pytest` is installed for writing tests.
- Reuse the existing LLM client patterns already in the repo root rather than
  introducing a new SDK unless we decide otherwise: `gemma_init.py` (local
  Ollama) and `qwen_init.py` (OpenRouter, note it already has a commented-out
  pathology-field-extraction prompt — a good starting point). Both load keys
  via `python-dotenv` from `../.env`.
- `.env` holds `OPENROUTER_API_KEY` / `HF_TOKEN` and is gitignored — never
  hardcode, print, or commit its contents.

## Non-goals

- No treatment recommendations, drug interactions, or anything beyond staging.
- No real patient data, ever — synthetic vignettes only.
- Don't build clinical-grade robustness (audit trails, regulatory compliance,
  etc.) — this is about learning the extraction → scoring → test pattern.

## Proposed file layout (adjust as we build)

- `her2_status.py` — scoring stage (pure functions)
- `extract.py` — LLM extraction stage
- `pipeline.py` — orchestration entry point
- `test_her2_status.py` — pytest suite (scoring stage first)
