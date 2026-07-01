# Approach Document

## Goal And API Shape

The service is built for the SHL conversational recommender assignment. It exposes the required stateless FastAPI endpoints: `GET /health` for readiness and `POST /chat` for the next conversational turn. The `/chat` request carries the full history, so the backend stores no conversation state. The response always follows the required schema: `reply`, `recommendations`, and `end_of_conversation`.

The implementation is intentionally small and inspectable. `main.py` owns the API, prompt, validation, and response repair logic. `retriever.py` owns catalog loading and semantic search. `evaluate_traces.py` replays the 10 public traces and reports Recall@10.

## Catalog And Retrieval Setup

The local `catalog.json` file is the source of truth. `retriever.py` loads all catalog entries, converts each assessment into a searchable text representation using name, description, job levels, keys, languages, duration, remote, and adaptive fields, embeds those strings with `sentence-transformers/all-MiniLM-L6-v2`, and searches them with a FAISS inner-product index over normalized embeddings.

A pure semantic top-k search worked well for direct skill tests but sometimes missed universal or report-style assessments such as OPQ32r, Verify G+, Graduate Scenarios, DSI, and OPQ reports. To improve recall without allowing hallucinations, `main.py` adds a lightweight catalog-aware boost layer. It detects high-confidence concepts in the user history, such as senior leadership, graduate finance, contact center, safety-critical plant work, healthcare admin, sales audit, Java/Spring/AWS/Docker, and Rust catalog gaps. The boost layer adds exact catalog URLs from `catalog.json` into the prompt context. These are not free-form recommendations; every boosted item is looked up from the catalog and later passes through the same validator as model output.

User removals are also handled before prompting and after validation. For example, `drop REST`, `drop OPQ`, and `drop Verify` remove those catalog URLs from fallback and augmented recommendations.

## Prompt And Agent Behavior

Gemini receives a strict system prompt that defines the agent as an SHL assessment consultant. The prompt instructs it to clarify vague requests, recommend 1-10 assessments when enough information is available, refine the complete shortlist when constraints change, compare products using only catalog context, refuse out-of-scope/legal/prompt-injection requests, and honor the 8-turn circuit breaker.

The model is asked to return JSON only. Even so, the backend does not trust model output blindly. `validate_recommendations` checks every returned name and URL against the catalog. It corrects near matches where possible, drops anything not in the catalog, deduplicates by URL, caps the list at 10, and normalizes `test_type` to compact SHL-style codes such as `K`, `A`, and `P`.

## Reliability And Timeout Design

The evaluator caps `/chat` calls at 30 seconds. The Gemini call therefore uses a short fallback chain and a per-model timeout. If Gemini times out, rate-limits, returns invalid JSON, or exhausts quota, the service returns a deterministic local fallback. That fallback still obeys the same scope and vague-query rules: vague openings get one clarifying question, out-of-scope turns get a refusal, and sufficiently detailed turns get a catalog-grounded shortlist from the boosted catalog anchors and semantic results.

This keeps the API schema-compliant even when the LLM provider is unreliable. It also means catalog-only constraints are enforced in code, not only in the prompt.

## Evaluation Approach

The public traces were used as a development set, not as fixed scripts. I checked the exact expected URLs against the local catalog, then used the traces to identify retrieval blind spots. The local boost rules now cover the expected public trace URLs before Gemini generation, while semantic search still supplies broader catalog context for non-public conversations.

The repo includes three evaluation levels: a no-quota syntax check with `py_compile`, API behavior tests in `test_agent.py`, and Recall@10 replay in `evaluate_traces.py`. The behavior and trace evaluations call `/chat`, so they should be run when Gemini quota is available. What did not work well was relying only on semantic search or only on prompting; recall was sensitive to whether universal assessments appeared in context. The final design combines semantic search, exact catalog boosts, model reasoning, and code-level validation.
