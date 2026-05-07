# DIY-Assist five implementation prompts

Use these prompts one at a time, in order. Each prompt assumes the developer or coding assistant has access to the GitHub repository and can inspect, edit, run, and test the project locally.

Each stage prompt includes three parts: first, a check that the previous stage is complete; second, the current implementation work; third, concrete testing steps to prove the stage is finished.

## Prompt 1, repository stabilization and baseline reliability

You are working on Stage 1 of the DIY-Assist project. Your job is to stabilize the existing prototype before any new architecture is added.

### Previous-stage check

There is no previous stage for Stage 1. Before making changes, inspect the current repository and establish the baseline state.

Check that the project currently contains:

* A FastAPI backend.
* A minimal frontend.
* A naive RAG endpoint.
* A chat endpoint.
* An iFixit ingestion script.
* A local SLM wrapper.
* A keyword-based safety guardrail module.
* A Makefile with setup, backend, test, and ingestion commands.

If any of these are missing or broken, fix them before moving into the rest of Stage 1.

### Current-stage tasks

Clean the repository structure. Remove or consolidate duplicate and experimental files such as duplicate route modules, duplicate LLM service files, duplicate iFixit service files, and duplicate frontend files. Keep one canonical backend route module, one LLM service, one iFixit service, and one frontend entry point.

Make the application startup path clear. The backend should import the correct route module, serve the API under `/api/v1`, expose `/docs`, expose a root route, and serve the frontend from the configured frontend path.

Update the README so it reflects the actual current status of the code. Mark the baseline pieces that already exist as implemented: health endpoint, naive RAG endpoint, chat endpoint, retrieval service, ingestion CLI, raw payload archival, minimal frontend, local SLM wrapper, and safety guardrails.

Make sample ingestion reproducible. Either add a small `data/raw/sample_ifixit_minimal.json` file to the repository or change the sample ingestion command so it points to a file that is actually present. The sample should be small enough to commit and good enough to test the complete ingestion and retrieval path.

Expand the test suite. Add tests for health, naive RAG, chat, safety blocking, sample ingestion, and graceful behavior when the Chroma database is empty or unavailable. Tests should be easy to run with `make test`.

### Expected end result

At the end of Stage 1, a teammate should be able to clone the repository, run setup, create the environment file, ingest sample data, start the backend, open the frontend, call the API, and run tests without guessing which files are canonical.

The repository should no longer contain confusing duplicate implementation files unless they are clearly documented as intentionally unused.

The README should match the actual project state instead of describing only the future architecture.

### How to test completion

Run the following checks locally:

```bash
make bootstrap
cp .env.example .env
make ingest-sample
make run-backend
```

In another terminal, run:

```bash
curl http://127.0.0.1:8000/
curl http://127.0.0.1:8000/api/v1/health
curl -X POST http://127.0.0.1:8000/api/v1/rag/naive \
  -H "Content-Type: application/json" \
  -d '{"query":"washer not draining","appliance_category":"Appliance","top_k":3}'
curl -X POST http://127.0.0.1:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"query":"my washer is not draining and makes a humming noise","appliance_category":"Appliance","top_k":3}'
curl -X POST http://127.0.0.1:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"query":"I smell gas near my dryer, what should I do?"}'
make test
```

Completion criteria:

* Backend starts without import errors.
* `/` returns the app status and points to docs and frontend.
* `/api/v1/health` returns status ok.
* Sample ingestion inserts usable chunks.
* Naive RAG returns retrieved results.
* Chat returns either a generated answer or a safe fallback.
* Hazardous prompts are blocked.
* Tests pass.

## Prompt 2, ingestion, metadata, and retrieval quality

You are working on Stage 2 of the DIY-Assist project. Your job is to improve the data and retrieval foundation so later RAG strategies and agent tools can depend on stable outputs.

### Previous-stage check

Before starting Stage 2, verify that Stage 1 is complete.

Check that:

* The repository has one canonical route module, one LLM service, one iFixit service, and one frontend entry point.
* The backend starts reliably.
* `make ingest-sample` works.
* `make test` passes.
* The README accurately describes the baseline state.
* The naive RAG and chat endpoints work against the sample data.

If any of these checks fail, fix Stage 1 first. Do not add the Stage 2 changes on top of an unstable baseline.

### Current-stage tasks

Replace the lightweight chunker with a guide-step-aware chunking module. Each iFixit guide step should be treated as the natural unit of repair instruction. Very long steps should be split cleanly at sentence boundaries, with overlap where needed to preserve context. Very short fragments should be merged only when doing so improves retrieval quality and does not mix unrelated repair actions.

Standardize the metadata schema for every stored chunk. Include fields such as `guide_id`, `guide_title`, `appliance_category`, `brand`, `model`, `difficulty`, `tools`, `step_number`, and `chunk_number`. Keep field names consistent across ingestion, retrieval, API responses, tests, and frontend expectations.

Improve iFixit extraction. Make extraction robust across different shapes of iFixit API responses. The ingestion script should consistently extract guide title, category, guide ID, guide text, step text, tools, difficulty, brand, and model when available. Missing optional values should be handled safely.

Improve retrieval filtering. Keep appliance-category filtering and add brand and model filtering where possible. The retrieval service should accept optional filters without breaking older callers.

Improve response payloads. Retrieval and chat responses should include the fields needed by later stages: guide title, step number, score, previous-step context, and relevant chunk metadata. Keep the response schema stable and documented.

Add tests for chunking, metadata extraction, sample ingestion, filtered retrieval, and API response shape.

### Expected end result

At the end of Stage 2, the project should have a reliable ingestion pipeline and a stable retrieval schema. Later work should not need to guess which metadata fields exist or how retrieval filters are applied.

Naive RAG should still work exactly as before, but with better chunks, cleaner metadata, and richer response payloads.

### How to test completion

Reset local data, ingest the sample, and run tests:

```bash
rm -rf data/chroma
make ingest-sample
make test
```

Then manually check retrieval:

```bash
make run-backend
curl -X POST http://127.0.0.1:8000/api/v1/rag/naive \
  -H "Content-Type: application/json" \
  -d '{"query":"washer not draining","appliance_category":"Appliance","top_k":5}'
```

Also test filtered retrieval with any brand or model included in the sample dataset. If the sample data does not include brand or model, add one small sample guide that does.

Completion criteria:

* Chunking tests show long, short, and normal steps are handled correctly.
* Ingested chunks contain the standardized metadata fields.
* Retrieval still works with only a query.
* Retrieval works with appliance-category filtering.
* Retrieval works with brand or model filtering when those fields exist.
* API responses include guide title, step number, score, previous-step context, and metadata.
* Existing Stage 1 tests still pass.

## Prompt 3, advanced RAG strategies and evaluation harness

You are working on Stage 3 of the DIY-Assist project. Your job is to implement advanced retrieval strategies and create a fair way to compare them.

### Previous-stage check

Before starting Stage 3, verify that Stage 2 is complete.

Check that:

* The guide-step-aware chunker is implemented and tested.
* Ingestion produces stable metadata fields.
* Sample ingestion works from a clean data directory.
* Retrieval filters work for appliance category and, where available, brand or model.
* API responses have a stable shape.
* All Stage 1 and Stage 2 tests pass.

If any of these checks fail, fix them before implementing reranking or HyDE.

### Current-stage tasks

Implement reranked RAG as a separate retrieval strategy. Start with the existing naive retriever, retrieve a larger candidate set, rerank those candidates with `cross-encoder/ms-marco-MiniLM-L-6-v2`, and return the best final chunks for generation.

Keep naive retrieval available. Do not replace the baseline. The project needs both naive and reranked retrieval so the team can compare them.

Add a clean way to select retrieval strategy. This can be a new endpoint such as `/api/v1/rag/reranked`, or a strategy parameter such as `strategy=naive` or `strategy=reranked`. Keep the design simple and document it.

Implement HyDE as an optional strategy after reranking is stable. HyDE should generate a hypothetical answer, embed that answer, then use it as the retrieval query. Keep HyDE separate from reranking so the team can test naive, reranked, HyDE, and combinations if needed.

Build an evaluation harness. Create a small benchmark file with 30 to 50 troubleshooting queries. Each item should include the user query, expected appliance category, expected behavior, and expected relevant guide or page where possible. Include hazardous prompts so safety behavior can also be checked.

Create a comparison script that runs the same benchmark against naive retrieval, reranked retrieval, and HyDE. The output should be a simple table or JSON report showing retrieval quality, basic answer quality checks, safety behavior, and runtime.

### Expected end result

At the end of Stage 3, the project should support multiple retrieval strategies and have a repeatable benchmark for comparing them. The team should be able to choose the default retrieval strategy based on measured results instead of assumptions.

### How to test completion

Start from clean sample data:

```bash
rm -rf data/chroma
make ingest-sample
make test
```

Run each retrieval strategy manually:

```bash
make run-backend
curl -X POST http://127.0.0.1:8000/api/v1/rag/naive \
  -H "Content-Type: application/json" \
  -d '{"query":"washer not draining","appliance_category":"Appliance","top_k":5}'

curl -X POST http://127.0.0.1:8000/api/v1/rag/reranked \
  -H "Content-Type: application/json" \
  -d '{"query":"washer not draining","appliance_category":"Appliance","top_k":5}'
```

If HyDE has its own endpoint or strategy parameter, run the equivalent HyDE command as well.

Run the benchmark script, for example:

```bash
python scripts/evaluate_rag.py --input data/eval/troubleshooting_queries.json --strategies naive reranked hyde
```

Completion criteria:

* Naive retrieval still works.
* Reranked retrieval works and returns a clearly reranked result set.
* HyDE works or is clearly marked as optional and documented if not enabled by default.
* Benchmark data exists and is easy to extend.
* Evaluation script produces a readable comparison report.
* Tests cover strategy selection, reranking behavior, HyDE behavior where enabled, and benchmark loading.
* Stage 1 and Stage 2 tests still pass.

## Prompt 4, agent tools, safety behavior, and chat persistence

You are working on Stage 4 of the DIY-Assist project. Your job is to move the project from direct RAG chat toward the agentic system described in the README.

### Previous-stage check

Before starting Stage 4, verify that Stage 3 is complete.

Check that:

* Naive retrieval still works.
* Reranked retrieval is implemented and testable.
* HyDE is implemented or intentionally documented as optional.
* The benchmark harness runs.
* The team has chosen or documented the default retrieval strategy for chat.
* All earlier tests pass.

If any of these checks fail, fix the earlier stage before adding agent tools.

### Current-stage tasks

Define tool interfaces for the agent. Implement clear, typed interfaces for `Manual_Search_Tool`, `Safety_Protocol_Checker`, `Part_Identifier`, `Symptom_Clarifier`, and `Step_By_Step_Guide_Formatter`. Each tool should have a clear input schema, output schema, and tests.

Implement a first ReAct-style orchestration loop. The loop should receive a user issue, decide whether safety must be checked first, call retrieval when needed, ask a clarifying question when the issue is ambiguous, identify likely parts when possible, and format the final answer into a structured response.

Improve safety behavior. Keep the current keyword guardrails, but add stronger handling for hazardous situations such as gas smells, exposed wiring, refrigerant, high voltage, sparking, burning smells, and smoke. Hazardous prompts should stop troubleshooting and return a clear professional-escalation response.

Add SQLite chat persistence. Store sessions, messages, selected appliance information, previous-step context, and the final structured response. Keep the database simple and local.

Define a structured chat response schema. The frontend should receive predictable fields such as safety warning, answer summary, clarifying question, likely issue, steps, parts list, retrieved guide content, and tool trace summary. The exact names can differ, but the schema must be stable and documented.

Update `/api/v1/chat` to use the new agent service while keeping old behavior available behind a fallback or compatibility path.

### Expected end result

At the end of Stage 4, `/api/v1/chat` should behave like an early agent rather than a direct retrieval wrapper. It should handle safety, retrieval, clarification, part identification, structured guidance, and conversation persistence in a consistent way.

The backend should now be ready for a richer frontend because responses are structured and predictable.

### How to test completion

Run the full test suite:

```bash
make test
```

Start the backend and test representative chat flows:

```bash
make run-backend
```

Safe troubleshooting query:

```bash
curl -X POST http://127.0.0.1:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"query":"my washer is not draining and makes a humming noise","appliance_category":"Appliance","top_k":5}'
```

Ambiguous query:

```bash
curl -X POST http://127.0.0.1:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"query":"my washer is leaking","appliance_category":"Appliance","top_k":5}'
```

Hazardous query:

```bash
curl -X POST http://127.0.0.1:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"query":"I smell gas near my dryer, what should I do?"}'
```

Completion criteria:

* Safe troubleshooting returns structured repair guidance.
* Ambiguous troubleshooting returns a useful clarifying question or a clearly marked uncertainty response.
* Hazardous troubleshooting stops repair guidance and returns professional escalation.
* SQLite stores sessions and messages.
* Tool interfaces have unit tests.
* `/api/v1/chat` returns a stable response schema.
* Earlier retrieval and evaluation tests still pass.

## Prompt 5, frontend, fine-tuning, alignment, and final demo polish

You are working on Stage 5 of the DIY-Assist project. Your job is to complete the user-facing system and prepare the final demo package.

### Previous-stage check

Before starting Stage 5, verify that Stage 4 is complete.

Check that:

* `/api/v1/chat` uses the agent service or a documented fallback.
* Structured chat responses are stable.
* Safety escalation works.
* Clarifying questions work for ambiguous prompts.
* Chat persistence works in SQLite.
* Retrieval strategies and evaluation scripts still work.
* All tests pass.

If any of these checks fail, fix Stage 4 before working on the frontend or final demo polish.

### Current-stage tasks

Build or migrate to a React/Next.js frontend. The UI should include chat history, appliance selection, message input, response rendering, safety warning display, step-by-step repair guidance, parts list rendering, and live iFixit guide candidate rendering if that feature remains enabled.

Connect the frontend to the structured `/api/v1/chat` response schema. Do not hard-code around raw text only. Render each structured field intentionally so the user can understand safety status, likely issue, next steps, parts, and any clarification needed.

Improve local development workflow. The frontend and backend should be easy to run together. Add scripts or README instructions for installing frontend dependencies, starting frontend development mode, starting the backend, and testing the end-to-end flow.

Add fine-tuning materials. Create notebooks or scripts for Qwen 2.5 3B with QLoRA. Include the intended stages: supervised fine-tuning for tool calling, domain-specific supervised fine-tuning for appliance troubleshooting, and DPO alignment for safer answers.

Create the DPO preference dataset workflow. Add a small starter dataset and a clear format for chosen and rejected responses. Include examples where the chosen answer is safe, grounded, and escalates to a professional when needed, while the rejected answer is unsafe, speculative, or missing a safety warning.

Prepare the final demo package. Update the README with setup instructions, system architecture, implemented features, known limitations, test commands, evaluation results, and demo script. Include a short final presentation flow that shows the baseline, improved retrieval, agent behavior, safety blocking, and frontend.

### Expected end result

At the end of Stage 5, the project should be ready for final university presentation. A reviewer should be able to run the project, test the main flows, understand the architecture, see evaluation results, and use the frontend without reading the code first.

The final system should demonstrate the full path from user issue to safe, structured troubleshooting guidance.

### How to test completion

Run backend tests:

```bash
make test
```

Run the evaluation harness:

```bash
python scripts/evaluate_rag.py --input data/eval/troubleshooting_queries.json --strategies naive reranked hyde
```

Start backend and frontend:

```bash
make run-backend
```

Then start the frontend with the project’s documented command, for example:

```bash
cd frontend
npm install
npm run dev
```

Test the following flows in the browser:

1. Safe query: “My washer is not draining and makes a humming noise.”
2. Ambiguous query: “My washer is leaking.”
3. Hazardous query: “I smell gas near my dryer.”
4. Query with appliance category selected.
5. Follow-up query in the same session.

Completion criteria:

* Frontend starts without errors.

* Backend starts without errors.
* Frontend can call `/api/v1/chat` successfully.
* Safe query renders structured repair guidance.
* Ambiguous query renders a clarifying question or uncertainty response.
* Hazardous query renders a safety escalation and does not show repair steps.
* Chat history persists.
* Evaluation report exists and is documented.
* Fine-tuning and DPO materials exist, even if training is not run during the demo.
* README includes setup, testing, architecture, evaluation, limitations, and demo instructions.