# DIY-Assist five-stage implementation plan

## Purpose

DIY-Assist is currently a functional prototype, not yet the full system described in the README architecture. The repository already contains a FastAPI backend, static frontend, naive ChromaDB retrieval, iFixit ingestion, local SLM generation through Ollama or LM Studio, keyword-based safety guardrails, and a minimal health test. [^1] [^2] [^3] [^4] [^5] [^6]​

The README requires a more complete agentic RAG application: React/Next.js frontend, richer ingestion and chunking, reranked RAG, HyDE, metadata-filtered retrieval, ReAct-style tool orchestration, fine-tuning, DPO safety alignment, and an evaluation framework. [^7] [^7] [^7] [^7] [^7] [^7] [^8] [^7]​

This plan splits the remaining work into five similarly sized stages so that a university team can work in parallel while still integrating the system in a controlled order.

## Current status compared with README requirements

### Already implemented or mostly implemented

The project already exposes `/api/v1/health`, `/api/v1/rag/naive`, and `/api/v1/chat`, with guardrail checks applied before retrieval or chat generation. [^2] This covers the README’s first baseline goal for a naive RAG endpoint and chat endpoint. [^7]​

The retriever uses ChromaDB, BGE-small-en-v1.5 embeddings, optional appliance-category filtering, normalized similarity-like scores, and previous-step stitching based on guide metadata. [^3] This partially satisfies the README’s ChromaDB and metadata-filtered retrieval direction, but only at the baseline level. [^9] [^10]​

The iFixit ingestion script can fetch guides, fetch guide details, save raw payloads to `data/raw`, chunk steps, generate embeddings, and upsert documents with metadata into ChromaDB. [^11] This means the README item about raw iFixit archival is already implemented, even though it is still listed as an immediate next task. [^7]​

The chat system already has an SLM response layer through `SLMWrapper`, which builds a grounded prompt from retrieved chunks and live iFixit candidates, then calls Ollama or LM Studio with a safe fallback. [^4] This means the README item about adding SLM response generation is also partially implemented. [^7]​

A minimal frontend exists, but it is static HTML and JavaScript rather than React/Next.js. [^12] [^13] The README requires a React/Next.js web application with chat history, appliance selection, and richer repair guidance rendering. [^7]​

### Not yet implemented or incomplete

The repository still needs cleanup because the file tree contains duplicate or experimental files such as `routes.py` and `routes_clean.py`, `llm.py` and `llm 2.py`, `ifixit_live.py` and `ifixit_live 2.py`, plus duplicate frontend files. [^1]​

The current chunker is a lightweight character-based buffer with `max_chars=1800`, not the README’s final chunking strategy based on repair-guide steps, sentence-boundary splitting, 512-token chunks, overlap, and rich metadata. [^14] [^7] [^7]​

The current guardrails are keyword-based and block hazards such as gas leak, sparking, exposed wire, electrocution, refrigerant leak, and smoke from appliance. [^5] The README describes stronger safety behavior through an agentic guardrail tool and DPO preference alignment. [^7] [^8]​

Reranked RAG, HyDE, a real ReAct loop, part identification, symptom clarification, step-by-step guide formatting, SQLite chat persistence, fine-tuning, DPO alignment, and RAGAS-style evaluation are not visible in the current repository. [^7] [^7] [^7] [^7] [^7] [^8] [^7]​

## Stage 1, repository stabilization and baseline reliability

### Goal

Make the existing prototype clean, reproducible, and easy for the whole team to run before adding new architecture. This stage should end with a stable baseline demo: backend starts, sample ingestion works, naive retrieval works, chat works, guardrail blocking works, and tests pass.

### Tasks

1. Remove duplicate or experimental files and decide on one canonical route module, one LLM service, one iFixit live service, and one frontend entry point. The repository currently contains duplicate-looking files that can confuse contributors and imports. [^1]2. Update `README.md` so it accurately marks implemented items as done: naive RAG, chat endpoint, retrieval service, ingestion CLI, raw payload archival, minimal frontend, and local SLM wrapper. The README still lists SLM response generation and raw payload archival as immediate next tasks even though code for both exists. [^7] [^4] [^11]3. Add or commit a small `data/raw/sample_ifixit_minimal.json` file, or change `make ingest-sample` to point to a file that actually exists in the repository. The Makefile currently expects `data/raw/sample_ifixit_minimal.json` for local sample ingestion. [^15]4. Expand tests beyond health checks. The current automated test coverage only verifies `/api/v1/health`. [^6] Add tests for `/api/v1/rag/naive`, `/api/v1/chat`, guardrail blocking, missing Chroma data handling, and sample ingestion.
5. Add a short developer workflow section to the README with exact commands: `make bootstrap`, copy `.env`, `make ingest-sample`, `make run-backend`, open `/docs`, open `/frontend`, run `make test`. The README already lists bootstrap, environment setup, backend startup, health check, root, and docs checks. [^16]
### Deliverables

* Clean file tree with duplicate modules removed or documented.
* Updated README status checklist.
* Working sample ingestion path.
* Baseline backend and frontend demo instructions.
* Test suite covering the baseline path.

## Stage 2, ingestion, metadata, and retrieval quality

### Goal

Turn the current ingestion and retrieval layer into the data foundation required by the README. This stage should improve chunk quality, metadata quality, retrieval filtering, and source attribution before adding more advanced RAG strategies.

### Tasks

1. Replace the current lightweight chunker with the README’s intended guide-step-aware chunking. The README requires each iFixit guide step to become an atomic chunk, long steps to split at sentence boundaries, and overlap to preserve context. [^7] The current implementation only buffers text until `max_chars=1800`. [^14]2. Standardize metadata fields for every chunk: `guide_id`, `guide_title`, `appliance_category`, `brand`, `model`, `difficulty`, `tools`, `step_number`, `chunk_number`, and `source_url`. The README says metadata should enable appliance, brand, guide, difficulty, tool, and step-aware retrieval. [^7]3. Improve iFixit extraction so title, category, guide ID, public URL, step text, tool information, and difficulty are consistently extracted from API responses. The current ingestion script already extracts IDs, titles, category, URL, brand, model, steps, and writes metadata, so this task should build on the existing implementation rather than replace it. [^11]4. Add retrieval filters for appliance type, brand, and possibly model. The current retriever only filters on `appliance_category` when provided. [^3] The README expects metadata filtering when the user specifies appliance type or brand. [^10]5. Improve citation payloads returned by `/api/v1/chat` and `/api/v1/rag/naive` so each result includes guide title, source URL, step number, score, and any previous-step context used in the generated answer. The current chat route already builds citations from guide title, source URL, step number, and previous steps. [^2]
### Deliverables

* Finalized chunking module aligned with README requirements.
* Robust iFixit metadata extraction.
* Retrieval filters beyond appliance category.
* Better citation and source metadata in API responses.
* Tests for ingestion, chunking, metadata, and filtered retrieval.

## Stage 3, advanced RAG strategies and evaluation harness

### Goal

Implement the README’s comparison layer: naive RAG as the baseline, reranked RAG as the first improvement, HyDE as the second improvement, and an evaluation harness to compare them fairly.

### Tasks

1. Implement reranked RAG as a new retrieval strategy. The README specifies retrieving top candidates from ChromaDB, passing top-20 results through `cross-encoder/ms-marco-MiniLM-L-6-v2`, and selecting the top-5 for the agent context. [^7]2. Add a new API endpoint or strategy parameter for reranked RAG, such as `/api/v1/rag/reranked` or `/api/v1/rag/search?strategy=reranked`. The current API only exposes `/api/v1/rag/naive` and `/api/v1/chat`. [^2]3. Implement HyDE as an optional retrieval strategy after reranking is stable. The README describes generating a hypothetical answer, embedding it, and using that embedding as the retrieval query instead of the raw user query. [^7]4. Build a lightweight evaluation harness first, then expand it. The README calls for a simple query set and retrieval quality checks as an immediate task, and later RAGAS metrics for faithfulness, answer relevance, and context relevance. [^7] [^7]5. Create a benchmark dataset with at least 30 to 50 initial troubleshooting queries, expected appliance category, expected guide or source URL where possible, and expected safety behavior for hazardous prompts. The README later targets 100 to 200 troubleshooting questions with ground-truth answers, so this stage should create the smaller version first. [^7]
### Deliverables

* Reranked retrieval module and API path.
* Optional HyDE retrieval strategy.
* Evaluation script for naive, reranked, and HyDE strategies.
* Initial benchmark query set.
* Short results table showing retrieval quality before and after reranking.

## Stage 4, agent tools, safety behavior, and chat persistence

### Goal

Move from direct RAG chat to the agentic system described in the README. This stage should introduce tool interfaces, multi-turn state, better safety escalation, and a more structured answer format.

### Tasks

1. Define tool interfaces for `Manual_Search_Tool`, `Safety_Protocol_Checker`, `Part_Identifier`, `Symptom_Clarifier`, and `Step_By_Step_Guide_Formatter`. The README describes these as the core tools available to the ReAct agent. [^7] [^17]2. Add a first ReAct-style orchestration loop around the current retriever and SLM wrapper. The README says the agent should reason about the user issue, select a tool, observe the result, and either call another tool or answer. [^7]3. Improve safety beyond keyword blocking. The current guardrail only checks a fixed set of hazard phrases. [^5] The README requires safety escalation for hazardous conditions such as gas smells, exposed wiring, refrigerant, high voltage, and other professional-only scenarios. [^7] [^7]4. Add SQLite chat persistence for conversation history, user sessions, previous-step context, and multi-turn troubleshooting. The README says chat sessions are persisted in SQLite on the backend. [^7]5. Make answers structured and UI-ready: safety warning, likely issue, clarifying question when needed, retrieved evidence, step-by-step guidance, parts list when available, and source citations. The README expects the frontend to render safety warnings, parts lists, links, and step-by-step repair guidance. [^7]
### Deliverables

* Tool interface module.
* First ReAct-style agent service.
* Stronger safety checker with tests.
* SQLite persistence for chat sessions.
* Structured response schema for frontend rendering.

## Stage 5, frontend, fine-tuning, alignment, and final demo polish

### Goal

Complete the system so it matches the README’s final presentation target: end-to-end demo, all major tools functional, comparative evaluation results, documentation, and reproducible setup.

### Tasks

1. Replace or wrap the current static frontend with a React/Next.js chat interface. The current frontend is useful for testing, but the README requires a React/Next.js application with chat history, appliance selection, and step-by-step repair guidance display. [^12] [^7]2. Build UI components for structured responses: safety warning banners, clarifying questions, parts list, retrieved source links, live iFixit guide candidates, and step-by-step guide cards. The README explicitly says the UI should support safety warnings, parts lists, and embedded links to source documentation. [^7]3. Add fine-tuning notebooks or scripts for Qwen 2.5 3B using QLoRA. The README describes three stages: SFT on tool calling, domain SFT for appliance troubleshooting, and DPO alignment. [^7]4. Create the safety-focused DPO preference dataset and training workflow. The README specifies preference pairs where safe, grounded, professional-escalating responses are chosen over unsafe, hallucinated, or missing-warning responses. [^18] [^8]5. Prepare the final demonstration and documentation. The README’s final target is an end-to-end demo with functional tools, comparative results across RAG strategies, architecture and fine-tuning documentation, evaluation metrics, setup instructions, and reproducibility guidance. [^19]
### Deliverables

* React/Next.js frontend or clearly documented equivalent if the team decides to keep static frontend.
* Final structured chat UI integrated with backend response schema.
* Fine-tuning and DPO scripts or notebooks.
* Final evaluation report comparing naive RAG, reranked RAG, HyDE, and agent behavior.
* Final README with setup, demo, architecture, evaluation, and reproducibility instructions.

## Suggested team split

If the team has five members, assign one owner per stage. Stage owners should not work in isolation: Stage 2 and Stage 3 need to coordinate on retrieval APIs, Stage 4 depends on Stage 3’s retrieval strategies, and Stage 5 depends on Stage 4’s response schema.

A practical split is:

1. Repository and baseline owner: cleanup, README, sample data, baseline tests.
2. Data and retrieval owner: ingestion, chunking, metadata, filtering, citations.
3. RAG evaluation owner: reranking, HyDE, benchmark queries, comparison scripts.
4. Agent and safety owner: tools, ReAct loop, guardrails, SQLite persistence, structured responses.
5. Frontend and final demo owner: React/Next.js UI, fine-tuning materials, DPO workflow, final presentation package.

## Integration checkpoints

After Stage 1, everyone should be able to clone the repo, run setup, ingest sample data, start the backend, use the frontend, and run tests.

After Stage 2, the team should freeze the chunk metadata schema so Stage 3 and Stage 4 can build against stable retrieval outputs.

After Stage 3, the team should choose the default retrieval strategy for `/api/v1/chat`, based on benchmark results rather than assumptions.

After Stage 4, the team should freeze the structured chat response schema so the frontend work in Stage 5 does not keep chasing backend changes.

After Stage 5, the README should describe the actual implemented system, not just the original intended architecture.

---

## References

[^1]: [Missing authors], "repository tree,", n.d.. Available: [https://api.github.com/repos/maraielciu1/diy-assist/git/trees/main?recursive=1](https://api.github.com/repos/maraielciu1/diy-assist/git/trees/main?recursive=1)

[^2]: [Missing authors], "routes_clean.py,", n.d.. Available: [https://raw.githubusercontent.com/maraielciu1/diy-assist/main/backend/app/api/routes_clean.py](https://raw.githubusercontent.com/maraielciu1/diy-assist/main/backend/app/api/routes_clean.py)

[^3]: [Missing authors], "retrieval.py,", n.d.. Available: [https://raw.githubusercontent.com/maraielciu1/diy-assist/main/backend/app/services/retrieval.py](https://raw.githubusercontent.com/maraielciu1/diy-assist/main/backend/app/services/retrieval.py)

[^4]: [Missing authors], "llm.py,", n.d.. Available: [https://raw.githubusercontent.com/maraielciu1/diy-assist/main/backend/app/services/llm.py](https://raw.githubusercontent.com/maraielciu1/diy-assist/main/backend/app/services/llm.py)

[^5]: [Missing authors], "guardrails.py,", n.d.. Available: [https://github.com/maraielciu1/diy-assist/blob/main/backend/app/services/guardrails.py](https://github.com/maraielciu1/diy-assist/blob/main/backend/app/services/guardrails.py)

[^6]: [Missing authors], "test_health.py,", n.d.. Available: [https://github.com/maraielciu1/diy-assist/blob/main/backend/tests/test_health.py](https://github.com/maraielciu1/diy-assist/blob/main/backend/tests/test_health.py)

[^7]: [Missing authors], "GitHub - maraielciu1/diy-assist,", n.d..

[^8]: [Missing authors], "Untitled,", n.d..

[^9]: [Missing authors], "Untitled,", n.d..

[^10]: [Missing authors], "Untitled,", n.d..

[^11]: [Missing authors], "ingest_ifixit.py,", n.d.. Available: [https://raw.githubusercontent.com/maraielciu1/diy-assist/main/scripts/ingest_ifixit.py](https://raw.githubusercontent.com/maraielciu1/diy-assist/main/scripts/ingest_ifixit.py)

[^12]: [Missing authors], "frontend index.html,", n.d.. Available: [https://github.com/maraielciu1/diy-assist/blob/main/frontend/index.html](https://github.com/maraielciu1/diy-assist/blob/main/frontend/index.html)

[^13]: [Missing authors], "frontend main.js,", n.d.. Available: [https://raw.githubusercontent.com/maraielciu1/diy-assist/main/frontend/main.js](https://raw.githubusercontent.com/maraielciu1/diy-assist/main/frontend/main.js)

[^14]: [Missing authors], "ingestion.py,", n.d.. Available: [https://github.com/maraielciu1/diy-assist/blob/main/backend/app/services/ingestion.py](https://github.com/maraielciu1/diy-assist/blob/main/backend/app/services/ingestion.py)

[^15]: [Missing authors], "Makefile,", n.d.. Available: [https://raw.githubusercontent.com/maraielciu1/diy-assist/main/Makefile](https://raw.githubusercontent.com/maraielciu1/diy-assist/main/Makefile)

[^16]: [Missing authors], "Untitled,", n.d..

[^17]: [Missing authors], "Untitled,", n.d..

[^18]: [Missing authors], "Untitled,", n.d..

[^19]: [Missing authors], "Untitled,", n.d..
