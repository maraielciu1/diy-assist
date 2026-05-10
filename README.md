# DIY-Assist: Agentic RAG for Appliance Troubleshooting

## Stage 1 Baseline Status

Implemented now:

- [x] FastAPI backend with API mounted under `/api/v1`
- [x] Root route (`/`) with links to docs and frontend
- [x] Health endpoint (`GET /api/v1/health`)
- [x] Naive RAG endpoint (`POST /api/v1/rag/naive`)
- [x] Chat endpoint (`POST /api/v1/chat`)
- [x] Retrieval service (`backend/app/services/retrieval.py`)
- [x] iFixit ingestion CLI (`scripts/ingest_ifixit.py`)
- [x] Raw payload archival to `data/raw/`
- [x] Minimal frontend served at `/frontend`
- [x] Local SLM wrapper (`backend/app/services/llm.py`)
- [x] Keyword safety guardrails (`backend/app/services/guardrails.py`)

## Quickstart

1. Bootstrap and create env file:
   - `make bootstrap`
   - `cp .env.example .env`
2. Ingest sample data:
   - `make ingest-sample`
3. Start backend:
   - `make run-backend`
4. Open:
   - API docs: `http://127.0.0.1:8000/docs`
   - Frontend: `http://127.0.0.1:8000/frontend`

## API Smoke Tests

```bash
curl http://127.0.0.1:8000/
curl http://127.0.0.1:8000/api/v1/health
curl -X POST http://127.0.0.1:8000/api/v1/rag/naive \
  -H "Content-Type: application/json" \
  -d '{"query":"washer not draining","appliance_category":"Appliance","top_k":3}'
curl -X POST http://127.0.0.1:8000/api/v1/rag/reranked \
  -H "Content-Type: application/json" \
  -d '{"query":"washer not draining","appliance_category":"Appliance","top_k":3}'
curl -X POST http://127.0.0.1:8000/api/v1/rag/hyde \
  -H "Content-Type: application/json" \
  -d '{"query":"washer not draining","appliance_category":"Appliance","top_k":3}'
curl -X POST http://127.0.0.1:8000/api/v1/rag/hyde_reranked \
  -H "Content-Type: application/json" \
  -d '{"query":"washer not draining","appliance_category":"Appliance","top_k":3}'
curl -X POST http://127.0.0.1:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"query":"my washer is not draining and makes a humming noise","appliance_category":"Appliance","top_k":3}'
curl -X POST http://127.0.0.1:8000/api/v1/chat/reranked \
  -H "Content-Type: application/json" \
  -d '{"query":"my washer is not draining and makes a humming noise","appliance_category":"Appliance","top_k":3}'
curl -X POST http://127.0.0.1:8000/api/v1/chat/hyde \
  -H "Content-Type: application/json" \
  -d '{"query":"my washer is not draining and makes a humming noise","appliance_category":"Appliance","top_k":3}'
curl -X POST http://127.0.0.1:8000/api/v1/chat/hyde_reranked \
  -H "Content-Type: application/json" \
  -d '{"query":"my washer is not draining and makes a humming noise","appliance_category":"Appliance","top_k":3}'
curl -X POST http://127.0.0.1:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"query":"I smell gas near my dryer, what should I do?"}'
```

## Test Suite

Run:

- `make test`

Current tests cover:

- Health and root routes
- Naive RAG success and guardrail blocking
- Chat success, hazard blocking, and retriever-unavailable fallback
- Ingestion from a local sample payload

## Stage 2 Retrieval Schema

`POST /api/v1/rag/naive` and `POST /api/v1/chat` accept:

- `query` (required)
- `appliance_category` (optional)
- `brand` (optional)
- `model` (optional)
- `top_k` (optional)

Chunk metadata fields are standardized as:

- `guide_id`, `guide_title`, `appliance_category`, `brand`, `model`
- `difficulty`, `tools`, `step_number`, `chunk_number`
- `source_url`, `guide_text`

RAG results include top-level `score`, `guide_title`, `step_number`, `previous_steps`, and full `metadata`.
Chat citations include `guide_title`, `step_number`, `score`, `previous_steps`, and full `metadata`.

## Stage 3 Retrieval Strategies

Simple endpoint-based strategy selection (baseline remains available):

- `POST /api/v1/rag/naive`: dense retrieval (baseline)
- `POST /api/v1/rag/reranked`: dense retrieval → CrossEncoder reranking
- `POST /api/v1/rag/hyde`: HyDE (hypothetical answer embedding) → dense retrieval
- `POST /api/v1/rag/hyde_reranked`: HyDE → dense candidates → CrossEncoder reranking

Chat endpoints mirror retrieval strategy:

- `POST /api/v1/chat`: Stage 4 agent orchestration by default (see below). Retrieval strategy is controlled by `CHAT_RETRIEVAL_STRATEGY` / `retrieval_strategy` (`naive`, `reranked`, `hyde`, `hyde_reranked`). Set `use_legacy_chat: true` for the pre-agent direct RAG + SLM behavior.
- `POST /api/v1/chat/reranked`
- `POST /api/v1/chat/hyde`
- `POST /api/v1/chat/hyde_reranked`

### Stage 3 verification

Stage 3 is implemented and tested: reranked + HyDE routes, `scripts/compare_strategies.py`, `eval/benchmark_stage3.json`, and `backend/tests/test_stage3_strategies.py`. Default chat retrieval strategy is documented via `CHAT_RETRIEVAL_STRATEGY` (default `naive`).

## Stage 4 Agent + SQLite persistence

- **Tools** (typed I/O in `backend/app/services/agent/`): `Manual_Search_Tool`, `Safety_Protocol_Checker`, `Part_Identifier`, `Symptom_Clarifier`, `Step_By_Step_Guide_Formatter`.
- **Orchestration**: safety first → optional clarification for ambiguous symptoms → retrieval → parts hints → formatted steps → SLM answer.
- **Persistence**: SQLite at `CHAT_DB_PATH` (default `./data/chat.sqlite`); responses include `session_id` for follow-up turns.
- **Structured payload**: responses include `structured` (`answer_summary`, `clarifying_question`, `likely_issue`, `steps`, `parts_list`, `retrieved_guide_snippets`, `tool_trace`) plus legacy `answer` and `citations`.
- **Compatibility**: send `"use_legacy_chat": true` on `POST /api/v1/chat` to use the Stage 1–3 direct RAG path only.

## Architecture Plan 

## 1. Project Overview

DIY-Assist is an intelligent, agentic diagnostic application that guides homeowners safely through appliance troubleshooting. [^1] The core problem it addresses is the gap between dense, highly technical manufacturer documentation and the homeowner's ability to safely interpret and apply that information. Existing solutions such as web forums provide contradictory advice, while static manuals and iFixit guides are difficult to navigate without knowing exact part names. [^1] Generic LLMs are prone to hallucinating non-existent parts or troubleshooting steps and lack strict safety guardrails. [^1]​

The system uses a fine-tuned Small Language Model (Qwen 2.5 3B) as the core decision-making agent, operating within a ReAct (Reason + Act) loop to understand user symptoms, ask clarifying questions, and dynamically select diagnostic tools. [^1] Safety is enforced through RLHF-based guardrails: if the agent detects hazardous conditions such as gas smells or exposed wiring, it halts troubleshooting and advises the user to contact a professional. [^1]​

## 2. System Architecture

The application follows a client-server architecture with three main layers.

The frontend is a React (Next.js) web application providing a chat-based interface with conversation history, appliance selection, and step-by-step repair guidance display. Chat sessions are persisted in a SQLite database on the backend. The UI will support rich message rendering including safety warnings, parts lists, and embedded links to source documentation.

The backend is a Python FastAPI server that hosts the fine-tuned Qwen 2.5 3B model, manages the ReAct agent loop, orchestrates tool calls, and serves the REST API. FastAPI was chosen for its async support, automatic OpenAPI documentation, and native compatibility with the Python ML ecosystem.

The data layer consists of a ChromaDB vector database for RAG retrieval, a SQLite database for chat persistence and user sessions, and a local file store for ingested documents and model artifacts.

## 3. Dataset Selection

The project uses two categories of datasets: a RAG knowledge base for retrieval and fine-tuning datasets for model specialization.

### 3.1 RAG Knowledge Base

The primary dataset is MyFixit, a semi-structured collection of 31,601 repair manuals scraped from iFixit across 15 device categories, including Appliance (1,333 manuals, 5,744 steps), Household (1,710 manuals, 7,859 steps), and Electronics (2,343 manuals, 9,765 steps). [^2] Each manual includes step-by-step instructions, required tools, difficulty ratings, and device metadata, making it ideal for structured retrieval.

This will be supplemented with additional data scraped from the iFixit API (v2.0), which provides programmatic access to guides, device wikis, and community Q\&A. [^3] The approach covers all appliance categories with breadth rather than narrowing to specific device types.

Additional datasets for the knowledge base include the spare-part-replacement-notes-v1 dataset (20.5K entries with manufacturer, model number, and spare part type annotations) [^4] and the maintenance\_gpt\_added dataset (5.82K maintenance instruction-response pairs covering problems like clamping issues, pressure faults, and hydraulic failures). [^5]​

### 3.2 Fine-Tuning Datasets

For teaching the model tool-calling behavior, the Glaive Function Calling v2 dataset (113K examples of user queries paired with structured JSON function calls) will be used. This is the standard dataset for training small models to output tool calls and has been validated with Qwen 2.5 3B on free Colab T4 hardware. [^6]​

A custom appliance-specific instruction dataset will also be created, containing examples of multi-turn troubleshooting conversations, safety-critical decision points, and tool-use sequences tailored to the appliance repair domain. This dataset will be formatted in the ChatML/ShareGPT template compatible with Qwen 2.5.

### 3.3 DPO Preference Dataset

For alignment training via DPO, a preference dataset of paired responses will be constructed, with a focus on safety-critical scenarios. Each pair consists of a "chosen" response (safe, grounded in documentation, appropriately escalates to professionals) and a "rejected" response (unsafe advice, hallucinated part names, missing safety warnings). This dataset can be partially generated using GPT-4 to create realistic preference pairs, then human-reviewed for quality. A target of 2,000-5,000 preference pairs is sufficient for effective DPO on a small model.

## 4. Chunking Strategy

The chunking strategy is designed around the natural structure of repair manuals rather than arbitrary token windows.

Each iFixit guide step becomes a single chunk, preserving the atomic unit of a repair instruction. Steps that exceed 512 tokens are split at sentence boundaries with a 50-token overlap to maintain context. Steps shorter than 100 tokens are merged with adjacent steps to avoid fragments that lack sufficient context for meaningful retrieval.

Every chunk is enriched with structured metadata: appliance category (e.g., "Washing Machine"), brand/model (e.g., "Samsung WF45R6100AW"), guide title, difficulty level, required tools, and step number within the guide. This metadata enables filtered retrieval: when a user specifies their appliance type, the search can be scoped to the relevant category before semantic matching begins.

For manufacturer PDFs and unstructured documents, a recursive character text splitter with 512-token chunks and 50-token overlap will be used, with metadata extracted from document titles, headers, and table of contents where available.

## 5. Vector Database Choice

ChromaDB is selected as the vector store for the following reasons. It is Python-native and integrates seamlessly with the FastAPI backend with no external service dependencies. It runs in-process, which is ideal for a local server setup and Colab-based development. It supports metadata filtering natively, enabling the scoped retrieval by appliance type described above. It persists to disk, allowing the knowledge base to survive server restarts.

The embedding model is BGE-small-en-v1.5 (33M parameters, 384 dimensions). It offers an excellent quality-to-size ratio, ranking competitively on MTEB benchmarks while being small enough to run inference quickly on CPU. Each document chunk is embedded at ingestion time and stored alongside its metadata in ChromaDB.

For reranking, a cross-encoder model (cross-encoder/ms-marco-MiniLM-L-6-v2) will be applied as a second-stage ranker on the top-k results from ChromaDB to improve precision before passing context to the agent.

## 6. Model Choices

### 6.1 Agent Orchestrator: Qwen 2.5 3B (Fine-Tuned)

Qwen 2.5 3B is the core SLM, chosen for several reasons. It has the best quality-to-size ratio for Colab T4 constraints (15GB VRAM). It is instruction-tuned out of the box with strong multilingual and reasoning capabilities. It is released under the Apache 2.0 license for unrestricted use. The 3B parameter count, when loaded in 4-bit quantization, requires only \~1.7GB of VRAM for the base model plus \~100MB for LoRA adapters, leaving ample headroom for inference and training on a free T4 GPU. [^6]​

### 6.2 Embedding Model: BGE-small-en-v1.5

A compact (33M parameter) embedding model from BAAI that produces 384-dimensional vectors. Selected for fast CPU inference and strong retrieval performance relative to its size.

### 6.3 Reranker: cross-encoder/ms-marco-MiniLM-L-6-v2

A cross-encoder reranker applied to the top-20 retrieved chunks to re-score and select the top-5 most relevant passages before they enter the agent's context window.

## 7. Agent Architecture (ReAct Loop)

The agent operates in a ReAct (Reasoning + Acting) loop. At each turn, the model receives the conversation history, reasons about what information it needs, selects a tool to call, observes the result, and then either calls another tool or generates a final response to the user.

The agent has access to the following tools:

Manual\_Search\_Tool: performs semantic search over the ChromaDB knowledge base, optionally filtered by appliance category. Returns the top-k relevant repair guide passages with source attribution.

Safety\_Protocol\_Checker: a rule-based + model-assisted tool that evaluates whether a repair task involves hazardous conditions (high voltage, gas lines, refrigerant, etc.). If triggered, it overrides the agent's response with a professional-referral warning.

Part\_Identifier: given a symptom description, retrieves likely faulty components from the knowledge base and returns part names, part numbers, and approximate cost ranges where available.

Symptom\_Clarifier: generates targeted follow-up questions when the user's initial description is ambiguous (e.g., "my washer is leaking" prompts questions about leak location, timing, water temperature, etc.).

Step\_By\_Step\_Guide\_Formatter: takes raw retrieved repair steps and formats them into a clean, numbered guide with tool requirements and safety notes for the user.

## 8. RAG Strategies

Three RAG approaches will be implemented and compared.

### 8.1 Naive RAG (Baseline)

The user query is embedded directly using BGE-small-en-v1.5, the top-k chunks are retrieved from ChromaDB via cosine similarity, and those chunks are injected into the prompt as context. This serves as the baseline for measuring improvements.

### 8.2 Reranked RAG

Same retrieval as naive RAG, but the top-20 results are passed through the cross-encoder reranker, which re-scores each query-chunk pair. Only the top-5 after reranking enter the agent's context. This typically improves precision significantly at minimal latency cost.

### 8.3 HyDE (Hypothetical Document Embeddings)

Before retrieval, the agent generates a hypothetical answer to the user's question. This hypothetical answer is then embedded and used as the retrieval query instead of the raw user question. The intuition is that the embedding of a well-formed answer will be closer in vector space to the actual relevant documents than a short user question. This is particularly valuable for troubleshooting queries where users describe symptoms rather than solutions.

### 8.4 Metadata-Filtered Retrieval

When the user specifies an appliance type or brand (detected via the agent's reasoning step), ChromaDB's metadata filter narrows the search space before semantic matching occurs. This eliminates irrelevant results from unrelated appliance categories.

## 9. Fine-Tuning Strategy

### 9.1 QLoRA Configuration

Fine-tuning uses QLoRA (Quantized Low-Rank Adaptation) via the Unsloth library, which provides 2x faster training and 60% less VRAM usage compared to standard HuggingFace implementations. [^7]​

Configuration details: the base model is loaded in 4-bit NormalFloat (NF4) quantization. LoRA adapters are applied to the attention projection layers (q\_proj, k\_proj, v\_proj, o\_proj) with rank r=16 and alpha=32. Gradient checkpointing is enabled to reduce memory usage. Training uses a batch size of 1 with gradient accumulation over 8 steps, a learning rate of 2e-4 with cosine annealing, and a maximum sequence length of 2048 tokens. Expected training time on a free Colab T4 is approximately 30-60 minutes for 100-200 training steps.

### 9.2 Fine-Tuning Stages

Stage 1 (SFT on tool-calling): the model is first fine-tuned on the Glaive Function Calling v2 dataset to learn structured JSON tool-call output format.

Stage 2 (Domain SFT): the model is further fine-tuned on the custom appliance troubleshooting dataset to specialize in repair domain reasoning and safety-aware responses.

Stage 3 (DPO alignment): preference optimization using the safety-focused preference dataset (see Section 3.3).

## 10. RLHF via Direct Preference Optimization (DPO)

DPO is chosen over PPO for alignment because PPO requires four models in memory simultaneously (policy, reference policy, reward model, and value network), totaling approximately 22GB of VRAM, which exceeds the T4's 15GB capacity. DPO requires only two models (the policy being trained and a frozen reference), fitting within approximately 15GB with QLoRA and gradient checkpointing. [^8]​

DPO achieves approximately 90-95% of RLHF/PPO quality with roughly 10% of the implementation complexity. [^9] It requires no separate reward model training, runs as a standard supervised training loop, and is stable across a wide range of learning rates and batch sizes.

The DPO training uses the TRL library's DPOTrainer with a beta parameter of 0.1, learning rate of 1e-5, and the same QLoRA configuration as the SFT stages. The preference dataset focuses on three safety dimensions: hazard detection (gas leaks, electrical risks, refrigerant handling), escalation behavior (knowing when to recommend a professional), and groundedness (preferring responses anchored in retrieved documentation over speculative answers).

## 11. Evaluation Framework

### 11.1 RAG Evaluation

Using the RAGAS framework to measure: faithfulness (are the agent's answers supported by retrieved context?), answer relevance (does the answer address the user's question?), and context relevance (are the retrieved passages actually relevant to the query?). A test set of 100-200 appliance troubleshooting questions with ground-truth answers will be constructed for benchmarking.

### 11.2 Agent Evaluation

Task completion rate: percentage of troubleshooting scenarios where the agent reaches a correct diagnosis and repair recommendation. Tool selection accuracy: whether the agent selects the appropriate tool at each reasoning step. Safety compliance rate: percentage of hazardous scenarios where the agent correctly triggers the Safety\_Protocol\_Checker.

### 11.3 Fine-Tuning Evaluation

Perplexity on a held-out test set of appliance repair conversations. Tool-call format accuracy: percentage of model outputs that produce valid, parseable JSON tool calls. A/B comparison of responses before and after DPO alignment, evaluated by human judges on helpfulness, safety, and groundedness.

### 11.4 Toxicity and Hallucination Handling

Hallucination mitigation is achieved through three layers: strict source grounding (the agent is prompted to answer only from retrieved context), confidence scoring (low-confidence responses trigger a fallback to "I don't have enough information" rather than speculation), and the DPO-trained preference for grounded over speculative responses. Toxicity is handled through a keyword-based safety filter for dangerous repair scenarios and the DPO-aligned safety escalation behavior.

## 12. Data Ingestion Pipeline

The system supports ingestion of new datasets through a modular pipeline. New documents (PDFs, HTML pages, plain text) are uploaded via the web UI or a CLI tool. The ingestion pipeline extracts text, applies the chunking strategy described in Section 4, generates embeddings using BGE-small-en-v1.5, enriches chunks with extracted metadata, and stores everything in ChromaDB. The pipeline is idempotent: re-ingesting the same document updates existing chunks rather than creating duplicates.

For iFixit data, a dedicated scraper module uses the iFixit API to fetch guides by category, transforms them into the chunked format, and loads them into the vector store. This scraper can be run periodically to capture new community-contributed guides.

## 13. Tech Stack Summary

Frontend: React with Next.js, styled with Tailwind CSS, deployed as a static build served by the FastAPI backend.
Backend: Python 3.11+, FastAPI, Uvicorn ASGI server.
ML/AI: HuggingFace Transformers, Unsloth (fine-tuning), TRL (DPO training), Sentence-Transformers (embeddings).
Vector Database: ChromaDB (persistent mode).
Persistence: SQLite (chat history, user sessions).
Development: Google Colab T4 (training), local machine (inference and development).

## 14. Timeline

Weeks 5-6 (current): architecture presentation, dataset selection finalization, environment setup, and team role assignment.

Weeks 7-8: data ingestion pipeline implementation (iFixit scraper, PDF loader, chunking, ChromaDB population). Basic FastAPI backend with naive RAG endpoint. Initial React frontend with chat interface. Begin QLoRA fine-tuning experiments on Colab.

Weeks 8-9 (progress check): present working RAG pipeline with naive retrieval. Demonstrate fine-tuned Qwen 2.5 3B with tool-calling capability. Show initial agent loop with at least 2 functional tools. Compare naive RAG vs. reranked RAG results.

Weeks 9-10: implement HyDE and metadata-filtered retrieval strategies. Construct and train on the DPO preference dataset. Build the evaluation framework (RAGAS integration, test set construction). Integrate safety guardrails and the Safety\_Protocol\_Checker tool.

Weeks 11-12 (final presentation): full end-to-end demo with all tools functional. Comparative evaluation results across RAG strategies. Documentation of architecture, fine-tuning process, and evaluation metrics. Source code repository with setup instructions and reproducibility guide.

---

## References

[^1]:  "RFC_DIYAssist_Agentic_RAG_for_Appliance_Trouble,", n.d..

[^2]:  "MyFixit Dataset - GitHub,", n.d.. Available: [https://github.com/rub-ksv/MyFixit-Dataset](https://github.com/rub-ksv/MyFixit-Dataset)

[^3]: "iFixit API v2.0 Documentation,", n.d.. Available: [https://www.ifixit.com/api/2.0/doc/Guides](https://www.ifixit.com/api/2.0/doc/Guides)

[^4]:  "spare-part-replacement-notes-v1 on HuggingFace,", n.d.. Available: [https://huggingface.co/datasets/deepakkumar07/spare-part-replacement-notes-v1/viewer/](https://huggingface.co/datasets/deepakkumar07/spare-part-replacement-notes-v1/viewer/)

[^5]:  "maintenance_gpt_added on HuggingFace,", n.d.. Available: [https://huggingface.co/datasets/mandarchaudharii/maintenance_gpt_added/viewer](https://huggingface.co/datasets/mandarchaudharii/maintenance_gpt_added/viewer)

[^6]:  "Fine-Tuning a 3B Model for Function Calling with QLoRA,", n.d.. Available: [https://datahacker.rs/llm_log-015-fine-tuning-llms-teach-a-3b-model-to-call-functions-with-qlora-unsloth-on-free-colab-t4/](https://datahacker.rs/llm_log-015-fine-tuning-llms-teach-a-3b-model-to-call-functions-with-qlora-unsloth-on-free-colab-t4/)

[^7]: "Fine-Tune Any LLM for Free on Colab T4,", n.d.. Available: [https://docs.bswen.com/blog/2026-03-21-free-llm-finetuning-colab](https://docs.bswen.com/blog/2026-03-21-free-llm-finetuning-colab)

[^8]: "DPO vs RLHF Hardware Requirements,", n.d.. Available: [https://agentfactory.panaversity.org/docs/Turing-LLMOps-Proprietary-Intelligence/alignment-safety/dpo-vs-rlhf](https://agentfactory.panaversity.org/docs/Turing-LLMOps-Proprietary-Intelligence/alignment-safety/dpo-vs-rlhf)

[^9]: "RLHF vs DPO vs PPO Comparison,", n.d.. Available: [https://mljourney.com/rlhf-vs-dpo-vs-ppo-how-to-align-llms-without-losing-your-mind/](https://mljourney.com/rlhf-vs-dpo-vs-ppo-how-to-align-llms-without-losing-your-mind/)
