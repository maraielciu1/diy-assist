# Stage 5 Demo Script

## Setup

1. Start the backend:

```bash
make run-backend
```

2. Start the Next.js frontend:

```bash
make frontend-install
make run-frontend
```

3. Open `http://127.0.0.1:3000`.

## Presentation Flow

1. Show the baseline retrieval endpoint in `/docs` and mention that naive, reranked, HyDE, and HyDE + reranked routes remain available.
2. Open the frontend and submit: `My washer is not draining and makes a humming noise.`
3. Point out the structured sections: summary, likely issue, repair guidance, parts to inspect, snippets, and citations.
4. Submit the ambiguous query: `My washer is leaking.`
5. Point out that the agent asks for a clarifying detail instead of guessing.
6. Submit the hazardous query: `I smell gas near my dryer.`
7. Point out that the response escalates to a professional and does not show repair steps.
8. Submit a follow-up in the same session and show that the backend returns the same session id.
9. Run the evaluation command:

```bash
make evaluate-rag
```

10. Open `data/eval/report_stage5.json` and summarize retrieval hit rate, guardrail behavior, and runtime.

## Known Limitations To Mention

- The seed fine-tuning datasets are format examples, not production-sized training corpora.
- Local SLM quality depends on the configured LM Studio model.
- Live iFixit lookup depends on network availability and can be disabled with `USE_IFIXIT_LIVE_LOOKUP=false`.
- The Next.js frontend calls the FastAPI backend directly during local development.
