# NeverMiss — "never miss a connection"

Ambient networking agent. A wearable detects a handshake; each person's agent exchanges a
policy-bounded **configuration** (offers / needs); a match engine scores fit and notifies only
when it's grounded; everything is stored as episodic memory you can query in plain English
weeks later ("who did I meet who knows AWS?").

Originally scoped as a Hofstra IT 257K project (with Sanjana Choudankar); the gesture layer
(YOLO + LSTM, 85%+ on a 600-image set) exists separately. **This repo is the agent layer.**

## Stack and where each piece lives

| Piece | Framework | File |
|---|---|---|
| Match engine: handshake → configs → negotiate → score → notify/store → persist, checkpointed, conditional edges | **LangGraph** `StateGraph` + `MemorySaver` | `nevermiss/match_graph.py` |
| Scoring and recall prompts as composable chains (`prompt \| llm \| JsonOutputParser`) | **LangChain** (LCEL) | `match_graph.py`, `memory.py` |
| Two personal agents negotiating what to disclose, capped by a deterministic share policy | **CrewAI** (2 agents, 2 tasks) | `nevermiss/negotiation_crew.py` |
| Typed state at every boundary | Pydantic | `nevermiss/models.py` |
| Episodic memory + recall (lexical prefilter → LLM rerank → grounded answer) | JSONL now; Mem0/pgvector later | `nevermiss/memory.py` |
| Model switch: Claude / Ollama / deterministic stub | `langchain-anthropic`, `langchain-ollama` | `nevermiss/llm.py` |

## Guardrails (all deterministic, all tested)
- Gesture confidence < 0.80 → dropped before any LLM call.
- Disclosure policy has the last word over the CrewAI proposal: `private` users never leak needs.
- LLM match output must validate against `MatchScore`; **every reason must cite something the counterpart disclosed**, or the score is capped below the notify threshold.
- Recall answers may only reference real `event_id`s; contact info is released only for matches that were notified.

## Run
```bash
pip install -r requirements.txt
NEVERMISS_LLM=stub   python -m nevermiss.cli      # deterministic demo
NEVERMISS_LLM=ollama python -m nevermiss.cli      # ollama pull llama3.1:8b
NEVERMISS_LLM=anthropic python -m nevermiss.cli   # export ANTHROPIC_API_KEY=...
python -m pytest tests/ -q
```
