"""Episodic memory + natural-language recall ("who did I meet who knows AWS?").

Storage is a JSONL file (swap for Mem0 / pgvector later; the interface is add/search/recall).
Recall = lexical pre-filter -> LangChain rerank chain -> grounded answer that cites event_ids.
"""
from __future__ import annotations
import json, os
from typing import List
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from .llm import get_llm
from .models import Encounter

_RECALL_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "You answer questions about people the user has met, using ONLY the encounter records given. "
               "Return JSON: ranked_event_ids (most relevant first) and answer (2 sentences, name the person and "
               "what they offered). If nothing matches, return an empty list and say so."),
    ("human", "Question: {question}\n\nEncounters:\n{records}"),
])


class EpisodicMemory:
    def __init__(self, path: str = "data/encounters.jsonl", llm=None):
        self.path = path
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.chain = _RECALL_PROMPT | (llm or get_llm(kind="recall")) | JsonOutputParser()

    def add(self, enc: Encounter) -> None:
        with open(self.path, "a") as f:
            f.write(enc.model_dump_json() + "\n")

    def all(self) -> List[Encounter]:
        if not os.path.exists(self.path):
            return []
        return [Encounter.model_validate_json(l) for l in open(self.path) if l.strip()]

    def search(self, query: str, k: int = 10) -> List[Encounter]:
        q = {t for t in query.lower().split() if len(t) > 2}
        def hit(e: Encounter) -> int:
            blob = (e.other.headline + " " + " ".join(e.shared_from_them) + " " + (e.context_note or "")).lower()
            return sum(t in blob for t in q)
        return sorted((e for e in self.all() if hit(e) > 0), key=hit, reverse=True)[:k]

    def recall(self, question: str) -> dict:
        cands = self.search(question) or self.all()[-10:]
        records = "\n".join(f"- {e.event_id} | {e.timestamp:%Y-%m-%d} | {e.other.name}, {e.other.headline} | "
                            f"offered: {e.shared_from_them} | where: {e.context_note}" for e in cands)
        out = self.chain.invoke({"question": question, "records": records or "(none)"})
        valid = {e.event_id for e in cands}
        out["ranked_event_ids"] = [i for i in out.get("ranked_event_ids", []) if i in valid]   # no invented ids
        out["people"] = [{"name": e.other.name, "contact": e.other.contact if e.notified else None}
                         for e in cands if e.event_id in out["ranked_event_ids"]]
        return out
