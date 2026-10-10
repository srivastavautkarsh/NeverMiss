"""The match engine: a LangGraph StateGraph driven by LangChain prompt chains.

    handshake ──► load_configs ──► negotiate_share (CrewAI) ──► score_match (LLM chain)
                                                                      │
                                              ┌───── score >= threshold ─────┐
                                              ▼                              ▼
                                           notify                         store_only
                                              └──────────► persist ◄─────────┘

Why a graph and not a straight chain: the notify/store decision is a conditional edge,
the whole run is checkpointed (resumable if the phone loses signal mid-handshake), and
each node is independently testable with the stub LLM.
"""
from __future__ import annotations

import json
from typing import Dict, List, Optional, TypedDict

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from .llm import get_llm
from .memory import EpisodicMemory
from .models import AgentConfig, Encounter, HandshakeEvent, MatchScore
from .negotiation_crew import negotiate_disclosure

NOTIFY_THRESHOLD = 70
MIN_GESTURE_CONFIDENCE = 0.80


class MatchState(TypedDict, total=False):
    event: HandshakeEvent
    my_config: AgentConfig
    their_config: AgentConfig
    shared_from_me: List[str]
    shared_from_them: List[str]
    match: MatchScore
    notified: bool
    rejected_reason: Optional[str]
    encounter: Encounter


# ---------------------------------------------------------------- LangChain chain
_SCORE_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a networking agent. Compare what two people OFFER and NEED and decide whether "
     "they should follow up. Return ONLY JSON with keys: score (0-100 integer), reasons (list of "
     "short strings, each citing an actual offer/need phrase), best_offer_to_need (string or null), "
     "suggested_followup (one sentence). Use only the disclosed fields; never invent facts."),
    ("human",
     "ME:\n  headline: {my_headline}\n  needs: {my_needs}\n  offers: {my_offers}\n\n"
     "THEM (disclosed):\n  headline: {their_headline}\n  offers: {their_offers}\n  needs: {their_needs}\n\n"
     "Context: {context}"),
])


def build_score_chain(llm=None):
    """prompt | llm | parser — the LangChain LCEL chain the graph calls."""
    return _SCORE_PROMPT | (llm or get_llm()) | JsonOutputParser()


# ---------------------------------------------------------------- graph nodes
class MatchEngine:
    def __init__(self, directory: Dict[str, AgentConfig], memory: EpisodicMemory, llm=None,
                 use_crew: bool = True):
        self.directory, self.memory = directory, memory
        self.score_chain = build_score_chain(llm)
        self.use_crew = use_crew
        self.graph = self._build()

    # node 1: did the wearable really see a handshake, and do we know who it was?
    def load_configs(self, s: MatchState) -> MatchState:
        ev = s["event"]
        if ev.confidence < MIN_GESTURE_CONFIDENCE:
            return {"rejected_reason": f"gesture confidence {ev.confidence:.2f} < {MIN_GESTURE_CONFIDENCE}"}
        if ev.other not in self.directory:
            return {"rejected_reason": f"unknown counterpart {ev.other}"}
        return {"my_config": self.directory[ev.me], "their_config": self.directory[ev.other]}

    # node 2: two agents agree what each side discloses (CrewAI), bounded by share_level
    def negotiate_share(self, s: MatchState) -> MatchState:
        mine, theirs = negotiate_disclosure(s["my_config"], s["their_config"], use_crew=self.use_crew)
        return {"shared_from_me": mine, "shared_from_them": theirs}

    # node 3: LLM scores the match from DISCLOSED fields only; code validates the output
    def score_match(self, s: MatchState) -> MatchState:
        me, them, ev = s["my_config"], s["their_config"], s["event"]
        raw = self.score_chain.invoke({
            "my_headline": me.headline, "my_needs": me.needs, "my_offers": me.offers,
            "their_headline": them.headline, "their_offers": s["shared_from_them"],
            "their_needs": them.needs if them.share_level != "private" else [],
            "context": ev.context_note or "none",
        })
        match = MatchScore(**raw)                       # schema guardrail: bad JSON -> exception, not a silent pass
        # grounding guardrail: every reason must cite something the COUNTERPART actually disclosed.
        # (Citing only my own needs is how a model talks itself into a match with anyone.)
        their_disclosed = " ".join(s["shared_from_them"] + (them.needs if them.share_level != "private" else [])).lower()
        match.reasons = [r for r in match.reasons if any(tok in their_disclosed for tok in _key_tokens(r))]
        if not match.reasons:
            match.score = min(match.score, NOTIFY_THRESHOLD - 1)   # ungrounded -> never notify
        return {"match": match}

    def route(self, s: MatchState) -> str:
        if s.get("rejected_reason"):
            return "drop"
        return "notify" if s["match"].score >= NOTIFY_THRESHOLD else "store_only"

    def notify(self, s: MatchState) -> MatchState:
        return {"notified": True}

    def store_only(self, s: MatchState) -> MatchState:
        return {"notified": False}

    def persist(self, s: MatchState) -> MatchState:
        ev = s["event"]
        enc = Encounter(event_id=ev.event_id, timestamp=ev.timestamp, me=ev.me, other=s["their_config"],
                        shared_from_me=s["shared_from_me"], shared_from_them=s["shared_from_them"],
                        match=s["match"], notified=s["notified"], context_note=ev.context_note)
        self.memory.add(enc)
        return {"encounter": enc}

    def _build(self):
        g = StateGraph(MatchState)
        for name in ("load_configs", "negotiate_share", "score_match", "notify", "store_only", "persist"):
            g.add_node(name, getattr(self, name))
        g.set_entry_point("load_configs")
        g.add_conditional_edges("load_configs", lambda s: "drop" if s.get("rejected_reason") else "ok",
                                {"drop": END, "ok": "negotiate_share"})
        g.add_edge("negotiate_share", "score_match")
        g.add_conditional_edges("score_match", self.route, {"notify": "notify", "store_only": "store_only", "drop": END})
        g.add_edge("notify", "persist")
        g.add_edge("store_only", "persist")
        g.add_edge("persist", END)
        return g.compile(checkpointer=MemorySaver())

    def run(self, event: HandshakeEvent) -> MatchState:
        # thread_id = event_id so a retried handshake resumes from its checkpoint instead of re-scoring
        return self.graph.invoke({"event": event}, config={"configurable": {"thread_id": event.event_id}})


def _key_tokens(text: str) -> List[str]:
    return [t.strip("'\".,()") for t in text.lower().split() if len(t) > 3]
