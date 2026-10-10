"""One place to pick the model. Same graph runs on Claude, a local Ollama model, or a
deterministic stub for tests - which is how the pipeline is unit-tested without a GPU."""
from __future__ import annotations
import json, os
from typing import List
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage


def get_llm(provider: str | None = None, kind: str = "match"):
    provider = provider or os.getenv("NEVERMISS_LLM", "stub")
    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(model=os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-5"), temperature=0)
    if provider == "ollama":
        from langchain_ollama import ChatOllama
        return ChatOllama(model=os.getenv("OLLAMA_MODEL", "llama3.1:8b"), temperature=0, format="json")
    return stub_llm(kind)


def stub_llm(kind: str = "match"):
    """Canned, schema-valid answers, one stub per chain so shapes never cross."""
    responses = {"match": [
        {"score": 82, "reasons": ["their 'AWS architecture' offer matches your 'ship on a cloud provider' need",
                                  "both in enterprise SaaS"],
         "best_offer_to_need": "AWS architecture <-> cloud deployment",
         "suggested_followup": "Ask for a 20-min call about EKS vs ECS for your launch."}],
        "recall": [
        {"ranked_event_ids": ["evt-001", "evt-999"], "answer": "You met Priya Natarajan at the Hofstra AI mixer; she leads AWS solutions and offered architecture help."}],
    }[kind]
    return FakeMessagesListChatModel(responses=[AIMessage(content=json.dumps(r)) for r in responses] * 20)
