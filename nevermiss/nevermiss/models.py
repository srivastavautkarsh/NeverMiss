"""Typed state for the whole system. Pydantic so every agent boundary is schema-checked."""
from __future__ import annotations
from datetime import datetime
from typing import List, Literal, Optional
from pydantic import BaseModel, Field


class AgentConfig(BaseModel):
    """What a person's agent is allowed to say about them. This is the 'configuration'
    two agents exchange after a confirmed handshake. Nothing outside it is ever shared."""
    user_id: str
    name: str
    headline: str                              # "Director of New Business, KOPPER"
    offers: List[str] = Field(default_factory=list)   # what they can help with
    needs: List[str] = Field(default_factory=list)    # what they're looking for
    share_level: Literal["public", "professional", "private"] = "professional"
    contact: Optional[str] = None              # only released on a strong match


class HandshakeEvent(BaseModel):
    """Emitted by the wearable layer (LSTM gesture model / OpenCV face detect)."""
    event_id: str
    timestamp: datetime
    me: str                                    # user_id of the wearer
    other: str                                 # user_id resolved via proximity / BLE / face
    confidence: float                          # gesture-model confidence 0..1
    location: Optional[str] = None
    context_note: Optional[str] = None         # "AI club mixer, Hofstra"


class MatchScore(BaseModel):
    """Structured output the LLM must produce. Numbers here are checked by code."""
    score: int = Field(ge=0, le=100)
    reasons: List[str]
    best_offer_to_need: Optional[str] = None   # "their AWS expertise <-> your cloud shipping need"
    suggested_followup: str


class Encounter(BaseModel):
    """Episodic memory record. This is what 'who did I meet who knows AWS?' searches."""
    event_id: str
    timestamp: datetime
    me: str
    other: AgentConfig
    shared_from_me: List[str]                  # what MY agent agreed to disclose
    shared_from_them: List[str]
    match: MatchScore
    notified: bool
    context_note: Optional[str] = None
