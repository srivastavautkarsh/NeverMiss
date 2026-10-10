"""python -m pytest tests/ -q   (stub LLM; no model, no network)"""
import os, sys, tempfile
from datetime import datetime
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from nevermiss import AgentConfig, HandshakeEvent, EpisodicMemory, MatchEngine
from nevermiss.negotiation_crew import apply_policy
from nevermiss.cli import DIRECTORY

def engine():
    mem = EpisodicMemory(path=os.path.join(tempfile.mkdtemp(), "enc.jsonl"))
    return MatchEngine(DIRECTORY, mem, use_crew=False), mem

def ev(other, conf=0.95, eid="e1"):
    return HandshakeEvent(event_id=eid, timestamp=datetime(2026,9,24,18), me="utkarsh", other=other, confidence=conf, context_note="mixer")

def test_strong_match_notifies_and_persists():
    eng, mem = engine()
    out = eng.run(ev("priya"))
    assert out["notified"] and out["match"].score >= 70
    assert mem.all()[0].other.name == "Priya Natarajan"

def test_grounding_guardrail_blocks_ungrounded_score():
    eng, _ = engine()
    out = eng.run(ev("dan"))                      # stub LLM says 82, but cites nothing Dan disclosed
    assert out["notified"] is False and out["match"].score < 70

def test_low_gesture_confidence_is_dropped_before_any_llm_call():
    eng, mem = engine()
    out = eng.run(ev("priya", conf=0.3))
    assert out["rejected_reason"].startswith("gesture confidence") and mem.all() == []

def test_private_share_policy_caps_disclosure():
    dan = DIRECTORY["dan"]
    assert apply_policy(dan, ["commercial leasing", "tenants for Hempstead office space"]) == ["commercial leasing"]

def test_recall_filters_invented_event_ids_and_gates_contact():
    eng, mem = engine()
    eng.run(ev("priya", eid="evt-001")); eng.run(ev("dan", eid="evt-002"))
    out = mem.recall("who can help me ship on AWS")
    assert out["ranked_event_ids"] == ["evt-001"]               # stub returned evt-999 too; dropped
    assert out["people"][0]["contact"] == "priya@example.com"   # released only because match notified
