"""Demo: two handshakes at a mixer, then recall two weeks later.
   NEVERMISS_LLM=stub python -m nevermiss.cli        (no model needed)
   NEVERMISS_LLM=ollama python -m nevermiss.cli      (local llama3.1 via Ollama)
   NEVERMISS_LLM=anthropic python -m nevermiss.cli   (needs ANTHROPIC_API_KEY)"""
import json, os, sys
from datetime import datetime
from nevermiss import AgentConfig, HandshakeEvent, EpisodicMemory, MatchEngine

DIRECTORY = {
    "utkarsh": AgentConfig(user_id="utkarsh", name="Utkarsh Srivastava", headline="Founder, Gest; MBA Finance+AI",
                           offers=["LLM pipelines with guardrails", "FedRAMP/SOC2 automation"],
                           needs=["ship on a cloud provider", "design partner for wearables"], share_level="professional",
                           contact="utkarsh@example.com"),
    "priya": AgentConfig(user_id="priya", name="Priya Natarajan", headline="Solutions Architect lead, AWS",
                         offers=["AWS architecture", "EKS and serverless design reviews"],
                         needs=["startups building on Bedrock"], share_level="professional", contact="priya@example.com"),
    "dan": AgentConfig(user_id="dan", name="Dan Okafor", headline="Partner, real-estate law",
                       offers=["commercial leasing"], needs=["tenants for Hempstead office space"], share_level="private"),
}

def main():
    if os.path.exists("data/encounters.jsonl"):
        os.remove("data/encounters.jsonl")
    mem = EpisodicMemory()
    engine = MatchEngine(DIRECTORY, mem)
    events = [
        HandshakeEvent(event_id="evt-001", timestamp=datetime(2026, 9, 24, 18, 5), me="utkarsh", other="priya",
                       confidence=0.93, location="Hofstra", context_note="AI club mixer"),
        HandshakeEvent(event_id="evt-002", timestamp=datetime(2026, 9, 24, 18, 40), me="utkarsh", other="dan",
                       confidence=0.91, location="Hofstra", context_note="AI club mixer"),
        HandshakeEvent(event_id="evt-003", timestamp=datetime(2026, 9, 24, 19, 0), me="utkarsh", other="priya",
                       confidence=0.41, context_note="probably a wave, not a handshake"),
    ]
    for ev in events:
        out = engine.run(ev)
        if out.get("rejected_reason"):
            print(f"{ev.event_id}: DROPPED ({out['rejected_reason']})")
        else:
            m = out["match"]
            print(f"{ev.event_id}: {out['their_config'].name:18s} score={m.score:3d} notify={out['notified']}  "
                  f"disclosed_by_them={out['shared_from_them']}")
    print("\n--- two weeks later ---")
    print(json.dumps(mem.recall("Who did I meet who can help me ship on AWS?"), indent=1, default=str))

if __name__ == "__main__":
    sys.exit(main())
