"""Disclosure negotiation between the two personal agents.

Two CrewAI agents, one per person, each with the goal of sharing *enough* to find a useful
match while staying inside that person's share_level. The crew's output is a proposal; the
deterministic policy in `apply_policy` always has the last word, so an over-eager LLM can
never leak a field the user marked private. If CrewAI/LLM is unavailable the policy alone runs.
"""
from __future__ import annotations
import json, os
from typing import List, Tuple
from .models import AgentConfig

POLICY = {"public": lambda c: c.offers + c.needs,
          "professional": lambda c: c.offers,
          "private": lambda c: c.offers[:1]}          # a private person still shares one hook


def apply_policy(cfg: AgentConfig, proposed: List[str]) -> List[str]:
    allowed = set(POLICY[cfg.share_level](cfg))
    return [p for p in proposed if p in allowed] or list(allowed)[:1]


def negotiate_disclosure(me: AgentConfig, them: AgentConfig, use_crew: bool = True) -> Tuple[List[str], List[str]]:
    mine_default, theirs_default = POLICY[me.share_level](me), POLICY[them.share_level](them)
    if not use_crew or os.getenv("NEVERMISS_LLM", "stub") == "stub":
        return mine_default, theirs_default
    try:
        from crewai import Agent, Crew, Task
        my_agent = Agent(role=f"Personal agent for {me.name}",
                         goal="Propose which of my principal's offers/needs to disclose so the other side can judge fit, "
                              "never exceeding the share policy.",
                         backstory=f"Represents {me.headline}. Share level: {me.share_level}.", verbose=False)
        their_agent = Agent(role=f"Personal agent for {them.name}",
                            goal="Same, for my own principal.", backstory=f"Represents {them.headline}. Share level: {them.share_level}.",
                            verbose=False)
        t1 = Task(description=f"My principal's offers: {me.offers}. Needs: {me.needs}. Counterpart headline: {them.headline}. "
                              f"Return ONLY a JSON list of the strings to disclose.", expected_output="JSON list of strings", agent=my_agent)
        t2 = Task(description=f"My principal's offers: {them.offers}. Needs: {them.needs}. Counterpart headline: {me.headline}. "
                              f"Return ONLY a JSON list of the strings to disclose.", expected_output="JSON list of strings", agent=their_agent)
        out = Crew(agents=[my_agent, their_agent], tasks=[t1, t2], verbose=False).kickoff()
        proposals = [json.loads(str(t.raw)) for t in out.tasks_output]
        return apply_policy(me, proposals[0]), apply_policy(them, proposals[1])
    except Exception:          # any crew/LLM failure degrades to the deterministic policy
        return mine_default, theirs_default
