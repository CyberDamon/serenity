"""News pre-filter — tags each NewsItem before it enters the framework prompts.

Runs zero LLM calls (pure regex on title + summary). Fast enough to run on
every assemble_news_context() call without meaningful latency overhead.

Tags inform framework prompts so each theory's LLM call gets structured
context about what kind of events the news represents, rather than having to
re-extract this from raw prose.

Tag catalogue
─────────────
Inspired by Buzan securitization theory + Polanyi double movement, extended
to cover the most actionable signals for all 8 IR frameworks:

  securitization_signal     — government frames issue as existential/emergency threat
  de_securitization_signal  — government explicitly walks back emergency framing
  social_protection_backlash — workers, unions, anti-globalization mobilization
  escalation_signal         — military buildup, hostile action, troops deployed
  de_escalation_signal      — ceasefire offer, withdrawal, diplomatic opening
  diplomatic_breakthrough   — agreement signed or talks producing concrete progress
  diplomatic_breakdown      — talks collapsed, ambassador expelled, treaty violation
  elite_defection_signal    — senior officials resigning or distancing from leader
  credible_commitment       — costly action (deployment, sanction, broken alliance)
  bluff_signal              — statement/warning with no costly follow-through
  economic_coercion         — sanctions, tariffs, export bans used as leverage
  regime_stress             — protests, economic crisis, legitimacy challenge
"""

from __future__ import annotations

import re

from serenity.data.news.types import NewsItem

# ─────────────────────────────────────────────────────────────────────────────
# Pattern sets (title + summary, case-insensitive)
# Each entry: (tag_name, list_of_regex_patterns)
# First pattern that matches adds the tag; multiple tags can apply.
# ─────────────────────────────────────────────────────────────────────────────

_TAG_RULES: list[tuple[str, list[str]]] = [

    # ── Buzan: securitization ────────────────────────────────────────────────
    ("securitization_signal", [
        r"\b(existential threat|state of emergency|national emergency)\b",
        r"\b(unprecedented (threat|danger|crisis)|red line|casus belli)\b",
        r"\b(invoke|invok(ed|ing)) (article 5|emergency powers|war powers)\b",
        r"\b(full (military )?mobilization|wartime (economy|footing))\b",
        r"\b(security (threat|crisis|emergency))\b.{0,40}\b(declare|declared|announce)\b",
        r"\bnational security.{0,30}(threat|crisis|emergency)\b",
        r"\b(military readiness|combat readiness|highest alert|defcon)\b",
    ]),

    # ── Buzan: de-securitization ─────────────────────────────────────────────
    ("de_securitization_signal", [
        r"\b(lift(ed|ing)|end(ed|ing)|terminat(ed|ing)) (emergency|sanctions|alert)\b",
        r"\b(normalize|normaliz(ed|ing)) relations\b",
        r"\bstand(ing|s)? down\b",
        r"\b(de-escalat|deescalat)(e|ed|ing|ion)\b",
        r"\b(confidence.building|csbm|hotline|direct talks)\b",
    ]),

    # ── Polanyi: social protection backlash ──────────────────────────────────
    ("social_protection_backlash", [
        r"\b(general strike|labor strike|workers.{0,10}(protest|rally|walk.?out))\b",
        r"\b(anti.?immigrant|anti.?globali[sz]|anti.?trade)\b",
        r"\btariffs?.{0,30}(demand|push|call|vote|workers|unions?|protect)\b",
        r"\b(protectionist|import (duty|ban)|trade (war|barrier))\b",
        r"\b(yellow vest|gilets jaunes|populist (surge|wave|victory))\b",
        r"\b(workers.{0,15}(rights|union|anger)|union (strike|action|walkout))\b",
        r"\b(economic nationalism|buy (local|american|british|chinese))\b",
        r"\b(job loss(es)?|factory clos(ure|ing)|deindustriali[sz])\b.{0,30}\b(protest|anger|blame)\b",
    ]),

    # ── Military escalation ───────────────────────────────────────────────────
    ("escalation_signal", [
        r"\b(troop(s)?|forces?|soldiers?|troops?).{0,20}(deploy|mass|cross|enter|invade|advance)\b",
        r"\b(deploy|dispatch|send|move).{0,30}\b(carrier|warship|troops?|forces?|battalion|regiment)\b",
        r"\b(military (buildup|build-up|operation|offensive|invasion))\b",
        r"\b(airstrike|missile (strike|attack|launch)|artillery (fire|barrage))\b",
        r"\b(forces? (advance|push|cross|enter|seize|capture))\b",
        r"\b(naval (blockade|clash|confrontation)|warship(s)? (enter|approach))\b",
        r"\b(shot (fired|down)|killed|casualties|dead)\b.{0,30}\b(soldier|troops?|military)\b",
        r"\b(cyber.?attack|hack(ed|ing)).{0,30}\b(government|infrastructure|military)\b",
        r"\bcarrier.{0,20}(strike group|group|fleet).{0,30}(deploy|south china|taiwan|persian)\b",
    ]),

    # ── De-escalation / ceasefire ─────────────────────────────────────────────
    ("de_escalation_signal", [
        r"\b(ceasefire|cease.fire|truce|armistice)\b.{0,40}\b(agree|sign|hold|broker)\b",
        r"\b(troops? (withdraw|pull back|retreat)|military (withdrawal|pullout))\b",
        r"\b(prisoner.{0,10}(swap|exchange|release))\b",
        r"\b(peace (talks?|negotiation|process))\b.{0,30}\b(resume|restart|progress|advance)\b",
        r"\b(humanitarian corridor|safe passage)\b",
    ]),

    # ── Diplomatic breakthrough ───────────────────────────────────────────────
    ("diplomatic_breakthrough", [
        r"\b(agreement|accord|deal|treaty|pact)\b.{0,40}\b(sign(ed)?|reach(ed)?|finaliz(ed)?)\b",
        r"\b(joint (statement|communiqué|declaration))\b",
        r"\bhistoric.{0,20}\b(deal|agreement|accord|meeting)\b",
        r"\b(summit|talks|negotiations)\b.{0,40}\b(succeed|progress|breakthrough|produc)\b",
        r"\b(ambassador|envoy|diplomat).{0,30}\b(return|appoint|meet)\b",
        r"\bnormali[sz](ation|ing|ed) (of )?(relations|ties)\b",
    ]),

    # ── Diplomatic breakdown ──────────────────────────────────────────────────
    ("diplomatic_breakdown", [
        r"\b(talks?|negotiations?|summit)\b.{0,40}\b(collapse|fail|break.?down|stall|suspend)\b",
        r"\b(ambassador|diplomat).{0,20}\b(expel|recall|summoned|expelled|recalled)\b",
        r"\b(sanction(s)?|embargo).{0,30}\b(impose|expand|new|additional)\b",
        r"\b(withdraw(n|ing)? from|pull(ing)? out of).{0,30}\b(treaty|agreement|deal|accord)\b",
        r"\b(relations? (sever|cut|downgrade|suspend))\b",
        r"\bdeclare(d|s)?.{0,20}(persona non grata|ambassador|diplomat)\b",
    ]),

    # ── Selectorate: elite defection ─────────────────────────────────────────
    ("elite_defection_signal", [
        r"\b(minister|general|official|chief|secretary|commander|director)\b.{0,20}resign(s|ed|ing|ation)?\b",
        r"\b(resign(s|ed|ation|ing)|step(s|ped|ping) down|quit(s|ting)?)\b.{0,40}\b(minister|general|official|chief|secretary)\b",
        r"\b(senior|top|key|cabinet|defense|foreign|finance|interior).{0,20}(official|minister|aide|advisor).{0,20}(resign|leave|quit|flee|defect)\b",
        r"\b(defect(ed|ion|s)|flee(d|s|ing)) (to the west|abroad|to (us|eu|uk|nato))\b",
        r"\b(disavow|distance(d|s)? (him|her|them)selves?|publicly criticize).{0,30}(leader|president|government)\b",
        r"\b(inner circle|close ally|loyalist).{0,20}(turn(ed|s)?|flip(ped|s)?|against|betray)\b",
        r"\b(assets? (abroad|offshore|frozen)|foreign account).{0,20}(elite|official|oligarch)\b",
        r"\bciting disagreement.{0,30}(resign|quit|step down|leave)\b",
    ]),

    # ── Schelling: credible commitment (costly signal) ───────────────────────
    ("credible_commitment", [
        r"\b(deploy(s|ed|ing)?|dispatch(es|ed|ing)?|send(s|ing)?|move(s|d|ing)?).{0,30}\b(carrier|warship|troops?|forces?|battalion|regiment|fighter)\b",
        r"\b(troops?|forces?|warship|carrier|aircraft).{0,20}(deploy(s|ed|ing)?|station(s|ed|ing)?)\b",
        r"\b(sanctions?).{0,20}(impos(ed|es|ing)?|activated|trigger(ed|s)?)\b",
        r"\b(military (exercise|drill|maneuver)).{0,20}(launch|begin|start|conduct)\b",
        r"\b(treaty|mutual defense|article 5).{0,20}(invoke(d|s)?|trigger(ed|s)?|activate(d|s)?)\b",
        r"\b(nuclear (alert|readiness|posture)).{0,20}(raise(d|s)?|elevate(d|s)?|increase(d|s)?)\b",
        r"\b(cut (off|ties)|sever(ed|s)?) (diplomatic|trade|financial) (relations|ties|links)\b",
        r"\b(expelled?|recalled?).{0,20}(ambassador|diplomat|envoy)\b",
    ]),

    # ── Schelling: bluff signal (cheap talk) ─────────────────────────────────
    ("bluff_signal", [
        r"\b(warn(s|ed|ing)|caution(s|ed|ing)|threaten(s|ed|ing))\b.{0,80}\b(statement|press|briefing|said|told reporters)\b",
        r"\bwarn(s|ed|ing)?.{0,30}(consequences|repercussions|action)\b",
        r"\b(strongly (condemn(s|ed)?|criticize(s|d)?|protest(s|ed)?|object(s|ed)?))\b",
        r"\b(red line|not rule out|reserve(s)? the right).{0,40}\b(said|stated|warned|told)\b",
        r"\b(call(s|ed|ing) (on|for)|demand(s|ed|ing)).{0,40}\b(must|should|immediate|stop|halt)\b",
        r"\b(united nations|un security council) (resolution|statement|call)\b",
        r"\bbut (takes?|took) no (action|step)\b",
        r"\b(verbal|rhetorical|diplomatic) (warning|threat|pressure)\b",
    ]),

    # ── Economic coercion ─────────────────────────────────────────────────────
    ("economic_coercion", [
        r"\b(sanction(s)?|embargo|asset freeze|travel ban)\b.{0,30}\b(target|hit|impos)\b",
        r"\b(tariff(s)?|import (duty|tax|levy)).{0,30}\b(impos|raise|hike|new|additional)\b",
        r"\b(export (control|ban|restriction)).{0,30}\b(chip|semiconductor|technology|weapon)\b",
        r"\b(cut off|block(ed)?|deny).{0,30}\b(swift|payment|financial|dollar|oil|gas|energy)\b",
        r"\b(economic (weapon|warfare|pressure|coercion))\b",
    ]),

    # ── Regime stress (Bremmer J-curve + BdM) ───────────────────────────────
    ("regime_stress", [
        r"\b(mass (protest|demonstration|rally|unrest)).{0,30}\b(demand|call|against)\b",
        r"\b(hyperinflation|currency (crash|collapse|crisis)|bank run)\b",
        r"\b(food (shortage|crisis|price|riots?))\b.{0,30}\b(anger|protest|demand)\b",
        r"\b(legitimacy (crisis|challenge)|public (outrage|fury|backlash))\b",
        r"\b(government (collapse|crisis)|political (chaos|vacuum|instability))\b",
        r"\b(coup (attempt|plot|rumor)|mutiny|insurrection)\b",
    ]),
]

# Pre-compile for speed
_COMPILED: list[tuple[str, list[re.Pattern[str]]]] = [
    (tag, [re.compile(pat, re.IGNORECASE) for pat in patterns])
    for tag, patterns in _TAG_RULES
]


def tag_item(item: NewsItem) -> None:
    """Mutate item.tags in-place with any matching labels.

    Scans title + summary (not full text — too slow for inline filtering).
    """
    text = f"{item.title} {item.summary}"
    for tag, compiled_patterns in _COMPILED:
        if tag not in item.tags and any(p.search(text) for p in compiled_patterns):
            item.tags.append(tag)


def tag_news(items: list[NewsItem]) -> list[NewsItem]:
    """Tag all items in-place and return the same list.

    Idempotent — calling twice does not duplicate tags.
    """
    for item in items:
        tag_item(item)
    return items


def tags_summary(items: list[NewsItem]) -> dict[str, int]:
    """Return {tag: count} across all items — useful for diagnostics."""
    counts: dict[str, int] = {}
    for item in items:
        for tag in item.tags:
            counts[tag] = counts.get(tag, 0) + 1
    return counts
