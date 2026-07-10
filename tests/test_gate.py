"""gate 单元测：规则直判 / LLM 裁决 / 判据必落 / fail-closed（评审 4A CRITICAL）。"""

from __future__ import annotations

from serenity.gate.gate import GateResult, GateVocab, classify_question, load_gate_vocab


def _vocab():
    return GateVocab(tickers={"NVDA", "ANET"}, domains={"optics_cpo", "memory_hbm_nand"})


class _NeverCalledLLM:
    model = "never"

    def complete(self, **kw):
        raise AssertionError("规则直判不应触发 LLM")


# ── 规则层 ──


def test_rule_direct_ticker_hit_is_in_domain():
    g = classify_question(
        title="Will NVDA announce a Blackwell successor at GTC 2026?",
        deadline=None, vocab=_vocab(), llm=_NeverCalledLLM(),
    )
    assert g.state == "in_domain"
    assert "NVDA" in g.matched_terms
    assert g.rationale  # 判据必须非空


def test_rule_domain_keyword_hit_is_in_domain():
    g = classify_question(
        title="Will HBM prices rise 50% by year end?",
        deadline=None, vocab=_vocab(), llm=_NeverCalledLLM(),
    )
    assert g.state == "in_domain"
    assert any("memory_hbm_nand" in t for t in g.matched_terms)


def test_rule_ticker_requires_whole_word():
    """短 ticker 不做子串匹配（防 'A'/'AI' 误击）。"""
    vocab = GateVocab(tickers={"ARM"}, domains=set())
    g = classify_question(
        title="Will the farm bill pass before October?",  # 'farm' 含 'arm' 子串
        deadline=None, vocab=vocab,
        llm=_ScriptedGateLLM({"state": "out_of_domain", "rationale": "agriculture policy, no covered domain"}),
    )
    assert g.state == "out_of_domain"


# ── LLM 层 ──


class _ScriptedGateLLM:
    def __init__(self, *payloads):
        self.payloads = list(payloads)
        self.model = "scripted"
        self.n_calls = 0

    def complete(self, **kw):
        from serenity.agent.llm_client import LLMResponse
        self.n_calls += 1
        item = self.payloads.pop(0) if len(self.payloads) > 1 else self.payloads[0]
        if isinstance(item, Exception):
            raise item
        return LLMResponse(text="", parsed_json=item, model=self.model, cost_usd=0.001)


def test_llm_adjacent_with_rationale():
    llm = _ScriptedGateLLM({
        "state": "adjacent",
        "rationale": "Fed rate decision → neocloud_financing costs via one causal hop",
    })
    g = classify_question(title="Will the Fed cut rates in September?",
                          deadline=None, vocab=_vocab(), llm=llm)
    assert g.state == "adjacent"
    assert "neocloud_financing" in g.rationale


def test_llm_out_of_domain():
    llm = _ScriptedGateLLM({"state": "out_of_domain", "rationale": "sports outcome, unrelated to coverage"})
    g = classify_question(title="Will the Lakers win the 2027 NBA title?",
                          deadline=None, vocab=_vocab(), llm=llm)
    assert g.state == "out_of_domain"


# ── fail-closed（评审 4A CRITICAL）──


def test_parse_failure_retries_once_then_fail_closed():
    llm = _ScriptedGateLLM({"state": "banana"}, {"nonsense": True})  # 两次都坏
    g = classify_question(title="Some ambiguous question about things",
                          deadline=None, vocab=_vocab(), llm=llm)
    assert llm.n_calls == 2  # 重试恰好一次
    assert g.state == "out_of_domain"  # fail-closed 弃权
    assert g.parse_error == "gate_parse_error"
    assert g.rationale  # 判据仍非空（写明 fail-closed）


def test_parse_failure_recovers_on_retry():
    llm = _ScriptedGateLLM(
        {"state": "banana"},
        {"state": "adjacent", "rationale": "recovered on retry with a valid causal hop to optics_cpo"},
    )
    g = classify_question(title="Q", deadline=None, vocab=_vocab(), llm=llm)
    assert g.state == "adjacent" and g.parse_error is None


def test_llm_exception_fail_closed():
    llm = _ScriptedGateLLM(RuntimeError("gateway down"))
    g = classify_question(title="Q", deadline=None, vocab=_vocab(), llm=llm)
    assert g.state == "out_of_domain"
    assert g.parse_error and g.parse_error.startswith("llm_error")


# ── vocab 载入 ──


def test_load_gate_vocab_from_belief_tables(tmpdb):
    from serenity.store.dao import session_scope
    from serenity.store.models import BeliefPrimitive, TickerKnowledge

    with session_scope() as s:
        s.add(BeliefPrimitive(claim="c" * 30, domain="optics_cpo", tickers="ANET,COHR",
                              stance="bullish", confidence="high", belief_set_version="v1"))
        s.add(TickerKnowledge(ticker="NVDA", thesis="t" * 30, confidence="high",
                              belief_set_version="v1"))
        # 其它版本不应混入
        s.add(BeliefPrimitive(claim="x" * 30, domain="macro_market", tickers="SPY",
                              stance="neutral", confidence="low", belief_set_version="v2"))
    vocab = load_gate_vocab("v1")
    assert vocab.tickers == {"ANET", "COHR", "NVDA"}
    assert vocab.domains == {"optics_cpo"}


def test_gate_result_dataclass_defaults():
    g = GateResult(state="in_domain", rationale="r")
    assert g.parse_error is None and g.matched_terms == []


def test_rule_common_word_ticker_requires_uppercase(tmpdb):
    """QA ISSUE-003 回归：合法 ticker 撞英文常用词（BE/OPEN）不得小写误触发。"""
    vocab = GateVocab(tickers={"BE", "OPEN", "NVDA"}, domains=set())
    llm = _ScriptedGateLLM({"state": "out_of_domain", "rationale": "podcast word mention, unrelated"})
    g = classify_question(
        title='Will "Think" be said during the next episode of the Podcast?',
        deadline=None, vocab=vocab, llm=llm,
    )
    assert g.state == "out_of_domain"  # 小写 be 不触发
    g2 = classify_question(title="Will BE stock double by 2027?",
                           deadline=None, vocab=vocab, llm=_NeverCalledLLM())
    assert g2.state == "in_domain" and "BE" in g2.matched_terms  # 大写全词才触发


def test_rule_keyword_word_boundary(tmpdb):
    """QA ISSUE-003 回归：'nand' 不得子串命中 'Hernandez'。"""
    vocab = GateVocab(tickers=set(), domains={"memory_hbm_nand"})
    llm = _ScriptedGateLLM({"state": "out_of_domain", "rationale": "US House election, unrelated"})
    g = classify_question(title="Will Melissa Hernandez win the CA-14 House seat?",
                          deadline=None, vocab=vocab, llm=llm)
    assert g.state == "out_of_domain"
    g2 = classify_question(title="Will NAND flash prices rise 30% by December?",
                           deadline=None, vocab=vocab, llm=_NeverCalledLLM())
    assert g2.state == "in_domain"
