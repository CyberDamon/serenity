"""serenity CLI。

命令面：
  serenity status              公开只读健康检查
  serenity smoke [--submit]    一次性钱包接入烟雾
  serenity bootstrap           SIWE 持久钱包换 API key，写回 .env
  serenity distill [--force]   蒸馏语料 → 信念库（一次性；实验期重建被拒，评审 3A）
  serenity inventory           M0 题源盘点：闸门三态分布 + 结算周期
  serenity daily [--submit]    三臂预测（generic/placebo/serenity）+ 选择性提交
  serenity reconcile           拉结算真值
  serenity stats               三臂配对报告（聚类 bootstrap CI）
  serenity calibrate           重校准（hold-out 门控）
  serenity inspect <qid>       单题三臂明细 + 归因
"""

from __future__ import annotations

import argparse
import sys

from serenity.yarrow import YarrowAPIError, YarrowClient


def _cmd_status(_args: argparse.Namespace) -> int:
    from serenity.distill.pipeline import active_version, paired_sample_count

    client = YarrowClient()
    open_q, _ = client.list_questions(status="open", qtype="binary")
    print(f"[status] open binary（首页）: {len(open_q)} 条")
    for q in open_q[:5]:
        print(f"  • {q.id[:8]}  {q.title[:70]}")
    v = None
    try:
        from serenity.store.dao import init_db
        init_db()
        v = active_version()
    except Exception as e:
        print(f"[status] 信念库读取失败: {e}")
    if v:
        n = paired_sample_count(v)
        print(f"[status] 信念库版本: {v}  已结算配对样本: {n}")
    else:
        print("[status] 尚无激活信念库——先跑 `serenity distill`")
    return 0


def _cmd_smoke(args: argparse.Namespace) -> int:
    from eth_account import Account

    client = YarrowClient()
    acct = Account.create()
    addr = acct.address
    priv = acct.key.hex()
    print(f"[smoke] step0 ephemeral wallet: {addr}")
    auth = client.siwe_login(address=addr, private_key=priv)
    print(f"[smoke] step1 SIWE ok, user_id={auth.user_id or '(new)'}")
    key_resp = client.create_api_key(auth.access)
    api_key = key_resp.get("api_key")
    client.api_key = api_key
    print(f"[smoke] step2 api-key created: {str(api_key)[:8]}…")
    questions, _ = client.list_questions(status="open", qtype="binary")
    picks = questions[:5]
    print(f"[smoke] step3 picked {len(picks)} open questions")
    for q in picks:
        print(f"         • {q.title[:70]}")
    if not args.submit:
        print("[smoke] --submit 未指定，跳过写操作（仅验证读+鉴权）。")
        return 0
    payload = [
        {"question_id": q.id, "probability_yes": 0.5,
         "report": {"reasoning": "smoke test — will be withdrawn immediately"}}
        for q in picks
    ]
    client.submit_forecasts(payload)
    print(f"[smoke] step4 submitted {len(payload)} forecasts @ 0.5")
    try:
        try:
            tr = client.get_track_record().get("track_record", {})
            print(f"[smoke] step5 track-record: baseline={tr.get('baseline_total')} "
                  f"resolved={tr.get('questions_resolved')}")
        except YarrowAPIError as e:
            if e.status_code == 404:
                print("[smoke] step5 track-record: (新钱包，暂无记录) ← 预期")
            else:
                raise
    finally:
        n = client.withdraw_forecasts([q.id for q in picks])
        print(f"[smoke] step6 withdrew {n} forecasts — 写路径全绿 ✅")
    return 0


def _upsert_env(path: str, mapping: dict[str, str]) -> None:
    """把 KEY=VALUE 写入/更新 .env（保留其余行）。.env 已被 gitignore。"""
    from pathlib import Path

    p = Path(path)
    lines = p.read_text().splitlines() if p.exists() else []
    remaining = dict(mapping)
    out: list[str] = []
    for line in lines:
        key = line.split("=", 1)[0].strip() if "=" in line and not line.lstrip().startswith("#") else None
        if key in remaining:
            out.append(f"{key}={remaining.pop(key)}")
        else:
            out.append(line)
    for k, v in remaining.items():
        out.append(f"{k}={v}")
    p.write_text("\n".join(out) + "\n")


def _cmd_bootstrap(args: argparse.Namespace) -> int:
    """SIWE bootstrap：持久钱包换长期 API key，写回 .env。绝不回显私钥/密钥明文。"""
    from eth_account import Account

    from serenity.config import settings

    priv = args.private_key or settings.yarrow_private_key.get_secret_value()
    if priv:
        acct = Account.from_key(priv)
        origin = "指定钱包" if args.private_key else ".env 已有私钥"
    else:
        acct = Account.create()
        priv = acct.key.hex()
        origin = "新生成专用钱包"
    addr = acct.address
    print(f"[bootstrap] {origin}: {addr}")

    client = YarrowClient()
    auth = client.siwe_login(address=addr, private_key=priv)
    print(f"[bootstrap] SIWE ok, user_id={auth.user_id or '(new)'}")
    key_resp = client.create_api_key(auth.access)
    api_key = key_resp.get("api_key") or key_resp.get("key")
    if not api_key:
        print(f"[bootstrap] ✗ 未从响应拿到 api_key，字段: {list(key_resp)}")
        return 1
    _upsert_env(".env", {
        "YARROW_WALLET_ADDRESS": addr,
        "YARROW_PRIVATE_KEY": priv,
        "YARROW_API_KEY": api_key,
    })
    print(f"[bootstrap] api-key 创建成功: {str(api_key)[:8]}… （已写入 .env，未回显明文）")
    print(f"[bootstrap] expires_at={key_resp.get('expires_at')}")
    print("[bootstrap] ✅ serenity Yarrow 身份就绪。长跑 daemon 只需 YARROW_API_KEY，可事后清空私钥。")
    return 0


def _cmd_distill(args: argparse.Namespace) -> int:
    from serenity.distill import run_distill

    rep = run_distill(
        corpus_path=args.corpus, model=args.model,
        batch_size=args.batch_size, concurrency=args.concurrency, force=args.force,
    )
    if rep.skipped_reason:
        print(f"[distill] ✗ {rep.skipped_reason}")
        return 1
    print(f"[distill] ✅ version={rep.version}  语料 {rep.n_tweets_used} 条（{rep.corpus_span}）")
    print(f"[distill] beliefs={rep.n_beliefs}（raw {rep.n_beliefs_raw}） tickers={rep.n_tickers} "
          f"claims={rep.n_claims}（时间戳截断拒绝 {rep.n_claims_rejected_timestamp}）")
    print(f"[distill] batches={rep.n_batches}（失败 {rep.n_batches_failed}） cost=${rep.cost_usd:.2f}")
    for w in rep.warnings:
        print(f"[distill] ⚠ {w}")
    return 0


def _cmd_inventory(args: argparse.Namespace) -> int:
    """M0 题源盘点：闸门三态分布 + 结算周期分布（不预测，不落 predictions）。"""
    from collections import Counter
    from datetime import UTC, datetime

    from serenity.agent.llm_client import make_client
    from serenity.config import settings
    from serenity.distill.pipeline import active_version
    from serenity.gate.gate import classify_question, load_gate_vocab
    from serenity.store.dao import init_db
    from serenity.yarrow.client import parse_yarrow_time

    init_db()
    version = active_version()
    if not version:
        print("[inventory] ✗ 无激活信念库——先跑 `serenity distill`")
        return 1
    vocab = load_gate_vocab(version)
    gate_llm = make_client(settings.gate_model)
    client = YarrowClient()
    now = datetime.now(UTC)

    states: Counter[str] = Counter()
    cats: Counter[str] = Counter()
    horizons: Counter[str] = Counter()
    samples: dict[str, list[str]] = {"in_domain": [], "adjacent": []}
    cost = 0.0
    n = 0
    for q in client.iter_questions(status="open", qtype="binary"):
        n += 1
        if n > args.max_scan:
            break
        g = classify_question(title=q.title, deadline=q.scheduled_resolve_time,
                              vocab=vocab, llm=gate_llm)
        cost += g.cost_usd
        states[g.state] += 1
        if g.state != "out_of_domain":
            cats[q.category or "?"] += 1
            if len(samples[g.state]) < 8:
                samples[g.state].append(q.title[:80])
            rt = parse_yarrow_time(q.scheduled_resolve_time)
            if rt is not None:
                days = (rt - now).days
                bucket = "<7d" if days < 7 else "<30d" if days < 30 else "<90d" if days < 90 else "90d+"
                horizons[bucket] += 1
    total = sum(states.values())
    print(f"[inventory] 扫描 {total} 条 open binary（gate 成本 ${cost:.2f}）")
    for st in ("in_domain", "adjacent", "out_of_domain"):
        c = states.get(st, 0)
        print(f"  {st:14} {c:4}  ({c / max(1, total):5.1%})")
    print(f"  对口题类别分布: {dict(cats)}")
    print(f"  对口题结算周期: {dict(horizons)}")
    for st, titles in samples.items():
        if titles:
            print(f"  {st} 样例:")
            for t in titles:
                print(f"    • {t}")
    in_adj = states.get("in_domain", 0) + states.get("adjacent", 0)
    print(f"[inventory] 结论：当前存量 in+adjacent = {in_adj} 条。"
          f"（设计要求 ≥5 条/周 才够 12 周攒 60 配对样本）")
    return 0


def _cmd_daily(args: argparse.Namespace) -> int:
    from serenity.yarrow.runner import run_daily

    res = run_daily(
        submit=args.submit, max_questions=args.max, max_scan=args.max_scan,
        min_lead_days=args.min_lead_days,
    )
    print(f"[daily] mode={res.mode} version={res.belief_set_version} seen={res.seen} "
          f"gate(in={res.gated_in} adj={res.gated_adjacent} out={res.gated_out}) "
          f"submitted={res.submitted} skipped={res.skipped}")
    for it in res.items[:30]:
        if it.final_prob is not None:
            probs = (f"gen={it.generic_prob:.3f} δ={it.delta_log_odds:+.2f} "
                     f"ser={it.final_prob:.3f} pla={it.placebo_prob:.3f}")
        elif it.generic_prob is not None:
            probs = f"gen={it.generic_prob:.3f} (out-shadow)"
        else:
            probs = "-"
        print(f"  • {it.submit_status:9} [{(it.gate_state or '?'):12}] {probs}  "
              f"{it.skip_reason or '':22} {it.title[:48]}")
    return 0


def _cmd_reconcile(_args: argparse.Namespace) -> int:
    from serenity.scoring.reconcile import reconcile

    res = reconcile()
    print(f"[reconcile] checked={res.checked} newly_resolved={res.newly_resolved} "
          f"pending={res.still_pending} snapshots={res.snapshots_written}")
    return 0


def _cmd_stats(args: argparse.Namespace) -> int:
    from serenity.distill.pipeline import active_version
    from serenity.scoring.paired import paired_report
    from serenity.store.dao import init_db

    init_db()
    version = args.version or active_version()
    rep = paired_report(version, n_boot=args.n_boot)
    print(f"[stats] 信念库版本={rep.version or '(all)'}  已结算配对样本 n={rep.n_paired}")
    if rep.brier_generic is not None:
        print(f"  Brier  generic={rep.brier_generic:.4f}  serenity={rep.brier_serenity:.4f}"
              + (f"  placebo={rep.brier_placebo:.4f}" if rep.brier_placebo is not None else "")
              + (f"  market={rep.brier_market:.4f} (n={rep.n_market})" if rep.brier_market is not None else ""))
    for label, ci in (("serenity vs generic", rep.vs_generic), ("serenity vs placebo", rep.vs_placebo)):
        if ci is None or ci.n == 0:
            continue
        print(f"  {label}: Δ={ci.mean_diff:+.4f}  95%CI=[{ci.ci_low:+.4f}, {ci.ci_high:+.4f}]  "
              f"n={ci.n} clusters={ci.n_clusters} → {ci.verdict}")
    for w in rep.warnings:
        print(f"  ⚠ {w}")
    return 0


def _cmd_calibrate(args: argparse.Namespace) -> int:
    from serenity.agent.calibration import fit_calibrator, reset_calibrator
    from serenity.scoring.reconcile import build_calibration_data

    if args.reset:
        reset_calibrator()
        print("[calibrate] 已回退恒等映射（kill-switch）")
        return 0
    pairs, eff_n = build_calibration_data(scope="all")
    cal = fit_calibrator(pairs, effective_n=eff_n)
    cal.save()
    print(f"[calibrate] method={cal.method} version={cal.version} n_train={cal.n_train} "
          f"effective_n={eff_n:.1f} "
          f"holdout: identity={cal.holdout_brier_identity} model={cal.holdout_brier_model}")
    if cal.method == "identity":
        print("[calibrate] 未上线新映射（样本不足或未过 hold-out 门控）——保持恒等 ✅ 安全")
    return 0


def _cmd_inspect(args: argparse.Namespace) -> int:
    """单题三臂明细 + 先验归因（本地 predictions 库）。"""
    import json

    from serenity.store.dao import init_db, session_scope
    from serenity.store.models import Prediction

    init_db()
    with session_scope() as s:
        query = s.query(Prediction).filter(Prediction.question_id.like(f"{args.question_id}%"))
        if args.date:
            query = query.filter(Prediction.prediction_date == args.date)
        row = query.order_by(Prediction.prediction_date.desc(), Prediction.id.desc()).first()
        if row is None:
            print(f"[inspect] 未找到匹配 '{args.question_id}' 的预测（本地库）")
            return 1
        print(f"question_id : {row.question_id}")
        print(f"title       : {row.title}")
        print(f"date        : {row.prediction_date}  category={row.category}")
        print(f"gate        : {row.gate_state}")
        print(f"  判据      : {row.gate_rationale}")
        print(f"三臂        : generic={row.generic_prob} placebo={row.placebo_prob} "
              f"serenity={row.final_prob}")
        print(f"先验        : direction={row.prior_direction} strength={row.prior_strength} "
              f"δ={row.delta_log_odds} (placebo δ={row.placebo_delta_log_odds})")
        print(f"  belief_ids: {row.belief_ids}")
        if row.prior_rationale:
            print(f"  归因      : {row.prior_rationale[:400]}")
        print(f"版本        : {row.belief_set_version}  parse_errors={row.parse_errors}")
        print(f"聚合        : raw={row.raw_prob} ir_std={row.ir_std} n_valid={row.n_ir_valid} "
              f"models={row.llm_models} self_check_delta={row.self_check_delta}")
        print(f"market_prob : {row.market_implied_prob}")
        print(f"status      : {row.submit_status}  skip_reason={row.skip_reason}")
        print(f"resolve_time: {row.question_resolve_time}  first_submit={row.first_submit_ts}")
        if row.research:
            r = json.loads(row.research)
            print(f"\n研究审计 (backend={r.get('backend')}, n_sources={r.get('n_sources')})")
            print(f"  搜索词: {r.get('queries')}")
            for src in r.get("sources", []):
                print(f"    [{src.get('source')}] {src.get('title','')[:60]}  {src.get('url','')}")
    return 0


def _setup_logging() -> None:
    import logging

    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("serenity").setLevel(logging.INFO)
    for noisy in ("httpx", "httpcore", "openai", "anthropic", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def main(argv: list[str] | None = None) -> int:
    _setup_logging()
    parser = argparse.ArgumentParser(prog="serenity")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("status", help="公开只读健康检查")

    smoke = sub.add_parser("smoke", help="agent_e2e 式接入烟雾（一次性钱包）")
    smoke.add_argument("--submit", action="store_true", help="执行写操作（提交+撤回）；默认只读")

    boot = sub.add_parser("bootstrap", help="SIWE bootstrap：持久钱包换 API key，写回 .env")
    boot.add_argument("--private-key", default=None,
                      help="指定钱包私钥（64 hex，可带 0x）；不给则用 .env 或新生成")

    dst = sub.add_parser("distill", help="蒸馏语料 → 信念库（实验期重建被拒，--force 开新段）")
    dst.add_argument("--corpus", default=None, help="语料 JSON 路径（默认 settings.corpus_json_path）")
    dst.add_argument("--model", default=None, help="蒸馏模型（默认 settings.distill_model）")
    dst.add_argument("--batch-size", type=int, default=40)
    dst.add_argument("--concurrency", type=int, default=4)
    dst.add_argument("--force", action="store_true", help="实验期强制重建 = 开新实验段")

    inv = sub.add_parser("inventory", help="M0 题源盘点：闸门三态分布 + 结算周期")
    inv.add_argument("--max-scan", type=int, default=200)

    daily = sub.add_parser("daily", help="三臂预测（generic/placebo/serenity）+ 选择性提交")
    daily.add_argument("--submit", action="store_true", help="真实提交；默认 dry-run")
    daily.add_argument("--max", type=int, default=None, help="本轮 in/adjacent 候选题数上限")
    daily.add_argument("--max-scan", type=int, default=150, help="闸门扫描上限（控成本）")
    daily.add_argument("--min-lead-days", type=int, default=1)

    sub.add_parser("reconcile", help="拉结算真值 → Brier 快照")

    st = sub.add_parser("stats", help="三臂配对报告（聚类 bootstrap 95%% CI）")
    st.add_argument("--version", default=None, help="限定信念库版本（默认当前激活版本）")
    st.add_argument("--n-boot", type=int, default=5000)

    cal = sub.add_parser("calibrate", help="重校准（hold-out 门控）")
    cal.add_argument("--reset", action="store_true", help="回退恒等映射（kill-switch）")

    insp = sub.add_parser("inspect", help="单题三臂明细 + 先验归因")
    insp.add_argument("question_id", help="question_id（支持前缀）")
    insp.add_argument("--date", default=None, help="限定 prediction_date（YYYY-MM-DD）；默认最新")

    args = parser.parse_args(argv)
    handlers = {
        "status": _cmd_status, "smoke": _cmd_smoke, "bootstrap": _cmd_bootstrap,
        "distill": _cmd_distill, "inventory": _cmd_inventory, "daily": _cmd_daily,
        "reconcile": _cmd_reconcile, "stats": _cmd_stats, "calibrate": _cmd_calibrate,
        "inspect": _cmd_inspect,
    }
    return handlers[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
