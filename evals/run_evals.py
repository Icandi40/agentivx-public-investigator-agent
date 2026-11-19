"""Run the golden set and the semantic cache experiment, then write EVALS.md.

  uv run python evals/run_evals.py golden            # live model calls
  uv run python evals/run_evals.py cache             # needs a golden run first
  uv run python evals/run_evals.py report            # rebuild EVALS.md from saved results

Every number in EVALS.md comes from the JSON files this script writes to
evals/results/. Nothing is typed in by hand.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from investigator.agent import InvestigatorAgent
from investigator.cache import FastEmbedder, SemanticCache
from investigator.evaluation import Grade, grade, percentile, summarize
from investigator.graph_store import SanctionsGraph
from investigator.llm import ModelConfig

ROOT = Path(__file__).resolve().parent.parent
EVAL_DIR = ROOT / "evals"
RESULTS = EVAL_DIR / "results"
GOLDEN = EVAL_DIR / "golden.jsonl"
PROBES = EVAL_DIR / "cache_probes.jsonl"
SWEEP_THRESHOLDS = [0.80, 0.84, 0.88, 0.90, 0.92, 0.94, 0.96, 0.98]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def stamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


class Budget:
    def __init__(self, limit_usd: float) -> None:
        self.limit = limit_usd
        self.spent = 0.0
        self._lock = threading.Lock()

    def add(self, cost: float) -> None:
        with self._lock:
            self.spent += cost

    @property
    def exceeded(self) -> bool:
        return self.spent >= self.limit


# --------------------------------------------------------------------- golden
def run_case(agent: InvestigatorAgent, case: dict[str, Any], budget: Budget, context: str) -> dict[str, Any]:
    if budget.exceeded:
        return {"id": case["id"], "skipped": "budget"}
    thread = f"{case['id']}-{time.time_ns()}"
    turns: list[dict[str, Any]] = []
    for question in case["turns"]:
        result = agent.ask(question, thread_id=thread)
        budget.add(float(result.get("cost_usd", 0.0)))
        turns.append({"question": question, "result": result})
    final = turns[-1]
    g = grade(case, final["result"], final["question"], context)
    return {
        "id": case["id"],
        "category": case["category"],
        "turns": turns,
        "grade": g.to_dict(),
        "case_cost_usd": sum(float(t["result"].get("cost_usd", 0.0)) for t in turns),
    }


def cmd_golden(args: argparse.Namespace) -> None:
    graph = SanctionsGraph.load()
    config = ModelConfig.from_env()
    if args.model:
        config.model = args.model
    if args.effort:
        config.effort = None if args.effort == "none" else args.effort
    agent = InvestigatorAgent(graph, config=config)
    cases = load_jsonl(GOLDEN)
    if args.limit:
        cases = cases[: args.limit]
    budget = Budget(args.budget)
    started = time.perf_counter()
    records: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        context = snapshot_context(graph)
        futures = {pool.submit(run_case, agent, case, budget, context): case["id"] for case in cases}
        for future in as_completed(futures):
            rec = future.result()
            records.append(rec)
            mark = "skip" if rec.get("skipped") else ("ok  " if rec["grade"]["success"] else "FAIL")
            print(f"{mark} {rec['id']:<12} spent ${budget.spent:.4f}", flush=True)
    records.sort(key=lambda r: r["id"])
    meta = {
        "model": config.model,
        "provider": config.provider,
        "effort": config.effort,
        "concurrency": args.concurrency,
        "snapshot": graph.snapshot.publication_date,
        "wall_clock_s": round(time.perf_counter() - started, 1),
        "note": args.note,
    }
    finish(records, meta, f"golden-{stamp()}")


def snapshot_context(graph: SanctionsGraph) -> str:
    """Facts the system prompt gives the model, which the prose check may treat as grounded."""
    return f"SDN list snapshot published {graph.snapshot.publication_date}"


def finish(records: list[dict[str, Any]], meta: dict[str, Any], run: str) -> None:
    done = [r for r in records if not r.get("skipped")]
    grades = [Grade(**r["grade"]) for r in done]
    latencies = [float(t["result"]["latency_s"]) for r in done for t in r["turns"]]
    costs = [float(r["case_cost_usd"]) for r in done]
    summary = summarize(grades, latencies, costs)
    summary.update(meta)
    summary.update(
        {
            "run": run,
            "turns_measured": len(latencies),
            "skipped_for_budget": len(records) - len(done),
            "verifier_retries": sum(t["result"].get("verify_retries", 0) for r in done for t in r["turns"]),
            "errors": sum(1 for r in done for t in r["turns"] if t["result"].get("status") == "error"),
            "failures": [{"id": g.case_id, "notes": g.notes} for g in grades if not g.success],
            "ungrounded_examples": [{"id": g.case_id, "spans": g.ungrounded} for g in grades if g.ungrounded],
        }
    )
    RESULTS.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(r, ensure_ascii=False) for r in records]
    (RESULTS / f"{run}.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (RESULTS / f"{run}.summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k not in {"failures", "ungrounded_examples"}}, indent=2))
    write_report()


def cmd_regrade(args: argparse.Namespace) -> None:
    """Re-score saved model outputs with the current golden file and grader. No model calls."""
    source = Path(args.results)
    original = json.loads(source.with_name(source.name.replace(".jsonl", ".summary.json")).read_text("utf-8"))
    graph = SanctionsGraph.load()
    context = snapshot_context(graph)
    cases = {c["id"]: c for c in load_jsonl(GOLDEN)}
    records = load_jsonl(source)
    for rec in records:
        if rec.get("skipped"):
            continue
        final = rec["turns"][-1]
        rec["grade"] = grade(cases[rec["id"]], final["result"], final["question"], context).to_dict()
    meta = {k: original[k] for k in ("model", "provider", "effort", "concurrency", "snapshot", "wall_clock_s")}
    meta["note"] = args.note
    meta["regraded_from"] = original["run"]
    finish(records, meta, f"{original['run']}-regraded-{stamp()}")


# ---------------------------------------------------------------------- cache
def created(path: Path) -> str:
    """The newest timestamp in a result file name (a regrade carries its own, later one)."""
    stamps = re.findall(r"\d{8}T\d{6}Z", path.name)
    return stamps[-1] if stamps else ""


def latest(pattern: str) -> Path | None:
    files = sorted(RESULTS.glob(pattern), key=created)
    return files[-1] if files else None


def cmd_cache(args: argparse.Namespace) -> None:
    golden_path = Path(args.golden_results) if args.golden_results else latest("golden-*.jsonl")
    if golden_path is None:
        sys.exit("run the golden suite first")
    graph = SanctionsGraph.load()
    embedder = FastEmbedder()
    cases = {c["id"]: c for c in load_jsonl(GOLDEN)}
    golden_records = [r for r in load_jsonl(golden_path) if not r.get("skipped")]
    probes = load_jsonl(PROBES)

    def warm(cache: SemanticCache) -> int:
        stored = 0
        for rec in golden_records:
            if len(rec["turns"]) != 1:
                continue
            turn = rec["turns"][0]
            result = dict(turn["result"]) | {"golden_id": rec["id"]}
            stored += cache.store(turn["question"], result)
        return stored

    # A probe is "answerable" from the cache when some cached golden answer is a correct answer to it
    # (valid_answers, labeled by hand from the question's meaning). Any other hit serves a wrong answer.
    answerable = [p for p in probes if p["valid_answers"]]

    # 1) Threshold sweep, guard on and off. Embeddings only, no model calls.
    sweep: list[dict[str, Any]] = []
    for guard in (True, False):
        for threshold in SWEEP_THRESHOLDS:
            cache = SemanticCache(graph, embedder, threshold=threshold, entity_guard=guard)
            warm(cache)
            correct = wrong = 0
            for p in probes:
                d = cache.lookup(p["question"])
                if d.hit and d.result is not None:
                    if d.result["golden_id"] in p["valid_answers"]:
                        correct += 1
                    else:
                        wrong += 1
            sweep.append(
                {
                    "entity_guard": guard,
                    "threshold": threshold,
                    "correct_hit_rate": correct / len(answerable),
                    "wrong_answer_rate": wrong / len(probes),
                }
            )

    # 2) Live run at the chosen threshold: every probe also runs through the agent,
    #    so each probe has a measured no-cache latency to compare against.
    config = ModelConfig.from_env()
    agent = InvestigatorAgent(graph, config=config)
    cache = SemanticCache(graph, embedder, threshold=args.threshold, entity_guard=True)
    stored = warm(cache)
    budget = Budget(args.budget)
    reused: dict[str, dict[str, Any]] = {}
    reused_from = None
    if args.reuse_live:
        previous = json.loads(Path(args.reuse_live).read_text(encoding="utf-8"))
        reused = {r["id"]: r for r in previous["rows"]}
        reused_from = previous["run"]
    rows: list[dict[str, Any]] = []
    for probe in probes:
        decision = cache.lookup(probe["question"])
        if probe["id"] in reused:
            live_latency = float(reused[probe["id"]]["uncached_latency_s"])
            live_cost = float(reused[probe["id"]]["uncached_cost_usd"])
        else:
            live = agent.ask(probe["question"], thread_id=f"{probe['id']}-{time.time_ns()}")
            live_latency, live_cost = float(live["latency_s"]), float(live.get("cost_usd", 0.0))
            budget.add(live_cost)
        row: dict[str, Any] = {
            "id": probe["id"],
            "kind": probe["kind"],
            "hit": decision.hit,
            "reason": decision.reason,
            "similarity": round(decision.similarity, 4),
            "lookup_ms": round(decision.lookup_ms, 2),
            "uncached_latency_s": live_latency,
            "uncached_cost_usd": live_cost,
        }
        if decision.hit and decision.result is not None:
            served_source = decision.result["golden_id"]
            row["served_from"] = served_source
            row["correct_hit"] = served_source in probe["valid_answers"]
            row["wrong_hit"] = not row["correct_hit"]
            if row["correct_hit"]:
                row["served_answer_passes_golden"] = grade(
                    cases[served_source], decision.result, probe["question"], snapshot_context(graph)
                ).success
        rows.append(row)
        print(f"{probe['id']:<12} hit={decision.hit!s:<5} sim={decision.similarity:.3f} {decision.reason}", flush=True)
        if budget.exceeded:
            print("budget reached, stopping")
            break

    def rate(kind: str, key: str) -> float:
        group = [r for r in rows if r["kind"] == kind]
        return sum(1 for r in group if r.get(key)) / len(group) if group else 0.0

    hit_lat = [r["lookup_ms"] / 1000 for r in rows if r["hit"]]
    miss_lat = [r["uncached_latency_s"] for r in rows if not r["hit"]]
    with_cache = [
        (r["lookup_ms"] / 1000) if r["hit"] else (r["uncached_latency_s"] + r["lookup_ms"] / 1000) for r in rows
    ]
    without_cache = [r["uncached_latency_s"] for r in rows]
    summary = {
        "run": f"cache-{stamp()}",
        "golden_source": golden_path.name,
        "model": config.model,
        "embed_model": embedder.model_name,
        "threshold": args.threshold,
        "cached_entries": stored,
        "probes": len(rows),
        "answerable_probes": len(answerable),
        "reused_live_measurements_from": reused_from,
        "correct_hit_rate": sum(1 for r in rows if r.get("correct_hit")) / len(answerable) if answerable else 0.0,
        "correct_hits": sum(1 for r in rows if r.get("correct_hit")),
        "wrong_hits": sum(1 for r in rows if r.get("wrong_hit")),
        "wrong_answer_rate": sum(1 for r in rows if r.get("wrong_hit")) / len(rows) if rows else 0.0,
        "paraphrase_hit_rate": rate("paraphrase", "hit"),
        "served_answers_passing_golden": sum(1 for r in rows if r.get("served_answer_passes_golden")),
        "near_miss_wrong_hit_rate": rate("near_miss", "wrong_hit"),
        "no_anchor_bypass_rate": 1.0 - rate("no_anchor", "hit"),
        "overall_hit_rate": sum(1 for r in rows if r["hit"]) / len(rows) if rows else 0.0,
        "hit_latency_p50_ms": statistics.median(hit_lat) * 1000 if hit_lat else 0.0,
        "miss_latency_p50_s": statistics.median(miss_lat) if miss_lat else 0.0,
        "mean_latency_without_cache_s": statistics.mean(without_cache) if without_cache else 0.0,
        "mean_latency_with_cache_s": statistics.mean(with_cache) if with_cache else 0.0,
        "p95_latency_without_cache_s": percentile(without_cache, 95),
        "p95_latency_with_cache_s": percentile(with_cache, 95),
        "cost_without_cache_usd": sum(r["uncached_cost_usd"] for r in rows),
        "cost_with_cache_usd": sum(0.0 if r["hit"] else r["uncached_cost_usd"] for r in rows),
        "live_run_cost_usd": budget.spent,
        "note": args.note,
        "sweep": sweep,
        "rows": rows,
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / f"{summary['run']}.summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k not in {"rows", "sweep"}}, indent=2))
    write_report()


# --------------------------------------------------------------------- report
def pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def write_report() -> None:
    golden_file = latest("golden-*.summary.json")
    cache_file = latest("cache-*.summary.json")
    lines = [
        "# Evaluation results",
        "",
        "Generated by `evals/run_evals.py report` from the JSON files in `evals/results/`. "
        "Do not edit by hand; rerun the script.",
        "",
    ]
    if golden_file is None:
        lines.append("No golden run yet.")
    else:
        s = json.loads(golden_file.read_text(encoding="utf-8"))
        lines += [
            "## Golden set",
            "",
            f"Run `{s['run']}`: model `{s['model']}` (provider {s['provider']}, effort {s['effort']}), "
            f"SDN snapshot {s['snapshot']}, {s['cases']} cases ({s['turns_measured']} turns), "
            f"concurrency {s['concurrency']}, wall clock {s['wall_clock_s']}s.",
            "",
            "| Metric | Value |",
            "| --- | --- |",
            f"| Task success rate | {pct(s['task_success_rate'])} |",
            f"| Citation recall (required records cited) | {pct(s['citation_recall'])} of {s['required_citations']} |",
            f"| Claim hallucination rate, first draft | {pct(s['draft_hallucination_rate'])} "
            f"of {s['draft_claims']} claims |",
            f"| Claim hallucination rate, after verifier retry | {pct(s['final_hallucination_rate'])} "
            f"of {s['final_claims']} claims |",
            f"| Answers with any ungrounded content shown | {pct(s['answers_with_ungrounded_content_rate'])} |",
            f"| Answers with an ungrounded prose span (heuristic) | {pct(s['answers_with_ungrounded_prose_rate'])} |",
            f"| Latency p50 per turn | {s['latency_p50_s']:.2f}s |",
            f"| Latency p95 per turn | {s['latency_p95_s']:.2f}s |",
            f"| Cost per question | ${s['cost_per_question_usd']:.5f} |",
            f"| Total cost of this run | ${s['cost_total_usd']:.4f} |",
            f"| Verifier retries triggered | {s['verifier_retries']} |",
            f"| Agent errors | {s['errors']} |",
            "",
            "### By category",
            "",
            "| Category | Cases | Success |",
            "| --- | --- | --- |",
        ]
        for cat, row in s["by_category"].items():
            lines.append(f"| {cat} | {row['cases']} | {pct(row['success_rate'])} |")
        lines += ["", "### Failed cases", ""]
        if s["failures"]:
            for f in s["failures"]:
                lines.append(f"- `{f['id']}`: {'; '.join(f['notes'])}")
        else:
            lines.append("None.")
        if s["ungrounded_examples"]:
            lines += ["", "### Prose spans the grounding heuristic flagged", ""]
            for ex in s["ungrounded_examples"]:
                lines.append(f"- `{ex['id']}`: {', '.join(ex['spans'])}")
        lines.append("")
    if cache_file is not None:
        c = json.loads(cache_file.read_text(encoding="utf-8"))
        lines += [
            "## Semantic cache",
            "",
            f"Run `{c['run']}`: embeddings `{c['embed_model']}` (local), threshold {c['threshold']}, entity guard on. "
            f"The cache was warmed with the single-turn answers from `{c['golden_source']}` "
            f"({c['cached_entries']} entries). {c['probes']} probe questions: paraphrases of cached questions, "
            "near misses (same wording, different record or program) and questions that name nothing on the list. "
            "Every probe was also answered live by the agent to get its uncached latency and cost"
            + (
                f" (live measurements reused from `{c['reused_live_measurements_from']}`)."
                if c.get("reused_live_measurements_from")
                else "."
            )
            + f" {c['answerable_probes']} probes have a correct answer in the cache (hand-labeled per probe in "
            "`evals/cache_probes.jsonl`); a hit that serves any other cached answer counts as a wrong answer.",
            "",
            "| Metric | Value |",
            "| --- | --- |",
            f"| Correct hits (answerable probes served a right answer) | {pct(c['correct_hit_rate'])} "
            f"({c['correct_hits']} of {c['answerable_probes']}) |",
            f"| Wrong answers served (all {c['probes']} probes) | {pct(c['wrong_answer_rate'])} ({c['wrong_hits']}) |",
            f"| Near misses served another record's answer | {pct(c['near_miss_wrong_hit_rate'])} |",
            f"| Paraphrase hit rate | {pct(c['paraphrase_hit_rate'])} |",
            f"| Served answers that pass the golden check | {c['served_answers_passing_golden']} |",
            f"| No-anchor questions bypassed | {pct(c['no_anchor_bypass_rate'])} |",
            f"| Overall hit rate on the probe mix | {pct(c['overall_hit_rate'])} |",
            f"| Hit latency p50 | {c['hit_latency_p50_ms']:.1f} ms |",
            f"| Miss latency p50 | {c['miss_latency_p50_s']:.2f}s |",
            f"| Mean latency, no cache vs cache | {c['mean_latency_without_cache_s']:.2f}s vs "
            f"{c['mean_latency_with_cache_s']:.2f}s |",
            f"| p95 latency, no cache vs cache | {c['p95_latency_without_cache_s']:.2f}s vs "
            f"{c['p95_latency_with_cache_s']:.2f}s |",
            f"| Model cost on the probe mix, no cache vs cache | ${c['cost_without_cache_usd']:.4f} vs "
            f"${c['cost_with_cache_usd']:.4f} |",
            "",
            "### Threshold sweep (embeddings only, no model calls)",
            "",
            "| Entity guard | Threshold | Correct hits (of answerable) | Wrong answers served (of all probes) |",
            "| --- | --- | --- | --- |",
        ]
        for row in c["sweep"]:
            lines.append(
                f"| {'on' if row['entity_guard'] else 'off'} | {row['threshold']:.2f} | "
                f"{pct(row['correct_hit_rate'])} | {pct(row['wrong_answer_rate'])} |"
            )
        lines.append("")
    runs = [p.name.replace(".summary.json", "") for p in sorted(RESULTS.glob("*.summary.json"), key=created)]
    if runs:
        lines += ["## Run log", "", "Every run kept in `evals/results/`, oldest first:", ""]
        for name in runs:
            data = json.loads((RESULTS / f"{name}.summary.json").read_text(encoding="utf-8"))
            if name.startswith("golden"):
                note = (
                    f"success {pct(data['task_success_rate'])}, {data['cases']} cases, "
                    f"${data['cost_total_usd']:.4f}, model {data['model']}, effort {data['effort']}"
                )
            else:
                note = f"live cost ${data['live_run_cost_usd']:.4f}"
            if data.get("note"):
                note += f". {data['note']}"
            lines.append(f"- `{name}`: {note}")
        lines.append("")
    lines += [
        "## Definitions",
        "",
        "- **Task success**: the status is one the case allows, every required record id is cited, every "
        "required value appears in the answer or its claims, no forbidden content appears, and misspelled "
        "names are labeled as possible matches.",
        "- **Citation recall**: required record ids cited by the final answer, over all required ids.",
        "- **Claim hallucination**: a structured claim whose value is not in any record it cites, or that "
        "cites a record no tool returned. Checked in code against the graph, no LLM judge. The verifier "
        "drops these claims before the user sees them, so the table reports the rate before dropping.",
        "- **Answers with any ungrounded content shown**: the final draft had a failing claim (its sentence "
        "may remain in the prose) or the prose has an ungrounded span.",
        "- **Ungrounded prose span**: a run of capitalized words, a number or a long identifier in the prose "
        "answer that appears in neither the tool outputs nor the question. A heuristic that over-flags.",
        "- **Latency**: wall clock per turn, from question in to verified answer out, including the plan "
        "call, tool calls and verification.",
        "- **Cost**: computed from the token usage the API returns, at the per-token prices in "
        "`src/investigator/llm.py`.",
        "",
    ]
    (ROOT / "EVALS.md").write_text("\n".join(lines), encoding="utf-8")
    print("wrote EVALS.md")


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    g = sub.add_parser("golden")
    g.add_argument("--limit", type=int, default=0)
    g.add_argument("--concurrency", type=int, default=4)
    g.add_argument("--budget", type=float, default=2.0, help="stop starting cases after this many USD")
    g.add_argument("--model", default="")
    g.add_argument("--effort", default="")
    g.add_argument("--note", default="")
    rg = sub.add_parser("regrade")
    rg.add_argument("results", help="a golden-*.jsonl file in evals/results")
    rg.add_argument("--note", default="")
    c = sub.add_parser("cache")
    c.add_argument("--golden-results", default="")
    c.add_argument("--threshold", type=float, default=0.90)
    c.add_argument("--budget", type=float, default=1.0)
    c.add_argument("--reuse-live", default="", help="reuse uncached latency and cost from a cache summary")
    c.add_argument("--note", default="")
    sub.add_parser("report")
    args = parser.parse_args()
    if args.command == "golden":
        cmd_golden(args)
    elif args.command == "regrade":
        cmd_regrade(args)
    elif args.command == "cache":
        cmd_cache(args)
    else:
        write_report()


if __name__ == "__main__":
    main()
