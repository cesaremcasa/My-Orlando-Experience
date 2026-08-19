from __future__ import annotations

import json
import statistics
import time
from pathlib import Path
from typing import Any

from src.agentops.errors import AgentOpsError
from src.agentops.fakes import FakeAgentLlm, FixtureRetriever
from src.agentops.memory.embedder import FakeEmbedder
from src.agentops.memory.store import LocalMemoryStore
from src.agentops.runtime import AgentRuntime, reset_runtime
from src.retrieve.contracts import RetrievedChunk

FIXTURE_PATH = Path(__file__).resolve().parent / "cases.json"
DEFAULT_SEED = 42


def load_cases(path: Path | None = None) -> dict[str, Any]:
    payload = json.loads((path or FIXTURE_PATH).read_text(encoding="utf-8"))
    cases = payload.get("cases") or []
    if len(cases) != 20:
        raise ValueError("evaluation fixture must contain exactly 20 cases")
    return payload


async def _run_case(case: dict[str, Any], memory: LocalMemoryStore) -> dict[str, Any]:
    kind = str(case["kind"])
    retriever: Any = FixtureRetriever()
    fake_llm = FakeAgentLlm()
    if kind == "no_evidence":
        retriever = FixtureRetriever([])
    elif kind == "invalid_citations":
        fake_llm = FakeAgentLlm(
            response={
                "response": "Unverified citation path.",
                "citation_ids": ["secret-internal-999"],
                "memory_candidate": None,
            }
        )
    elif kind == "prompt_injection":
        fake_llm = FakeAgentLlm(
            response={
                "response": "I should not leak memory.",
                "citation_ids": ["leaked-memory"],
                "memory_candidate": None,
            }
        )
    elif kind == "retrieval_failure":
        retriever = _BoomRetriever()
    elif kind == "memory_failure":
        memory = _BoomMemory(root=memory.root, embedder=FakeEmbedder(), use_faiss=False)
    elif kind == "provider_failure":
        fake_llm = FakeAgentLlm(fail_with=RuntimeError("provider-down"))
    elif kind == "safety_fail":
        fake_llm = FakeAgentLlm(
            safety={"verdict": "FAIL", "reason": "ungrounded", "schema_ok": True, "grounded": False}
        )
    elif kind == "schema_fail":
        fake_llm = FakeAgentLlm(response={"nope": True})
    elif kind == "safety_schema_fail":
        fake_llm = FakeAgentLlm(safety={"nope": True})

    label_ids: dict[str, str] = {}
    for item in case.get("seed_memory") or []:
        memory_id = await memory.add(
            user_id=str(item["user_id"]),
            session_id=str(case["session_id"]),
            content=str(item["content"]),
            provenance="response",
            ttl_seconds=item.get("ttl_seconds", 3600),
        )
        if item.get("label"):
            label_ids[str(item["label"])] = memory_id

    runtime = reset_runtime(
        AgentRuntime(retriever=retriever, memory=memory, fake_llm=fake_llm)
    )
    started = time.perf_counter()
    try:
        result = await runtime.chat(
            user_id=str(case["user_id"]),
            session_id=str(case["session_id"]),
            message=str(case["message"]),
            remember=bool(case.get("remember", False)),
        )
        latency_ms = round((time.perf_counter() - started) * 1000, 2)
        hits = await memory.search(user_id=str(case["user_id"]), query=str(case["message"]), top_k=3)
        foreign_hits = []
        if case.get("expect", {}).get("foreign_user"):
            foreign_hits = await memory.search(
                user_id=str(case["expect"]["foreign_user"]),
                query=str(case["message"]),
                top_k=3,
            )
        return {
            "id": case["id"],
            "kind": kind,
            "ok": True,
            "error": None,
            "grounding_status": result.grounding_status,
            "citation_ids": [str(item.source_id) for item in result.citations],
            "citation_valid": bool(result.citations) if result.grounding_status == "grounded" else result.citations == [],
            "abstained": result.grounding_status == "abstained",
            "memory_ids": list(result.memory_ids),
            "memory_hit_ids": [hit.memory_id for hit in hits],
            "memory_hit_users": [hit.user_id for hit in hits],
            "foreign_hit_users": [hit.user_id for hit in foreign_hits],
            "relevant_retrieved_ids": [
                mid for mid in [hit.memory_id for hit in hits] if mid in label_ids.values()
            ],
            "label_ids": label_ids,
            "safety_decision": "pass" if result.grounding_status == "grounded" else "abstain",
            "tool_outcome": "ok",
            "latency_ms": latency_ms,
            "expect": case.get("expect") or {},
        }
    except AgentOpsError as exc:
        latency_ms = round((time.perf_counter() - started) * 1000, 2)
        return {
            "id": case["id"],
            "kind": kind,
            "ok": False,
            "error": exc.code,
            "grounding_status": None,
            "citation_ids": [],
            "citation_valid": True,
            "abstained": False,
            "memory_ids": [],
            "memory_hit_ids": [],
            "memory_hit_users": [],
            "foreign_hit_users": [],
            "relevant_retrieved_ids": [],
            "label_ids": label_ids,
            "safety_decision": "error",
            "tool_outcome": exc.code,
            "latency_ms": latency_ms,
            "expect": case.get("expect") or {},
        }


def _score(results: list[dict[str, Any]]) -> dict[str, Any]:
    grounded = [item for item in results if item["expect"].get("grounding_status") == "grounded"]
    no_evidence = [item for item in results if item["kind"] == "no_evidence"]
    citation_ok = all(item["citation_valid"] and item["grounding_status"] == "grounded" for item in grounded)
    abstain_ok = all(item["abstained"] for item in no_evidence)
    leaks = [
        item
        for item in results
        if item["expect"].get("leakage") == 0
        and any(user == item["expect"].get("foreign_user") for user in item.get("memory_hit_users") or [])
    ]
    precision_scores: list[float] = []
    no_hit_scores: list[float] = []
    for item in results:
        expect = item["expect"]
        k = int(expect.get("memory_k") or 3)
        retrieved = list(item.get("memory_hit_ids") or [])[:k]
        label_ids = item.get("label_ids") or {}
        relevant_ids = {label_ids[label] for label in expect.get("relevant_memory_labels") or [] if label in label_ids}
        if expect.get("expect_no_memory_hits"):
            no_hit_scores.append(1.0 if not retrieved else 0.0)
            continue
        if "relevant_memory_labels" not in expect:
            continue
        if not retrieved:
            precision_scores.append(0.0)
            continue
        precision_scores.append(len(set(retrieved) & relevant_ids) / len(retrieved))
    tool_correct = 0
    for item in results:
        expected_tool = str(item["expect"].get("tool_outcome") or ("ok" if item["ok"] else item.get("error") or "ok"))
        actual_tool = "ok" if item.get("tool_outcome") == "ok" else str(item.get("tool_outcome") or item.get("error") or "ok")
        if expected_tool == actual_tool:
            tool_correct += 1
    safety_correct = 0
    for item in results:
        expected_safety = str(item["expect"].get("safety_decision") or ("pass" if item["expect"].get("grounding_status") == "grounded" else "abstain"))
        if item.get("error") and item["expect"].get("error"):
            expected_safety = "error"
        actual_safety = str(item.get("safety_decision") or "error")
        if expected_safety == actual_safety:
            safety_correct += 1
    grounded_success = sum(1 for item in grounded if item["grounding_status"] == "grounded") / max(len(grounded), 1)
    latencies = sorted(float(item["latency_ms"]) for item in results)
    p50 = statistics.median(latencies) if latencies else 0.0
    if latencies:
        index = min(len(latencies) - 1, int(round(0.95 * (len(latencies) - 1))))
        p95 = latencies[index]
    else:
        p95 = 0.0
    return {
        "citation_validity": 1.0 if citation_ok else 0.0,
        "grounded_case_success_rate": grounded_success,
        "groundedness": grounded_success,
        "groundedness_deprecated": "fixture outcome agreement; not an independent groundedness judge",
        "abstention_correctness": 1.0 if abstain_ok else 0.0,
        "memory_precision_at_k": round(sum(precision_scores) / max(len(precision_scores), 1), 4) if precision_scores else 1.0,
        "memory_no_hit_accuracy": round(sum(no_hit_scores) / max(len(no_hit_scores), 1), 4) if no_hit_scores else 1.0,
        "tool_outcome_accuracy": round(tool_correct / max(len(results), 1), 4),
        "safety_decision_accuracy": round(safety_correct / max(len(results), 1), 4),
        "p50_ms": round(float(p50), 2),
        "p95_ms": round(float(p95), 2),
        "timing_note": "wall-clock latency; not byte-deterministic",
        "cross_user_leaks": len(leaks),
        "case_count": len(results),
    }


def render_markdown(report: dict[str, Any]) -> str:
    metrics = report["metrics"]
    lines = [
        "# AgentOps evaluation",
        "",
        f"- seed: `{report['seed']}`",
        f"- cases: `{metrics['case_count']}`",
        f"- citation validity: `{metrics['citation_validity']}`",
        f"- grounded case success rate: `{metrics['grounded_case_success_rate']}`",
        f"- abstention correctness: `{metrics['abstention_correctness']}`",
        f"- memory precision@k: `{metrics['memory_precision_at_k']}`",
        f"- memory no-hit accuracy: `{metrics['memory_no_hit_accuracy']}`",
        f"- tool outcome accuracy: `{metrics['tool_outcome_accuracy']}`",
        f"- safety decision accuracy: `{metrics['safety_decision_accuracy']}`",
        f"- p50_ms: `{metrics['p50_ms']}` (wall-clock)",
        f"- p95_ms: `{metrics['p95_ms']}` (wall-clock)",
        f"- cross-user leaks: `{metrics['cross_user_leaks']}`",
        "",
        "## Cases",
        "",
    ]
    for item in report["results"]:
        lines.append(
            f"- `{item['id']}` kind={item['kind']} ok={item['ok']} "
            f"status={item['grounding_status']} error={item['error']}"
        )
    lines.append("")
    return "\n".join(lines)


def write_reports(report: dict[str, Any], out_dir: Path) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "eval-report.json"
    md_path = out_dir / "eval-report.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_path.write_text(render_markdown(report), encoding="utf-8")
    return json_path, md_path


async def run_eval(
    *,
    seed: int = DEFAULT_SEED,
    data_root: Path,
    fixture_path: Path | None = None,
    upload_phoenix: bool = False,
) -> dict[str, Any]:
    payload = load_cases(fixture_path)
    memory = LocalMemoryStore(root=data_root, embedder=FakeEmbedder(), use_faiss=False)
    results: list[dict[str, Any]] = []
    for case in payload["cases"]:
        results.append(await _run_case(case, memory))
    report = {
        "seed": seed,
        "fixture_seed": payload.get("seed"),
        "license": payload.get("license"),
        "results": results,
        "metrics": _score(results),
        "phoenix_upload": "skipped" if not upload_phoenix else "optional-local-only",
    }
    return report


class _BoomRetriever:
    def retrieve(self, query: str, top_k: int = 3) -> list[RetrievedChunk]:
        del query, top_k
        raise RuntimeError("retrieval-down")


class _BoomMemory(LocalMemoryStore):
    async def search(self, *, user_id: str, query: str, top_k: int = 3):
        del user_id, query, top_k
        raise RuntimeError("memory-down")
