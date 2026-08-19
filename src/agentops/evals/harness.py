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

    for item in case.get("seed_memory") or []:
        await memory.add(
            user_id=str(item["user_id"]),
            session_id=str(case["session_id"]),
            content=str(item["content"]),
            provenance="response",
            ttl_seconds=item.get("ttl_seconds", 3600),
        )

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
    memory_cases = [item for item in results if "memory_hits_min" in item["expect"]]
    precision_scores = []
    for item in memory_cases:
        hits = item["memory_hit_ids"]
        expected_min = int(item["expect"]["memory_hits_min"])
        relevant = len(hits) if expected_min > 0 else 0
        precision_scores.append((1.0 if (expected_min == 0 and not hits) or (expected_min > 0 and relevant >= expected_min) else 0.0))
    tool_success = sum(1 for item in results if item["ok"] or item["expect"].get("error")) / max(len(results), 1)
    safety_cases = [item for item in results if item["kind"] in {"safety_fail", "safety_schema_fail", "grounded", "relevant_memory", "expired_memory", "cross_user"}]
    safety_pass = sum(1 for item in safety_cases if item["grounding_status"] == "grounded") / max(len(safety_cases), 1)
    latencies = sorted(float(item["latency_ms"]) for item in results)
    p50 = statistics.median(latencies) if latencies else 0.0
    if latencies:
        index = min(len(latencies) - 1, int(round(0.95 * (len(latencies) - 1))))
        p95 = latencies[index]
    else:
        p95 = 0.0
    return {
        "citation_validity": 1.0 if citation_ok else 0.0,
        "groundedness": sum(1 for item in grounded if item["grounding_status"] == "grounded") / max(len(grounded), 1),
        "abstention_correctness": 1.0 if abstain_ok else 0.0,
        "memory_precision_at_k": sum(precision_scores) / max(len(precision_scores), 1) if precision_scores else 1.0,
        "tool_success": round(tool_success, 4),
        "safety_pass_rate": round(safety_pass, 4),
        "p50_ms": round(float(p50), 2),
        "p95_ms": round(float(p95), 2),
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
        f"- groundedness: `{metrics['groundedness']}`",
        f"- abstention correctness: `{metrics['abstention_correctness']}`",
        f"- memory precision@k: `{metrics['memory_precision_at_k']}`",
        f"- tool success: `{metrics['tool_success']}`",
        f"- safety pass rate: `{metrics['safety_pass_rate']}`",
        f"- p50_ms: `{metrics['p50_ms']}`",
        f"- p95_ms: `{metrics['p95_ms']}`",
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
