"""Replay persisted-state evidence; recorded engineering results, not model gains."""

import hashlib
import json
import math
import statistics
from pathlib import Path

from source_evidence import verify_sources

ROOT = Path(__file__).resolve().parents[1]


def main():
    evidence = ROOT / "evidence/enterprise-storage-v1/local-cpu"
    report = json.loads((evidence / "manifest.json").read_text(encoding="utf-8"))
    assert report["passed"] and report["discovered"] == 267 and report["skipped"] == 0
    assert report["cuda_visible_devices"] == ""
    assert len(report["tests"]) == len({r["test"] for r in report["tests"]}) == 267
    assert all(r["outcome"] == "passed" and math.isfinite(r["seconds"]) and r["seconds"] >= 0 for r in report["tests"])
    assert sum(r["test"].startswith("test_state_store.") for r in report["tests"]) == 18
    assert sum(r["test"].startswith("test_working_set.") for r in report["tests"]) == 16
    assert hashlib.sha256((evidence / "tests.log").read_bytes()).hexdigest() == report["test_log_sha256"]
    verify_sources(report["source_hashes"])
    measured = report["residency"]
    assert measured["source_nodes"] == 22 and measured["gpu_used"] is False
    cases = measured["cases"]
    assert len(cases) == 27
    assert {(c["repeats"], c["seed"], c["mode"]) for c in cases} == {
        (r, s, m) for r in (0, 2, 10) for s in (7, 29, 61) for m in ("direct", "working_set", "prefetch")}
    for case in cases:
        assert case["passed"] and case["failure_type"] is None
        rows, values = case["rows"], case["stream"]
        assert len(rows) == len(values) == case["requests"] == 20
        assert sum(a == b for a, b in zip(values, values[1:])) == case["repeats"]
        assert [r["position"] for r in rows] == list(range(20)) and [r["value"] for r in rows] == values
        assert all(r["correct"] for r in rows)
        times = sorted(r["milliseconds"] for r in rows)
        assert all(math.isfinite(t) and t > 0 for t in times)
        assert math.isclose(case["median_demand_ms"], statistics.median(times), rel_tol=1e-12)
        assert math.isclose(case["p95_demand_ms"], times[18], rel_tol=1e-12)
        assert math.isfinite(case["total_stream_ms"]) and sum(times) <= case["total_stream_ms"]
        assert case["open_with_replay_ms"] > 0 and case["payload_read_bytes"] >= 24000 * case["payload_read_count"]
        stats = case["working_stats"]
        if case["mode"] == "direct":
            assert stats is None and case["payload_read_count"] == 40 and case["prefetch_replies"] == []
        else:
            assert stats["resident_nodes"] <= 4 and stats["resident_bytes"] <= 120000 and stats["active_nodes"] == 2
            assert stats["cold_reads"] == case["payload_read_count"]
            assert 0 <= stats["prefetch_used"] <= stats["prefetched"]
            assert 0 <= stats["wasted_prefetch"] <= stats["prefetched"]
            if stats["prefetched"]:
                assert math.isclose(stats["prefetch_usefulness"], stats["prefetch_used"] / stats["prefetched"], rel_tol=1e-12)
            else:
                assert stats["prefetch_usefulness"] is None
            if case["mode"] == "working_set":
                assert case["payload_read_count"] == 21 - case["repeats"] and stats["prefetched"] == 0
            for reply in case["prefetch_replies"]:
                assert reply["status"] in ("ready", "discarded", "deferred")
    assert report["example"]["returncode"] == 0
    example = json.loads(report["example"]["stdout"])
    assert example["state_reopened"] and example["verified_decision"] and example["audit_events"] == 5
    assert example["prefetched_before_demand"] and example["revocation_invalidates_packet"]
    assert example["collected_working_copies"] == 1 and example["retained_source_versions"] == 2
    assert example["model_calls"] == 0
    summaries = []
    for repeat in (0, 2, 10):
        selected = {m: [c for c in cases if c["repeats"] == repeat and c["mode"] == m]
                    for m in ("direct", "working_set", "prefetch")}
        summaries.append(dict(repeat_fraction=repeat / 20,
            median_stream_ms={m: statistics.median(c["total_stream_ms"] for c in group) for m, group in selected.items()},
            payload_reads={m: sum(c["payload_read_count"] for c in group) for m, group in selected.items()}))
    print(json.dumps(summaries, indent=2))
    print("267 passing test records and 540 exact-reference replay checks verified; no model-quality or production-scale claim.")


if __name__ == "__main__":
    main()
