#!/usr/bin/env python3
"""Golden evals. Writes evals/eval_report.json deterministically
(no timestamps; byte-identical across runs on the same code)."""
import json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from rag_ablation.core import factor_deltas, reversal_holds, run_factorial

PINNED_QUALITY = {
    "M|exact_bm25|terse|0": 0.5, "M|exact_bm25|terse|1": 0.5, "M|exact_bm25|terse|2": 0.5,
    "M|exact_bm25|verbose|0": 0.4, "M|exact_bm25|verbose|1": 0.461538, "M|exact_bm25|verbose|2": 0.470588,
    "M|expanded_tfidf|terse|0": 1.0, "M|expanded_tfidf|terse|1": 1.0, "M|expanded_tfidf|terse|2": 1.0,
    "M|expanded_tfidf|verbose|0": 0.9, "M|expanded_tfidf|verbose|1": 0.961538, "M|expanded_tfidf|verbose|2": 0.970588,
    "S|exact_bm25|terse|0": 0.666667, "S|exact_bm25|terse|1": 0.666667, "S|exact_bm25|terse|2": 0.666667,
    "S|exact_bm25|verbose|0": 0.566667, "S|exact_bm25|verbose|1": 0.961538, "S|exact_bm25|verbose|2": 0.97619,
    "S|expanded_tfidf|terse|0": 0.666667, "S|expanded_tfidf|terse|1": 0.666667, "S|expanded_tfidf|terse|2": 0.666667,
    "S|expanded_tfidf|verbose|0": 0.566667, "S|expanded_tfidf|verbose|1": 0.961538, "S|expanded_tfidf|verbose|2": 0.97619,
}
TOL = 0.001


def main():
    table = run_factorial(10)
    deltas = factor_deltas(table)
    rev = reversal_holds(deltas)
    checks = []

    def add(name, passed, detail):
        checks.append({"name": name, "passed": bool(passed), "detail": detail})

    s, m = deltas["S"], deltas["M"]
    add("reversal_direction_S",
        max(s["writer_delta"], s["expansion_delta"]) > s["retriever_delta"],
        f"S: max(writer {s['writer_delta']}, expansion {s['expansion_delta']}) > retriever {s['retriever_delta']}")
    add("reversal_direction_M",
        m["retriever_delta"] > max(m["writer_delta"], m["expansion_delta"]),
        f"M: retriever {m['retriever_delta']} > max(writer {m['writer_delta']}, expansion {m['expansion_delta']})")
    add("reversal_holds_overall", rev["holds"], f"S={rev['S']} M={rev['M']}")

    # pinned per-cell quality table
    bad = []
    for key, expected in sorted(PINNED_QUALITY.items()):
        r, ret, w, width = key.split("|")
        got = table[(r, ret, w, int(width))]["quality"]
        if abs(got - expected) > TOL:
            bad.append(f"{key}: got {got}, want {expected}")
    add("per_cell_quality_pinned", not bad, "; ".join(bad) if bad else "all 24 cells within 0.001 of pinned values")
    add("cell_count_24", len(table) == 24, f"cells={len(table)}")

    for regime in ("S", "M"):
        ok = all(
            table[(regime, ret, "terse", wd)]["faithfulness"]
            >= table[(regime, ret, "verbose", wd)]["faithfulness"]
            for ret in ("exact_bm25", "expanded_tfidf") for wd in (0, 1, 2))
        add(f"terse_faithfulness_ge_verbose_{regime}", ok, f"regime {regime}, all retriever/width pairs")

    add("expansion_helps_S_more_than_M",
        s["expansion_delta"] > m["expansion_delta"],
        f"S expansion {s['expansion_delta']} > M expansion {m['expansion_delta']}")
    add("expanded_beats_exact_in_M", m["retriever_delta"] > 0.3,
        f"M retriever delta {m['retriever_delta']}")
    add("retrievers_similar_in_S", abs(s["retriever_delta"]) < 0.05,
        f"S retriever delta {s['retriever_delta']}")
    add("verbose_beats_terse_correctness_S_width2",
        table[("S", "exact_bm25", "verbose", 2)]["correctness"]
        > table[("S", "exact_bm25", "terse", 2)]["correctness"],
        "verbose at width 2 covers spread facts terse misses in S")

    again = run_factorial(10)
    add("determinism_factorial_rerun_identical", again == table, "two factorial runs compared")

    report = {
        "checks": checks,
        "deltas": deltas,
        "passed": sum(1 for c in checks if c["passed"]),
        "total": len(checks),
        "reversal": rev,
    }
    out = os.path.join(os.path.dirname(__file__), "eval_report.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, sort_keys=True)
        f.write("\n")
    print(f"evals: {report['passed']}/{report['total']} passed -> {out}")
    for c in checks:
        print(f"  {'PASS' if c['passed'] else 'FAIL'} {c['name']}: {c['detail']}")
    return 0 if report["passed"] == report["total"] else 1


if __name__ == "__main__":
    sys.exit(main())
