#!/usr/bin/env python3
"""CLI: generate / run / ablate / report / check / demo / serve.

Exit codes for `check`: 0 = reversal holds, 1 = reversal does not hold,
2 = bad input.
"""
import argparse
import json
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer

from rag_ablation.core import (
    REGIMES, corpus_to_dict, factor_deltas, generate_corpus,
    reversal_holds, run_factorial,
)


def _table_json(table):
    return {
        f"{r}|{ret}|{w}|{width}": v
        for (r, ret, w, width), v in sorted(table.items())
    }


def cmd_generate(args):
    docs, queries = generate_corpus(args.n)
    data = corpus_to_dict(docs, queries)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, sort_keys=True)
        print(f"wrote {args.output}: {len(data['documents'])} documents, {len(data['queries'])} queries")
    else:
        print(json.dumps(data, indent=2, sort_keys=True))
    return 0


def cmd_run(args):
    if args.regime and args.regime not in REGIMES:
        print(f"bad regime: {args.regime}", file=sys.stderr)
        return 2
    table = run_factorial(args.n)
    out = _table_json(table)
    if args.regime:
        out = {k: v for k, v in out.items() if k.startswith(args.regime + "|")}
    print(json.dumps(out, indent=2, sort_keys=True))
    return 0


def cmd_ablate(args):
    table = run_factorial(args.n)
    deltas = factor_deltas(table)
    print(json.dumps({"cells": _table_json(table), "deltas": deltas,
                      "reversal": reversal_holds(deltas)}, indent=2, sort_keys=True))
    return 0


def cmd_report(args):
    table = run_factorial(args.n)
    deltas = factor_deltas(table)
    rev = reversal_holds(deltas)
    lines = ["# Ablation report", "",
             "| Regime | Retriever | Writer | Width | Correctness | Faithfulness | Quality |",
             "|---|---|---|---|---|---|---|"]
    for (r, ret, w, width), v in sorted(table.items()):
        lines.append(f"| {r} | {ret} | {w} | {width} | {v['correctness']:.3f} | "
                     f"{v['faithfulness']:.3f} | {v['quality']:.3f} |")
    lines += ["", "## Factor deltas", "",
              "| Regime | Retriever delta | Writer delta | Expansion delta |",
              "|---|---|---|---|"]
    for regime in REGIMES:
        d = deltas[regime]
        lines.append(f"| {regime} | {d['retriever_delta']:.3f} | {d['writer_delta']:.3f} | "
                     f"{d['expansion_delta']:.3f} |")
    lines += ["", f"Reversal holds: {rev['holds']} (S: {rev['S']}, M: {rev['M']})"]
    text = "\n".join(lines) + "\n"
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"wrote {args.output}")
    else:
        print(text, end="")
    return 0


def cmd_check(args):
    if args.regime and args.regime not in REGIMES:
        print(f"bad regime: {args.regime}", file=sys.stderr)
        return 2
    if args.n < 1:
        print("bad input: --n must be >= 1", file=sys.stderr)
        return 2
    table = run_factorial(args.n)
    deltas = factor_deltas(table)
    rev = reversal_holds(deltas)
    if args.regime:
        ok = rev[args.regime]
        print(f"regime {args.regime} reversal direction: {ok}")
        return 0 if ok else 1
    print(json.dumps({"deltas": deltas, "reversal": rev}, indent=2, sort_keys=True))
    return 0 if rev["holds"] else 1


def cmd_demo(args):
    docs, queries = generate_corpus(args.n)
    print(f"generated {len(docs)} documents, {len(queries)} queries (regimes S and M)")
    table = run_factorial(args.n)
    deltas = factor_deltas(table)
    rev = reversal_holds(deltas)
    for regime in REGIMES:
        d = deltas[regime]
        print(f"regime {regime}: retriever delta {d['retriever_delta']:.3f}, "
              f"writer delta {d['writer_delta']:.3f}, expansion delta {d['expansion_delta']:.3f}")
    print(f"reversal holds: {rev['holds']}")
    return 0 if rev["holds"] else 1


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        if self.path == "/health":
            body = json.dumps({"status": "ok"}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        if self.path != "/ablate":
            self.send_response(404)
            self.end_headers()
            return
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b"{}"
        try:
            payload = json.loads(raw.decode() or "{}")
            n = int(payload.get("n", 10))
            if n < 1 or n > 50:
                raise ValueError("n out of range")
        except Exception:
            body = json.dumps({"error": "bad input"}).encode()
            self.send_response(400)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        table = run_factorial(n)
        deltas = factor_deltas(table)
        body = json.dumps({"cells": _table_json(table), "deltas": deltas,
                           "reversal": reversal_holds(deltas)},
                          sort_keys=True).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def cmd_serve(args):
    server = HTTPServer(("127.0.0.1", args.port), _Handler)
    print(f"serving on http://127.0.0.1:{args.port} (GET /health, POST /ablate)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(prog="cli.py", description="RAG writer/chunk ablation harness")
    sub = p.add_subparsers(dest="command", required=True)
    for name in ("generate", "run", "ablate", "report", "check", "demo"):
        sp = sub.add_parser(name)
        sp.add_argument("--n", type=int, default=10, help="documents per regime")
        if name == "generate":
            sp.add_argument("--output", default=None)
        if name == "report":
            sp.add_argument("--output", default=None)
        if name in ("run", "check"):
            sp.add_argument("--regime", default=None)
    sp = sub.add_parser("serve")
    sp.add_argument("--port", type=int, default=8765)
    args = p.parse_args(argv)
    funcs = {"generate": cmd_generate, "run": cmd_run, "ablate": cmd_ablate,
             "report": cmd_report, "check": cmd_check, "demo": cmd_demo,
             "serve": cmd_serve}
    return funcs[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
