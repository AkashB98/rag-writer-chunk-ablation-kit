# rag-writer-chunk-ablation-kit

A factorial ablation harness for retrieval-augmented generation (RAG) pipelines: it holds a synthetic corpus fixed and varies **retriever × writer × chunk-expansion width** one factor at a time, so answer quality can be *attributed* to the factor that actually caused it — before anyone re-engineers the wrong component.

**What it proves about its author:** ablation discipline. When a RAG system underperforms, the expensive mistake is rebuilding retrieval when the writer or the chunking was the problem (or vice versa). This kit demonstrates the controlled-experiment habit — planted ground truth, one factor varied at a time, a pre-stated headline hypothesis, and an honest write-up of what the synthetic setup can and cannot show.

> Distinct from `hybrid-retrieval-fusion-eval-kit`, which is a retriever-fusion bake-off (which fusion wins); this kit ablates retriever, writer, and chunk expansion against each other to find *which stage* owns the quality.

## Inspiration

Inspired by the pre-submission ablation findings reported in Zendel et al., *RMIT-ADM+S at TREC RAG 2026* ([arXiv:2611.16425](https://arxiv.org/abs/2611.16425), submitted 12 Nov 2026), where writer selection and adjacent-chunk expansion were the main factors affecting answer quality in their system. This kit is an independent, synthetic exploration of that factor-attribution question — not a replication (see Honest limits).

## The experiment

Two synthetic corpus regimes, both generated deterministically with known ground truth (10 documents per regime, 7 chunks of 4 sentences each):

- **Regime S — spread evidence.** Each answer's 3 facts sit in *adjacent* chunks (2, 3, 4). Queries use the documents' own vocabulary, so both retrievers find the centre chunk easily.
- **Regime M — lexical mismatch.** Each answer's 3 facts sit in a *single* chunk, but queries are written with synonyms that barely overlap the document wording (`swift` for `rapid`, `earnings` for `revenue`, …). A distractor chunk is rich in the query's synonym words but holds no facts.

Factors (full factorial, 24 cells):

| Factor | Levels |
|---|---|
| Retriever | `exact_bm25` — exact-token BM25-lite · `expanded_tfidf` — TF-IDF with a small fixed synonym table expanding query terms |
| Writer | `terse` — terse-faithful extractive stand-in, top chunk only · `verbose` — verbose-inclusive extractive stand-in, all expanded context plus unsupported filler |
| Chunk expansion | width 0 / ±1 / ±2 neighbouring chunks around the top hit |

Retrievers rank chunks *within* the query's source document — document retrieval is out of scope; chunk selection is what is being ablated.

**Scoring** (defined once, used everywhere): correctness = fraction of the query's ground-truth facts present in the answer; faithfulness = fraction of answer sentences supported by the retrieved+expanded context; **quality = (correctness + faithfulness) / 2**. Extractive writers copy sentences verbatim, so fact presence is tracked by sentence provenance.

**Factor attribution** per regime: retriever delta = mean quality(expanded) − mean quality(exact); writer delta = mean quality(verbose) − mean quality(terse); expansion delta = mean quality(width ±2) − mean quality(width 0).

## Results — the reversal is the centrepiece

| Regime | Retriever delta | Writer delta | Expansion delta | Dominant factor |
|---|---|---|---|---|
| S (spread evidence) | 0.000 | +0.168 | **+0.205** | writer / expansion |
| M (lexical mismatch) | **+0.500** | −0.056 | +0.035 | retriever |

**The dominant factor flips with the regime.** When evidence is spread and vocabulary matches, switching retrievers buys nothing (delta 0.000) — widening the chunk window and letting the writer use it is where the quality is. When the query wording mismatches the document, no writer or window rescues a retriever that fetched the wrong chunk — the retriever delta (+0.500) dwarfs everything else. Same pipeline, opposite prescription: that is why you ablate before you rebuild.

### 24-cell quality table (mean over 10 queries per regime)

| Regime | Retriever | Writer | Width | Correctness | Faithfulness | Quality |
|---|---|---|---|---|---|---|
| M | exact_bm25 | terse | 0 | 0.000 | 1.000 | 0.500 |
| M | exact_bm25 | terse | 1 | 0.000 | 1.000 | 0.500 |
| M | exact_bm25 | terse | 2 | 0.000 | 1.000 | 0.500 |
| M | exact_bm25 | verbose | 0 | 0.000 | 0.800 | 0.400 |
| M | exact_bm25 | verbose | 1 | 0.000 | 0.923 | 0.462 |
| M | exact_bm25 | verbose | 2 | 0.000 | 0.941 | 0.471 |
| M | expanded_tfidf | terse | 0 | 1.000 | 1.000 | 1.000 |
| M | expanded_tfidf | terse | 1 | 1.000 | 1.000 | 1.000 |
| M | expanded_tfidf | terse | 2 | 1.000 | 1.000 | 1.000 |
| M | expanded_tfidf | verbose | 0 | 1.000 | 0.800 | 0.900 |
| M | expanded_tfidf | verbose | 1 | 1.000 | 0.923 | 0.962 |
| M | expanded_tfidf | verbose | 2 | 1.000 | 0.941 | 0.971 |
| S | exact_bm25 | terse | 0 | 0.333 | 1.000 | 0.667 |
| S | exact_bm25 | terse | 1 | 0.333 | 1.000 | 0.667 |
| S | exact_bm25 | terse | 2 | 0.333 | 1.000 | 0.667 |
| S | exact_bm25 | verbose | 0 | 0.333 | 0.800 | 0.567 |
| S | exact_bm25 | verbose | 1 | 1.000 | 0.923 | 0.962 |
| S | exact_bm25 | verbose | 2 | 1.000 | 0.952 | 0.976 |
| S | expanded_tfidf | terse | 0 | 0.333 | 1.000 | 0.667 |
| S | expanded_tfidf | terse | 1 | 0.333 | 1.000 | 0.667 |
| S | expanded_tfidf | terse | 2 | 0.333 | 1.000 | 0.667 |
| S | expanded_tfidf | verbose | 0 | 0.333 | 0.800 | 0.567 |
| S | expanded_tfidf | verbose | 1 | 1.000 | 0.923 | 0.962 |
| S | expanded_tfidf | verbose | 2 | 1.000 | 0.952 | 0.976 |

## How to run

Python 3, standard library only. No installs, no network, no API keys.

```bash
python3 cli.py demo            # end-to-end: generate, ablate, print the reversal
python3 cli.py generate --output corpus.json
python3 cli.py run             # full 24-cell table as JSON (--regime S|M to filter)
python3 cli.py ablate          # cells + factor deltas + reversal verdict as JSON
python3 cli.py report          # markdown report (add --output report.md to save)
python3 cli.py check           # exit 0 = reversal holds, 1 = it does not, 2 = bad input
python3 cli.py serve --port 8765   # GET /health, POST /ablate {"n": 10} on 127.0.0.1
python3 run_tests.py           # 47 hermetic tests
python3 evals/run_evals.py     # 12 golden evals -> evals/eval_report.json
```

## Honest limits

No LLM writers — the two writers are deterministic stand-ins. This does not replicate TREC RAG 2026; effect sizes are artifacts of the synthetic setup.

In plain terms: the corpora are planted so each regime stresses one stage of the pipeline, the "writers" are extractive sentence copiers with opposite strictness (no language model is called anywhere), the retrievers are deliberately minimal teaching implementations (no RRF, no router, no embeddings), and retrieval is scoped to chunks within the known source document. What transfers to real systems is the method — factorial design, ground-truth scoring, factor attribution, and a headline claim stated before the run — not the numbers.

## POC framing

This is a proof of concept for evaluation methodology, built to be read and re-run in minutes. A production version would swap the synthetic generator for a labelled real corpus, the stand-in writers for real LLM writers behind the same scoring interface, and add embedding/hybrid retrievers as additional factor levels — the factorial table, deltas, and `check` gate would not change.

## Dev loop

Built in one tight loop: generator first (planted structure verified by unit tests on fact placement), retrievers second, then the factorial — with the reversal asserted by `check` and the golden evals from the first green run onward. If the reversal had failed, the fix would have belonged in the generator's planted structure, never in the scorer.

## License

MIT — see `LICENSE` (Akash Bhargava).
