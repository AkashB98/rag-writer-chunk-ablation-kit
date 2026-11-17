"""Core logic for rag-writer-chunk-ablation-kit.

A factorial ablation harness over synthetic RAG corpora. No LLM calls,
Python standard library only, fully deterministic.

Scoring (defined once, used everywhere):
  correctness  = fraction of a query's ground-truth facts present in the answer
  faithfulness = fraction of answer sentences supported by retrieved+expanded context
  quality      = (correctness + faithfulness) / 2
"""
import math
import re
from collections import Counter

STOPWORDS = {
    "a", "an", "the", "did", "does", "do", "what", "how", "were", "was", "is",
    "are", "at", "in", "on", "of", "for", "to", "and", "or", "by", "with",
    "during", "after", "before", "its", "it", "this", "that", "while",
}

# Fixed synonym table: query term -> document term.
# Regime M pairs map a query word to a *present* document word
# (swift->rapid, earnings->revenue, expansion->growth, produce->output).
# Regime S pairs map to words that do NOT occur in the corpus, so expanding
# a Regime S query changes nothing -- both retrievers rank alike there.
SYNONYM_TABLE = {
    "swift": "rapid",
    "earnings": "revenue",
    "expansion": "growth",
    "produce": "output",
    "efficiency": "productivity",
    "measurement": "assessment",
    "calibration": "tuning",
}

ENTITIES = [
    "Aurora", "Borealis", "Cedar", "Delta", "Ember", "Falcon",
    "Granite", "Harbor", "Ivory", "Juniper",
]
TOPICS = [
    "solar farm", "tidal station", "forest nursery", "wind turbine",
    "river gauge", "soil lab", "bee colony", "glacier post",
    "desert well", "coastal reef",
]

UNSUPPORTED_FILLER = (
    "In broader terms, this subject continues to attract attention "
    "from many observers."
)

REGIMES = ("S", "M")
RETRIEVERS = ("exact_bm25", "expanded_tfidf")
WRITERS = ("terse", "verbose")
WIDTHS = (0, 1, 2)


def tokenize(text):
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in STOPWORDS]


class Sentence:
    def __init__(self, text, fact_ids=()):
        self.text = text
        self.fact_ids = tuple(fact_ids)

    def to_dict(self):
        return {"text": self.text, "fact_ids": list(self.fact_ids)}


class Chunk:
    def __init__(self, doc_id, index, sentences):
        self.doc_id = doc_id
        self.index = index
        self.sentences = list(sentences)

    @property
    def text(self):
        return " ".join(s.text for s in self.sentences)

    def fact_ids(self):
        out = set()
        for s in self.sentences:
            out.update(s.fact_ids)
        return out


class Query:
    def __init__(self, qid, doc_id, text, fact_ids):
        self.qid = qid
        self.doc_id = doc_id
        self.text = text
        self.fact_ids = tuple(fact_ids)


def _generic_sentences(entity, topic, variant):
    """Four safe context sentences: no regime key words in them."""
    templates = [
        f"The daily log at the {entity} {topic} noted routine checks and steady staffing.",
        f"Staff at the {entity} {topic} filed the weekly summary and updated the notice board.",
        f"A short visit to the {entity} {topic} confirmed the site was tidy and well kept.",
        f"The {entity} {topic} team closed the day with a brief review of pending tasks.",
        f"Records for the {entity} {topic} were archived in the main office cabinet.",
        f"The {entity} {topic} schedule for the coming week was posted near the entrance.",
    ]
    # rotate deterministically by variant
    return [templates[(variant + k) % len(templates)] for k in range(4)]


def _doc_id(regime, i):
    return f"{regime}-{i:02d}"


def generate_regime_s(i):
    """Spread evidence: 3 facts in adjacent chunks 2,3,4; query uses doc vocab."""
    entity, topic = ENTITIES[i], TOPICS[i]
    doc = _doc_id("S", i)
    e, t = entity, topic
    v1, v2, v3 = 20 + i, 30 + 2 * i, 5 + i
    fid = lambda n: f"{doc}-F{n}"  # noqa: E731
    chunks = []
    # chunk 0,1,5,6: generic context
    for idx in (0, 1, 5, 6):
        chunks.append(Chunk(doc, idx, [Sentence(s) for s in _generic_sentences(e, t, idx + i)]))
    # chunk 2: fact F2 (+ one mild 'efficiency' mention spread over the chunk)
    c2 = [
        Sentence(f"The {e} {t} stored {v2} units of reserve before the main trial.", (fid(2),)),
        Sentence(f"One efficiency note for the {e} {t} was pinned to the reserve ledger."),
        Sentence(f"Reserve crates at the {e} {t} were counted and labelled by the morning crew."),
        Sentence(f"The {e} {t} reserve area was swept and checked before closing."),
    ]
    # chunk 3: centre chunk, fact F1, dense in query terms
    centre_ctx = [
        "Efficiency measurement and calibration checks were logged every morning.",
        "The calibration team repeated the efficiency measurement at midday.",
        "Efficiency, measurement, and calibration results were entered in the main register.",
    ]
    c3 = [Sentence(f"The {e} {t} recorded a core efficiency of {v1} percent during the main trial.", (fid(1),))]
    c3 += [Sentence(f"{s[:-1]} at the {e} {t}.") for s in centre_ctx]
    # chunk 4: fact F3
    c4 = [
        Sentence(f"The {e} {t} reduced losses to {v3} percent after the main trial.", (fid(3),)),
        Sentence(f"Loss figures for the {e} {t} were copied into the weekly ledger."),
        Sentence(f"The {e} {t} crew inspected the seals and fittings after the trial."),
        Sentence(f"A short note about the {e} {t} trial outcome was filed with the office."),
    ]
    by_idx = {c.index: c for c in chunks}
    by_idx[2] = Chunk(doc, 2, c2)
    by_idx[3] = Chunk(doc, 3, c3)
    by_idx[4] = Chunk(doc, 4, c4)
    ordered = [by_idx[k] for k in sorted(by_idx)]
    query = Query(f"{doc}-Q", doc,
                  f"What efficiency measurement calibration results did the {e} {t} record?",
                  (fid(1), fid(2), fid(3)))
    return ordered, query


def generate_regime_m(i):
    """Lexical mismatch: all 3 facts in single chunk 1; query uses synonyms.
    A distractor chunk (5) is rich in the query's synonym words but has no facts."""
    entity, topic = ENTITIES[i], TOPICS[i]
    doc = _doc_id("M", i)
    e, t = entity, topic
    v = 40 + 3 * i
    fid = lambda n: f"{doc}-F{n}"  # noqa: E731
    chunks = []
    for idx in (0, 2, 3, 4, 6):
        chunks.append(Chunk(doc, idx, [Sentence(s) for s in _generic_sentences(e, t, idx + i + 2)]))
    # chunk 3 gets at most two document key words total (keeps its score low)
    # (generic sentences contain none, so this is satisfied by construction)
    target = [
        Sentence(f"The {e} {t} showed rapid revenue growth after the new programme began.", (fid(1),)),
        Sentence(f"The {e} {t} raised output to {v} units in the final quarter.", (fid(2),)),
        Sentence(f"Rapid checks confirmed the {e} {t} kept revenue steady while output climbed.", (fid(3),)),
        Sentence(f"Observers recorded rapid revenue, steady growth, and rising output at the {e} {t} throughout the period."),
    ]
    distractor_texts = [
        f"Commentators used words like swift and earnings when discussing swift expansion plans near the {e} {t}.",
        f"A visiting speaker mentioned swift produce, swift earnings, and expansion rumours at the {e} {t} hall.",
        f"The newsletter joked about swift earnings and swift produce arriving before the expansion fair at the {e} {t}.",
        f"Visitors asked about swift expansion, produce stalls, and earnings displays outside the {e} {t}.",
    ]
    by_idx = {c.index: c for c in chunks}
    by_idx[1] = Chunk(doc, 1, target)
    by_idx[5] = Chunk(doc, 5, [Sentence(s) for s in distractor_texts])
    ordered = [by_idx[k] for k in sorted(by_idx)]
    query = Query(f"{doc}-Q", doc,
                  f"How swift were the earnings, expansion, and produce for the {e} {t}?",
                  (fid(1), fid(2), fid(3)))
    return ordered, query


def generate_corpus(n_per_regime=10):
    """Return (docs, queries): docs = {doc_id: [Chunk,...]}, queries = [Query]."""
    docs, queries = {}, []
    for i in range(n_per_regime):
        chunks, q = generate_regime_s(i)
        docs[q.doc_id] = chunks
        queries.append(q)
    for i in range(n_per_regime):
        chunks, q = generate_regime_m(i)
        docs[q.doc_id] = chunks
        queries.append(q)
    return docs, queries


def corpus_to_dict(docs, queries):
    return {
        "documents": [
            {"doc_id": d, "chunks": [
                {"index": c.index, "sentences": [s.to_dict() for s in c.sentences]}
                for c in docs[d]]}
            for d in sorted(docs)
        ],
        "queries": [
            {"qid": q.qid, "doc_id": q.doc_id, "text": q.text,
             "fact_ids": list(q.fact_ids)} for q in queries
        ],
    }


# ---------------- retrievers (rank chunks within the query's document) ----

def _idf(chunks):
    n = len(chunks)
    df = Counter()
    for c in chunks:
        for tok in set(tokenize(c.text)):
            df[tok] += 1
    return {tok: math.log(1 + (n - cnt + 0.5) / (cnt + 0.5)) for tok, cnt in df.items()}


def retrieve_exact_bm25(query_text, chunks):
    """BM25-lite: exact tokens only, k1=1.2, b=0.75. Returns top-1 chunk."""
    qtoks = tokenize(query_text)
    idf = _idf(chunks)
    avg_len = sum(len(tokenize(c.text)) for c in chunks) / max(1, len(chunks))
    best, best_score = None, None
    for c in chunks:
        counts = Counter(tokenize(c.text))
        length = max(1, sum(counts.values()))
        score = 0.0
        for tok in qtoks:
            tf = counts.get(tok, 0)
            if tf:
                denom = tf + 1.2 * (1 - 0.75 + 0.75 * length / avg_len)
                score += idf.get(tok, 0.0) * (tf * 2.2 / denom)
        if best_score is None or score > best_score:
            best, best_score = c, score
    return best


def retrieve_expanded_tfidf(query_text, chunks):
    """TF-IDF with synonym expansion: original terms weight 1, synonyms weight 2."""
    qtoks = tokenize(query_text)
    qvec = Counter()
    for tok in qtoks:
        qvec[tok] += 1.0
        if tok in SYNONYM_TABLE:
            qvec[SYNONYM_TABLE[tok]] += 2.0
    idf = _idf(chunks)
    best, best_score = None, None
    for c in chunks:
        counts = Counter(tokenize(c.text))
        length = math.sqrt(max(1, sum(counts.values())))
        score = 0.0
        for tok, qw in qvec.items():
            tf = counts.get(tok, 0)
            if tf:
                score += qw * tf * idf.get(tok, 0.0)
        score /= length
        if best_score is None or score > best_score:
            best, best_score = c, score
    return best


RETRIEVER_FUNCS = {"exact_bm25": retrieve_exact_bm25, "expanded_tfidf": retrieve_expanded_tfidf}


# ---------------- expansion & writers --------------------------------------

def expand_chunks(doc_chunks, top_chunk, width):
    if width not in WIDTHS:
        raise ValueError(f"width must be one of {WIDTHS}")
    ordered = sorted(doc_chunks, key=lambda c: c.index)
    pos = [c.index for c in ordered].index(top_chunk.index)
    lo = max(0, pos - width)
    hi = min(len(ordered), pos + width + 1)
    return ordered[lo:hi]


class AnswerSentence:
    def __init__(self, text, supported, fact_ids=()):
        self.text = text
        self.supported = supported
        self.fact_ids = tuple(fact_ids)


def write_terse(top_chunk):
    """Terse-faithful stand-in: only sentences from the top-ranked chunk."""
    return [AnswerSentence(s.text, True, s.fact_ids) for s in top_chunk.sentences]


def write_verbose(expanded_chunks):
    """Verbose-inclusive stand-in: all expanded context + 1 unsupported filler."""
    out = []
    for c in expanded_chunks:
        for s in c.sentences:
            out.append(AnswerSentence(s.text, True, s.fact_ids))
    out.append(AnswerSentence(UNSUPPORTED_FILLER, False, ()))
    return out


WRITER_NAMES = WRITERS


# ---------------- scoring ----------------------------------------------------

def score_answer(answer, ground_truth_fact_ids):
    gt = set(ground_truth_fact_ids)
    present = set()
    for s in answer:
        present.update(s.fact_ids)
    correctness = (len(gt & present) / len(gt)) if gt else 0.0
    faithfulness = (sum(1 for s in answer if s.supported) / len(answer)) if answer else 0.0
    quality = (correctness + faithfulness) / 2.0
    return {"correctness": correctness, "faithfulness": faithfulness, "quality": quality}


# ---------------- factorial --------------------------------------------------

def run_factorial(n_per_regime=10):
    docs, queries = generate_corpus(n_per_regime)
    cells = {}  # (regime, retriever, writer, width) -> list of score dicts
    for q in queries:
        regime = q.doc_id.split("-")[0]
        doc_chunks = docs[q.doc_id]
        for rname in RETRIEVERS:
            top = RETRIEVER_FUNCS[rname](q.text, doc_chunks)
            for width in WIDTHS:
                expanded = expand_chunks(doc_chunks, top, width)
                for wname in WRITERS:
                    ans = write_terse(top) if wname == "terse" else write_verbose(expanded)
                    cells.setdefault((regime, rname, wname, width), []).append(
                        score_answer(ans, q.fact_ids))
    table = {}
    for key, scores in sorted(cells.items()):
        n = len(scores)
        table[key] = {
            "correctness": round(sum(s["correctness"] for s in scores) / n, 6),
            "faithfulness": round(sum(s["faithfulness"] for s in scores) / n, 6),
            "quality": round(sum(s["quality"] for s in scores) / n, 6),
            "n": n,
        }
    return table


def factor_deltas(table):
    """Per regime: retriever delta (expanded-exact), writer delta (verbose-terse),
    expansion delta (width2-width0); each averaged over the other factors."""
    out = {}
    for regime in REGIMES:
        def mean(pred):
            vals = [v["quality"] for k, v in table.items() if k[0] == regime and pred(k)]
            return sum(vals) / len(vals) if vals else 0.0
        out[regime] = {
            "retriever_delta": round(mean(lambda k: k[1] == "expanded_tfidf") - mean(lambda k: k[1] == "exact_bm25"), 6),
            "writer_delta": round(mean(lambda k: k[2] == "verbose") - mean(lambda k: k[2] == "terse"), 6),
            "expansion_delta": round(mean(lambda k: k[3] == 2) - mean(lambda k: k[3] == 0), 6),
        }
    return out


def reversal_holds(deltas):
    s, m = deltas["S"], deltas["M"]
    s_ok = max(s["writer_delta"], s["expansion_delta"]) > s["retriever_delta"]
    m_ok = m["retriever_delta"] > max(m["writer_delta"], m["expansion_delta"])
    return {"S": s_ok, "M": m_ok, "holds": s_ok and m_ok}
