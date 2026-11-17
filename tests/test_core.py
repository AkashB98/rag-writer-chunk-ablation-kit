import sys, os, unittest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from rag_ablation.core import (
    SYNONYM_TABLE, expand_chunks, factor_deltas, generate_corpus,
    generate_regime_m, generate_regime_s, retrieve_exact_bm25,
    retrieve_expanded_tfidf, reversal_holds, run_factorial, score_answer,
    tokenize, write_terse, write_verbose, AnswerSentence,
)


class TestTokenize(unittest.TestCase):
    def test_lowercases_and_splits(self):
        self.assertEqual(tokenize("Rapid Revenue!"), ["rapid", "revenue"])

    def test_removes_stopwords(self):
        self.assertNotIn("the", tokenize("the rapid revenue"))

    def test_empty(self):
        self.assertEqual(tokenize(""), [])


class TestGeneration(unittest.TestCase):
    def setUp(self):
        self.docs, self.queries = generate_corpus(10)

    def test_counts(self):
        self.assertEqual(len(self.docs), 20)
        self.assertEqual(len(self.queries), 20)

    def test_each_doc_has_7_chunks_of_4_sentences(self):
        for chunks in self.docs.values():
            self.assertEqual(len(chunks), 7)
            for c in chunks:
                self.assertEqual(len(c.sentences), 4)

    def test_deterministic(self):
        docs2, q2 = generate_corpus(10)
        self.assertEqual([q.text for q in q2], [q.text for q in self.queries])
        self.assertEqual(docs2["S-00"][3].text, self.docs["S-00"][3].text)

    def test_regime_s_facts_spread_adjacent(self):
        chunks, q = generate_regime_s(0)
        fact_chunks = sorted({c.index for c in chunks if c.fact_ids()})
        self.assertEqual(fact_chunks, [2, 3, 4])
        self.assertEqual(len(q.fact_ids), 3)

    def test_regime_m_facts_single_chunk(self):
        chunks, q = generate_regime_m(0)
        fact_chunks = sorted({c.index for c in chunks if c.fact_ids()})
        self.assertEqual(fact_chunks, [1])
        self.assertEqual(len(chunks[1].fact_ids()), 3)

    def test_regime_s_query_uses_doc_vocab(self):
        chunks, q = generate_regime_s(0)
        doc_text = " ".join(c.text for c in chunks)
        for tok in ("efficiency", "measurement", "calibration"):
            self.assertIn(tok, tokenize(q.text))
            self.assertIn(tok, tokenize(doc_text))

    def test_regime_m_query_doc_mismatch(self):
        chunks, q = generate_regime_m(0)
        target_text = chunks[1].text
        for qt, dt in (("swift", "rapid"), ("earnings", "revenue"),
                       ("expansion", "growth"), ("produce", "output")):
            self.assertIn(qt, tokenize(q.text))
            self.assertIn(dt, tokenize(target_text))
            self.assertNotIn(qt, tokenize(target_text))

    def test_synonym_table_maps_m_pairs(self):
        self.assertEqual(SYNONYM_TABLE["swift"], "rapid")
        self.assertEqual(SYNONYM_TABLE["earnings"], "revenue")
        self.assertEqual(SYNONYM_TABLE["expansion"], "growth")
        self.assertEqual(SYNONYM_TABLE["produce"], "output")

    def test_ground_truth_facts_exist_in_corpus(self):
        for q in self.queries:
            corpus_facts = set()
            for c in self.docs[q.doc_id]:
                corpus_facts.update(c.fact_ids())
            self.assertEqual(set(q.fact_ids), corpus_facts & set(q.fact_ids))
            self.assertEqual(len(q.fact_ids), 3)


class TestRetrievers(unittest.TestCase):
    def setUp(self):
        self.docs, self.queries = generate_corpus(10)

    def _q(self, regime, i=0):
        return [q for q in self.queries if q.doc_id == f"{regime}-{i:02d}"][0]

    def test_exact_picks_centre_in_s(self):
        q = self._q("S")
        self.assertEqual(retrieve_exact_bm25(q.text, self.docs[q.doc_id]).index, 3)

    def test_expanded_picks_centre_in_s(self):
        q = self._q("S")
        self.assertEqual(retrieve_expanded_tfidf(q.text, self.docs[q.doc_id]).index, 3)

    def test_exact_picks_distractor_in_m(self):
        q = self._q("M")
        self.assertEqual(retrieve_exact_bm25(q.text, self.docs[q.doc_id]).index, 5)

    def test_expanded_picks_target_in_m(self):
        q = self._q("M")
        self.assertEqual(retrieve_expanded_tfidf(q.text, self.docs[q.doc_id]).index, 1)

    def test_retrievers_agree_in_s_all_docs(self):
        for q in self.queries:
            if q.doc_id.startswith("S"):
                a = retrieve_exact_bm25(q.text, self.docs[q.doc_id]).index
                b = retrieve_expanded_tfidf(q.text, self.docs[q.doc_id]).index
                self.assertEqual(a, b)

    def test_retrievers_disagree_in_m_all_docs(self):
        for q in self.queries:
            if q.doc_id.startswith("M"):
                a = retrieve_exact_bm25(q.text, self.docs[q.doc_id]).index
                b = retrieve_expanded_tfidf(q.text, self.docs[q.doc_id]).index
                self.assertNotEqual(a, b)


class TestExpansionAndWriters(unittest.TestCase):
    def setUp(self):
        self.docs, self.queries = generate_corpus(4)

    def test_expansion_widths(self):
        chunks, q = generate_regime_s(0)
        top = chunks[3]
        self.assertEqual(len(expand_chunks(chunks, top, 0)), 1)
        self.assertEqual(len(expand_chunks(chunks, top, 1)), 3)
        self.assertEqual(len(expand_chunks(chunks, top, 2)), 5)

    def test_expansion_clips_at_edge(self):
        chunks, q = generate_regime_m(0)
        self.assertEqual(len(expand_chunks(chunks, chunks[1], 2)), 4)
        self.assertEqual(len(expand_chunks(chunks, chunks[6], 2)), 3)

    def test_expansion_bad_width(self):
        chunks, q = generate_regime_s(0)
        with self.assertRaises(ValueError):
            expand_chunks(chunks, chunks[0], 3)

    def test_terse_only_top_chunk_all_supported(self):
        chunks, q = generate_regime_s(0)
        ans = write_terse(chunks[3])
        self.assertEqual(len(ans), 4)
        self.assertTrue(all(s.supported for s in ans))

    def test_verbose_includes_filler_unsupported(self):
        chunks, q = generate_regime_s(0)
        ans = write_verbose(expand_chunks(chunks, chunks[3], 1))
        self.assertEqual(len(ans), 13)
        self.assertEqual(sum(1 for s in ans if not s.supported), 1)

    def test_verbose_covers_spread_facts_at_width1(self):
        chunks, q = generate_regime_s(0)
        ans = write_verbose(expand_chunks(chunks, chunks[3], 1))
        sc = score_answer(ans, q.fact_ids)
        self.assertEqual(sc["correctness"], 1.0)

    def test_terse_misses_spread_facts(self):
        chunks, q = generate_regime_s(0)
        sc = score_answer(write_terse(chunks[3]), q.fact_ids)
        self.assertAlmostEqual(sc["correctness"], 1 / 3, places=6)


class TestScoring(unittest.TestCase):
    def test_quality_is_mean(self):
        ans = [AnswerSentence("x", True, ("F1",)), AnswerSentence("y", False, ())]
        sc = score_answer(ans, ("F1", "F2"))
        self.assertAlmostEqual(sc["correctness"], 0.5)
        self.assertAlmostEqual(sc["faithfulness"], 0.5)
        self.assertAlmostEqual(sc["quality"], 0.5)

    def test_empty_answer(self):
        sc = score_answer([], ("F1",))
        self.assertEqual(sc["quality"], 0.0)

    def test_perfect_answer(self):
        ans = [AnswerSentence("x", True, ("F1", "F2"))]
        sc = score_answer(ans, ("F1", "F2"))
        self.assertEqual(sc["quality"], 1.0)


class TestFactorial(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.table = run_factorial(10)
        cls.deltas = factor_deltas(cls.table)

    def test_24_cells(self):
        self.assertEqual(len(self.table), 24)

    def test_cell_n(self):
        for v in self.table.values():
            self.assertEqual(v["n"], 10)

    def test_reversal_s_direction(self):
        d = self.deltas["S"]
        self.assertGreater(max(d["writer_delta"], d["expansion_delta"]), d["retriever_delta"])

    def test_reversal_m_direction(self):
        d = self.deltas["M"]
        self.assertGreater(d["retriever_delta"], max(d["writer_delta"], d["expansion_delta"]))

    def test_reversal_holds(self):
        self.assertTrue(reversal_holds(self.deltas)["holds"])

    def test_retriever_delta_near_zero_in_s(self):
        self.assertLess(abs(self.deltas["S"]["retriever_delta"]), 0.05)

    def test_retriever_delta_large_in_m(self):
        self.assertGreater(self.deltas["M"]["retriever_delta"], 0.3)

    def test_expansion_helps_s_more_than_m(self):
        self.assertGreater(self.deltas["S"]["expansion_delta"], self.deltas["M"]["expansion_delta"])

    def test_terse_faithfulness_ge_verbose_both_regimes(self):
        for regime in ("S", "M"):
            for ret in ("exact_bm25", "expanded_tfidf"):
                for width in (0, 1, 2):
                    t = self.table[(regime, ret, "terse", width)]["faithfulness"]
                    v = self.table[(regime, ret, "verbose", width)]["faithfulness"]
                    self.assertGreaterEqual(t, v)

    def test_factorial_deterministic(self):
        self.assertEqual(run_factorial(10), self.table)


if __name__ == "__main__":
    unittest.main()
