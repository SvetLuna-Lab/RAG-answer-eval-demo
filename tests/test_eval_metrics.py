import unittest

from eval_metrics import keyword_coverage, context_overlap, compute_score


class TestEvalMetrics(unittest.TestCase):
    def test_keyword_coverage_full(self):
        answer = "The pipeline has a retriever and a generator."
        expected_keywords = ["retriever", "generator"]
        cov = keyword_coverage(answer, expected_keywords)
        self.assertAlmostEqual(cov, 1.0)

    def test_keyword_coverage_partial(self):
        answer = "The pipeline uses a retriever component."
        expected_keywords = ["retriever", "generator"]
        cov = keyword_coverage(answer, expected_keywords)
        self.assertAlmostEqual(cov, 0.5)

    def test_keyword_coverage_zero(self):
        answer = "No relevant terms here."
        expected_keywords = ["retriever", "generator"]
        cov = keyword_coverage(answer, expected_keywords)
        self.assertAlmostEqual(cov, 0.0)

    def test_context_overlap_basic(self):
        answer = "retrieval augmented generation"
        context = "retrieval augmented generation combines context and question."
        ov = context_overlap(answer, context)
        # все три токена есть в контексте
        self.assertAlmostEqual(ov, 1.0)

    def test_context_overlap_partial(self):
        answer = "retrieval augmented something else"
        context = "retrieval augmented generation combines context and question."
        ov = context_overlap(answer, context)
        # 3 токена из 4 совпадают
        self.assertAlmostEqual(ov, 3.0 / 4.0)

    def test_compute_score_balanced(self):
        cov = 0.8
        ov = 0.4
        score = compute_score(cov, ov, alpha=0.5)
        self.assertAlmostEqual(score, 0.6)

    def test_compute_score_weighted(self):
        cov = 0.8
        ov = 0.4
        score = compute_score(cov, ov, alpha=0.7)
        # 0.7*0.8 + 0.3*0.4 = 0.56 + 0.12 = 0.68
        self.assertAlmostEqual(score, 0.68)


if __name__ == "__main__":
    unittest.main()
