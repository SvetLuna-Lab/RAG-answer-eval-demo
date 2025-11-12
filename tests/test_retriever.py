import os
import unittest

from retriever import SimpleBM25Retriever


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CORPUS_DIR = os.path.join(PROJECT_ROOT, "data", "corpus")


class TestSimpleBM25Retriever(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.retriever = SimpleBM25Retriever(corpus_dir=CORPUS_DIR)

    def test_retrieves_doc1_for_basic_rag_question(self):
        query = "What are the components of a RAG pipeline?"
        results = self.retriever.retrieve(query, top_k=1)
        self.assertGreater(len(results), 0)
        top_doc, score = results[0]
        self.assertEqual(top_doc.doc_id, "doc1.txt")
        self.assertGreater(score, 0.0)

    def test_retrieves_doc2_for_metrics_question(self):
        query = "Which simple metrics are used to evaluate RAG answers?"
        results = self.retriever.retrieve(query, top_k=3)
        doc_ids = [doc.doc_id for doc, _ in results]
        self.assertIn("doc2.txt", doc_ids)


if __name__ == "__main__":
    unittest.main()
