import math
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass
class Document:
    doc_id: str
    text: str


class SimpleBM25Retriever:
    """
    Very simple BM25-like retriever over a bag-of-words representation.
    Good enough for a small evaluation harness without external dependencies.
    """

    def __init__(self, corpus_dir: str, k1: float = 1.5, b: float = 0.75) -> None:
        self.corpus_dir = corpus_dir
        self.k1 = k1
        self.b = b

        self.documents: List[Document] = []
        self.doc_lengths: Dict[str, int] = {}
        self.avg_doc_len: float = 0.0
        self.df: Dict[str, int] = {}  # document frequency per term
        self.N: int = 0               # number of documents in the corpus

        self._load_corpus()
        self._build_index()

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        # Simple tokenization: split on spaces, lowercase, and strip non-alphanumeric chars
        tokens = []
        for raw in text.lower().split():
            token = "".join(ch for ch in raw if ch.isalnum())
            if token:
                tokens.append(token)
        return tokens

    def _load_corpus(self) -> None:
        # Load all .txt files from the corpus directory into memory
        for fname in os.listdir(self.corpus_dir):
            if not fname.endswith(".txt"):
                continue
            path = os.path.join(self.corpus_dir, fname)
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()
            self.documents.append(Document(doc_id=fname, text=text))

        self.N = len(self.documents)

    def _build_index(self) -> None:
        # Build simple statistics for BM25: document lengths, avg document length, and DF per term
        total_len = 0
        self.df.clear()
        self.doc_lengths.clear()

        for doc in self.documents:
            tokens = self._tokenize(doc.text)
            total_len += len(tokens)
            self.doc_lengths[doc.doc_id] = len(tokens)

            seen_terms = set(tokens)
            for term in seen_terms:
                self.df[term] = self.df.get(term, 0) + 1

        self.avg_doc_len = total_len / self.N if self.N > 0 else 0.0

    def _bm25_score(self, query_tokens: List[str], doc: Document) -> float:
        if self.N == 0:
            return 0.0

        doc_tokens = self._tokenize(doc.text)
        doc_len = len(doc_tokens)
        if doc_len == 0:
            return 0.0

        # term -> term frequency in this document
        tf: Dict[str, int] = {}
        for t in doc_tokens:
            tf[t] = tf.get(t, 0) + 1

        score = 0.0
        for term in query_tokens:
            if term not in tf:
                continue
            df = self.df.get(term, 0)
            if df == 0:
                continue

            # Inverse document frequency with BM25-style smoothing
            idf = math.log((self.N - df + 0.5) / (df + 0.5) + 1.0)
            freq = tf[term]
            denom = freq + self.k1 * (1 - self.b + self.b * doc_len / (self.avg_doc_len or 1.0))
            score += idf * (freq * (self.k1 + 1) / denom)

        return score

    def retrieve(self, query: str, top_k: int = 3) -> List[Tuple[Document, float]]:
        """
        Compute BM25-like scores for all documents in the corpus
        and return the top_k documents with their scores.
        """
        tokens = self._tokenize(query)
        scores: List[Tuple[Document, float]] = []
        for doc in self.documents:
            s = self._bm25_score(tokens, doc)
            scores.append((doc, s))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]
