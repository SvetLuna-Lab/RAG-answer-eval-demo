from dataclasses import dataclass
from typing import List, Tuple

from retriever import SimpleBM25Retriever, Document


@dataclass
class RetrievedContext:
    doc_id: str
    text: str
    score: float


class RagPipeline:
    """
    Minimal RAG pipeline:
    - retriever: SimpleBM25Retriever
    - answer_stub: answer produced by simply extracting text from the top-1 document
    """

    def __init__(self, corpus_dir: str) -> None:
        self.retriever = SimpleBM25Retriever(corpus_dir=corpus_dir)

    def retrieve(self, question: str, top_k: int = 3) -> List[RetrievedContext]:
        results: List[Tuple[Document, float]] = self.retriever.retrieve(question, top_k=top_k)
        return [
            RetrievedContext(doc_id=doc.doc_id, text=doc.text, score=score)
            for doc, score in results
        ]

    def answer_stub(self, question: str, contexts: List[RetrievedContext], max_chars: int = 400) -> str:
        """
        Simple stub instead of an LLM:
        - take the text from the top-1 retrieved document and use it as an answer,
          slightly formatted.
        - in the future this can be replaced with a real LLM client.
        """
        if not contexts:
            return "No answer: no context retrieved."

        top = contexts[0]
        snippet = top.text.strip().replace("\n", " ")
        if len(snippet) > max_chars:
            snippet = snippet[: max_chars].rsplit(" ", 1)[0] + "..."

        return f"{snippet}"

    def run(self, question: str, top_k: int = 3) -> tuple[str, List[RetrievedContext]]:
        """
        Convenience method:
        - retrieve context for the question,
        - produce an answer using the stub generator.
        """
        contexts = self.retrieve(question, top_k=top_k)
        answer = self.answer_stub(question, contexts)
        return answer, contexts
