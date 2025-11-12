# RAG-answer-eval-demo

Small, self-contained demo of **RAG answer evaluation**.

The goal of this repository is to show, in a minimal way, how to:

- run a simple RAG-style pipeline (retrieve → generate stub answer),
- define an evaluation set with expected keywords and grounding constraints,
- compute basic metrics for answer quality and grounding.

The current “generator” is a stub that extracts text from the retrieved context.  
The code is structured so that a real LLM client can be plugged in later.

---

## Repository structure

```text
rag-answer-eval-demo/
├─ data/
│  ├─ corpus/                # tiny technical corpus (.txt)
│  │  ├─ doc1.txt
│  │  ├─ doc2.txt
│  │  └─ doc3.txt
│  └─ eval_questions.json    # questions + expected keywords + grounding info
├─ src/
│  ├─ __init__.py
│  ├─ retriever.py           # simple BM25-like retriever on bag-of-words
│  ├─ pipeline.py            # RAG pipeline (retrieve + answer_stub)
│  ├─ eval_metrics.py        # keyword coverage, context overlap, combined score
│  └─ run_eval.py            # main script: run eval-set and print/save metrics
├─ tests/
│  ├─ __init__.py
│  ├─ test_eval_metrics.py   # unit tests for metrics
│  └─ test_retriever.py      # basic sanity tests for retriever
├─ README.md
├─ requirements.txt
└─ .gitignore



Data format
Corpus

data/corpus/ contains a small set of .txt files:

doc1.txt

doc2.txt

doc3.txt

They describe basic RAG concepts, evaluation, and harness design.

Evaluation set

data/eval_questions.json is a list of question objects.
Example:


[
  {
    "id": "q1",
    "question": "What are the two main components of a RAG pipeline?",
    "expected_keywords": ["retriever", "generator"],
    "must_be_grounded_in": ["doc1.txt"]
  }
]


Fields:

id – question identifier.

question – natural language question.

expected_keywords – list of keywords that a good answer should contain.

must_be_grounded_in – list of corpus document filenames that should contain the supporting evidence for the answer.

This is enough to demonstrate both answer content and grounding checks.



Pipeline

Implemented in src/pipeline.py and src/retriever.py.

Retriever (SimpleBM25Retriever in retriever.py)

Loads all .txt documents from data/corpus/.

Uses a very simple BM25-like scoring on bag-of-words tokens.

Returns top-k documents with scores for a given query.

RAG pipeline (RagPipeline in pipeline.py)

retrieve(question, top_k) – calls the retriever and returns a list of contexts.

answer_stub(question, contexts) – simple generator stub:

takes the top-1 retrieved document,

extracts and trims its text to a given length,

returns this as the “answer”.

run(question, top_k) – convenience method:

runs retrieval and stub generation,

returns (answer, contexts).

The generator stub is intentionally simple. A real LLM can later be plugged in by replacing answer_stub with an API call while keeping the rest of the evaluation code unchanged.



Evaluation metrics

Implemented in src/eval_metrics.py.

keyword_coverage(answer, expected_keywords)

Fraction of expected_keywords that appear in the answer (case-insensitive substring match).

Range: [0.0, 1.0].

context_overlap(answer, context_text)

Token-level overlap between the answer and the reference context.

Computes the fraction of answer tokens that also appear in the context (after simple normalization).

Range: [0.0, 1.0].

Acts as a simple “non-hallucination” proxy.


**`compute_score(coverage, overlap, alpha=0.5)`**

Combined score is a weighted average of the two metrics:

```text
score = alpha * coverage + (1 - alpha) * overlap 


Default: alpha = 0.5, so both metrics are weighted equally.

evaluate_single(...)

Helper that computes all metrics for a single question and returns a dataclass EvalResult.

These metrics are intentionally simple and transparent for demonstration purposes.



Main evaluation script

src/run_eval.py is the main entry point.

It:

Loads the evaluation questions from data/eval_questions.json.

Instantiates the RagPipeline on the data/corpus/ directory.

For each question:

runs the pipeline (retrieve + answer_stub),

constructs a “gold context” by concatenating the documents listed in must_be_grounded_in,

computes metrics (keyword_coverage, context_overlap, combined score),

records the metrics and the IDs of retrieved documents.

Prints a short summary to stdout.

Saves a detailed JSON report to rag_eval_results.json in the project root.


Example summary line:


q1: score=0.750 (coverage=1.000, overlap=0.500)



Quick start

From the project root:


python src/run_eval.py


This will:

Run the RAG-style pipeline on the predefined evaluation set.

Print per-question metrics.

Write detailed results to:


rag_eval_results.json



You can inspect this file to see, for each question:

the question text,

the generated stub answer,

expected keywords and grounding documents,

computed metrics,

IDs and scores of retrieved documents.



Tests

The project includes a minimal test suite for the metrics and retriever.

Run from the project root:


python -m unittest discover -s tests


This will execute:

tests/test_eval_metrics.py – unit tests for keyword_coverage, context_overlap, and compute_score.

tests/test_retriever.py – basic sanity checks that the retriever can find the appropriate documents for simple queries.



Extending with a real LLM

To turn this demo into a real RAG evaluation harness, you can:

Replace answer_stub(...) in pipeline.py with a call to an LLM API, using:

the question,

the retrieved contexts.

Keep eval_metrics.py and run_eval.py unchanged, or add more advanced metrics.

The current structure is designed so that evaluation logic and data remain the same when you switch from a stub generator to a real model.
