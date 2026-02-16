# RAG Search Engine

This repository implements a Retrieval-Augmented Generation (RAG) search engine step by step, following the progression of the Boot.dev Learn Retrieval Augmented Generation course.

## Goal

Build a complete, testable RAG pipeline:

- Retrieval: fetch relevant context from a document corpus
- Ranking: improve results from basic keyword matching to stronger scoring strategies
- Generation: produce answers using retrieved context (instead of guessing from model memory)
- Iteration: measure quality and improve the system over time

## Why This Project Starts From Scratch

This repo intentionally builds the low-level pieces before using higher-level RAG frameworks:

- understand how text is cleaned, tokenized, indexed, and retrieved
- debug relevance problems with clear visibility into each step
- build intuition for scoring and ranking before adding abstraction
- avoid treating retrieval as a black box when quality drops in production

## Course-Aligned Roadmap

1. Preprocessing  
Normalize and clean raw corpora so they are ready for indexing and downstream retrieval tasks.

2. TF-IDF  
Construct inverted indexes and weighting schemes so keyword search can rank documents effectively.

3. Keyword Search  
Tune keyword retrieval with BM25 refinements and metadata boosts to improve lexical relevance.

4. Semantic Search  
Apply embeddings, similarity metrics, and vector databases to deliver semantic retrieval and RAG responses.

5. Chunking  
Partition documents into context-preserving segments so RAG pipelines can retrieve the right snippets efficiently.

6. Hybrid Search  
Blend lexical and semantic scores into unified retrieval pipelines that boost ranking quality.

7. LLMs  
Leverage large language models to expand queries, correct intent, and orchestrate retrieval workflows.

8. Reranking  
Re-score retrieved candidates with rerankers to surface the most relevant answers.

9. Evaluation  
Measure retrieval precision, recall, and relevance so you can systematically improve RAG performance.

10. Augmented Generation  
Combine retrieved context with LLMs to synthesize coherent, grounded answers for end users.

11. Agentic  
Deploy autonomous agents that iteratively refine queries and navigate complex retrieval workflows.

12. Multimodal  
Extend RAG to images and other modalities with multimodal embeddings and cross-modal retrieval.

## Current Status

- [x] Preprocessing
- [x] TF-IDF (TF, IDF, and TF-IDF inspection commands)
- [x] Keyword Search (baseline lookup + BM25 scoring/search commands)
- [x] Semantic Search (embeddings + cosine-similarity retrieval over movies)
- [ ] Chunking (chapter started: fixed-size chunking utility added)
- [ ] Hybrid Search
- [ ] LLMs
- [ ] Reranking
- [ ] Evaluation
- [ ] Augmented Generation
- [ ] Agentic
- [ ] Multimodal

## Progress So Far

- Built and persisted an inverted index from the movie corpus.
- Added term-frequency storage and TF/IDF/TF-IDF inspection commands.
- Added BM25 components (BM25 IDF, BM25 TF with tunable `k1`/`b`, and ranked BM25 query search).
- Persisted document lengths for BM25 length normalization.
- Kept a simple baseline keyword search flow for comparison against BM25 ranking.
- Added semantic embeddings using `sentence-transformers` (`all-MiniLM-L6-v2`).
- Added embedding cache loading/generation and semantic similarity search commands.
- Started chunking with a fixed-size chunking helper and CLI command.

## Current Output

Right now, this project provides a working local keyword-search baseline over `data/movies.json`:

- text normalization + tokenization + stemming + stop-word filtering
- inverted index creation and persistence
- per-document term frequency tracking
- document length tracking for BM25 normalization
- inverse document frequency (IDF) calculation
- TF-IDF score calculation (`tf * idf`) for a term in a document
- BM25 term score and BM25-ranked query search commands
- semantic embedding generation and caching
- cosine-similarity semantic retrieval
- initial fixed-size chunking workflow

### Generated Artifacts

- `cache/index.pkl`
- `cache/docmap.pkl`
- `cache/term_frequencies.pkl`
- `cache/doc_lengths.pkl`
- `cache/movie_embeddings.npy`

## Requirements

- Python 3.14+
- `uv` (recommended)

## Setup

```bash
uv sync
```

## Commands (Current)

Build the inverted index cache:

```bash
uv run python cli/keyword_search_cli.py build
```

Search by query (basic keyword flow):

```bash
uv run python cli/keyword_search_cli.py search "space adventure"
```

Get term frequency for a term in a document:

```bash
uv run python cli/keyword_search_cli.py tf 42 adventure
```

Get inverse document frequency (IDF) for a term:

```bash
uv run python cli/keyword_search_cli.py idf adventure
```

Get TF-IDF for a term in a document:

```bash
uv run python cli/keyword_search_cli.py tfidf 42 adventure
```

Get BM25 IDF for a term:

```bash
uv run python cli/keyword_search_cli.py bm25idf adventure
```

Get BM25 TF for a term in a document (optional `k1` and `b`):

```bash
uv run python cli/keyword_search_cli.py bm25tf 42 adventure
uv run python cli/keyword_search_cli.py bm25tf 42 adventure 1.5 0.75
```

Search by query using BM25 ranking:

```bash
uv run python cli/keyword_search_cli.py bm25search "space adventure"
```

Verify semantic model loading:

```bash
uv run python cli/semantic_search_cli.py verify
```

Generate embedding for arbitrary text:

```bash
uv run python cli/semantic_search_cli.py embed_text "A story about deep space survival"
```

Generate/check cached movie embeddings:

```bash
uv run python cli/semantic_search_cli.py verify_embeddings
```

Generate embedding for a query:

```bash
uv run python cli/semantic_search_cli.py embedquery "space adventure mission"
```

Run semantic search:

```bash
uv run python cli/semantic_search_cli.py search "space adventure mission" --limit 5
```

Chunk input text (fixed-size chunks):

```bash
uv run python cli/semantic_search_cli.py chunk "Your long text goes here" --chunk-size 200
```

## Data and Cache

- Input data: `data/movies.json`
- Stop words: `data/stopwords.txt`
- Cached files: `cache/index.pkl`, `cache/docmap.pkl`, `cache/term_frequencies.pkl`, `cache/doc_lengths.pkl`, `cache/movie_embeddings.npy`
