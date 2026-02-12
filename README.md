# RAG Search Engine

This repository implements a Retrieval-Augmented Generation (RAG) search engine step by step.

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

## Project Roadmap

This repo is structured in progressive stages:

- [x] Preprocessing (normalize + clean text)
- [x] Inverted index + term frequencies
- [ ] TF-IDF / BM25 ranking
- [ ] Semantic search (embeddings + vector similarity)
- [ ] Chunking
- [ ] Hybrid search
- [ ] Reranking
- [ ] Evaluation
- [ ] Augmented generation (grounded answers)

## Current Output

Right now, this project provides a working local keyword-search baseline over `data/movies.json`:

- text normalization + tokenization + stemming + stop-word filtering
- inverted index creation and persistence
- per-document term frequency tracking
- CLI commands for indexing, searching, and TF inspection

### Generated Artifacts

- `cache/index.pkl`
- `cache/docmap.pkl`
- `cache/term_frequencies.pkl`

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

Search by query:

```bash
uv run python cli/keyword_search_cli.py search "space adventure"
```

Get term frequency for a term in a document:

```bash
uv run python cli/keyword_search_cli.py tf 42 adventure
```

## Data and Cache

- Input data: `data/movies.json`
- Stop words: `data/stopwords.txt`
- Cached files: `cache/index.pkl`, `cache/docmap.pkl`, `cache/term_frequencies.pkl`
