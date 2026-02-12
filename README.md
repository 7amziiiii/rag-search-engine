# RAG Search Engine

This repository implements a **Retrieval-Augmented Generation (RAG)** search engine step by step, following the progression of the Boot.dev “Learn Retrieval Augmented Generation” course.

## Goal

Build a complete, testable RAG pipeline:

- **Retrieval**: fetch relevant context from a document corpus
- **Ranking**: improve results from basic keyword matching to stronger scoring strategies
- **Generation**: produce answers using retrieved context (instead of guessing from model memory)
- **Iteration**: measure quality and improve the system over time

## Why This Project Starts From Scratch

This repo intentionally builds the low-level pieces before using higher-level RAG frameworks:

- understand how text is cleaned, tokenized, indexed, and retrieved
- debug relevance problems with clear visibility into each step
- build intuition for scoring and ranking before adding abstraction
- avoid treating retrieval as a black box when quality drops in production

## Course-Aligned Roadmap

This repo is structured to mirror the course’s major topics (high level): :contentReference[oaicite:0]{index=0}

- [x] **Preprocessing** (normalize + clean text)
- [ ] **TF-IDF** (weighted retrieval over an inverted index)
- [ ] **Keyword Search** (BM25 + metadata boosts)
- [ ] **Semantic Search** (embeddings + vector similarity)
- [ ] **Chunking** (retrieve the right snippets, not entire docs)
- [ ] **Hybrid Search** (blend lexical + semantic signals)
- [ ] **LLMs** (query expansion / intent correction / orchestration)
- [ ] **Reranking** (re-score candidates to improve top results)
- [ ] **Evaluation** (measure precision/recall/relevance)
- [ ] **Augmented Generation** (grounded answers using retrieved context)
- [ ] **Agentic Search**
- [ ] **Multimodal Retrieval**

The course ultimately builds toward a full RAG pipeline in Python, including an LLM step using the Gemini API. :contentReference[oaicite:1]{index=1}

## Current Output (What This Repo Produces Today)

Right now, this project provides a working **local keyword-search baseline** over `data/movies.json`:

- text normalization + tokenization + stemming + stop-word filtering
- inverted index creation and persistence
- per-document term frequency tracking
- CLI commands for indexing, searching, and TF inspection

### Generated artifacts

- `cache/index.pkl`
- `cache/docmap.pkl`
- `cache/term_frequencies.pkl`

> Cache files are stored locally and should be ignored by git.

## Requirements

- Python 3.11+ (3.12+ recommended)
- `uv` (recommended)

## Setup

```bash
uv sync
