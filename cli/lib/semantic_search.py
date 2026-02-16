from sentence_transformers import SentenceTransformer
import numpy as np
import os
from lib.search_utils import CACHE_PATH, load_movies

# this only to hide the loading messages from transformers and huggingface hub
from huggingface_hub.utils import disable_progress_bars
from transformers.utils import logging as tf_logging
disable_progress_bars()
tf_logging.set_verbosity_error()
tf_logging.disable_progress_bar()


class SemanticSearch:

    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.embeddings = None
        self.documents = None
        self.document_map = {}

    def build_embeddings(self, documents):
        self.documents = documents
        for document in documents:
            self.document_map[document['id']] = document
        docs = [f'{doc["title"]} {doc["description"]}' for doc in documents]
        self.embeddings = self.model.encode(docs, show_progress_bar=True)
        # save it into CACHE_PATH/embeddings.npy
        np.save(f"{CACHE_PATH}/movie_embeddings.npy", self.embeddings)
        return self.embeddings

    def load_or_create_embeddings(self, documents):
        self.documents = documents
        for document in documents:
            self.document_map[document['id']] = document
        # check if the embeddings file exists
        embeddings_file = f"{CACHE_PATH}/movie_embeddings.npy"
        if os.path.exists(embeddings_file):
            self.embeddings = np.load(embeddings_file)
            if len(documents) == len(self.embeddings):
                return self.embeddings
        else:
            self.embeddings = self.build_embeddings(documents)
            return self.embeddings

    def search(self, query, limit):
        if self.embeddings is None:
            raise ValueError("No embeddings loaded. Call `load_or_create_embeddings` first.")

        query_embedding = self.generate_embedding(query)
        similarities = [cosine_similarity(query_embedding, doc_embedding)
                        for doc_embedding in self.embeddings]
        scored_docs = [(score, self.document_map[doc['id']])
                       for score, doc in zip(similarities, self.documents)]
        sorted_docs = sorted(scored_docs, key=lambda x: x[0], reverse=True)
        top_results = sorted_docs[:limit]
        return [{"score": score, "title": doc["title"], "description": doc["description"]}
                for score, doc in top_results]

    def generate_embedding(self, text):
        if text is None or text.strip() == "":
            raise ValueError("Input text cannot be empty.")
        embedding = self.model.encode(text)
        return embedding


def fixed_sized_chunking(text, chunk_size):
    text = text.strip().split(" ")

    chunks = []
    for i in range(0, len(text), chunk_size):
        if i + chunk_size > len(text):
            chunk_size = len(text) - i
        chunk = " ".join(text[i:i + chunk_size])
        chunks.append(chunk)
    return chunks


def chunk_text_command(text, chunk_size):
    chunks = fixed_sized_chunking(text, chunk_size)
    print(f"Chunking {len(text)} characters")
    for idx, chunk in enumerate(chunks, start=1):
        print(f"{idx}. {chunk}")


def search_command(query, limit):
    ss = SemanticSearch()
    movies = load_movies()
    ss.load_or_create_embeddings(movies)
    embed_query = ss.generate_embedding(query)
    top_results = ss.search(query, limit)
    for idx, result in enumerate(top_results, start=1):
        print(f"{idx}. {result['title']} (Score: {result['score']:.4f})")
        print(f"   Description: {result['description']}\n")


def embed_query_text_command(query: str):
    ss = SemanticSearch()
    query = query.strip()
    embedding = ss.generate_embedding(query)
    print(f"Query: {query}")
    print(f"First 5 dimensions: {embedding[:5]}")
    print(f"Dimensions: {embedding.shape}")


def verify_embeddings_command():
    ss = SemanticSearch()
    movies = load_movies()
    embeddings = ss.load_or_create_embeddings(movies)
    print(f"Number of docs:   {len(movies)}")
    print(f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions")


def embedding_command(text):
    ss = SemanticSearch()
    embedding = ss.generate_embedding(text)
    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")


def verify_model():
    ss = SemanticSearch()
    print(f"Model loaded: {ss.model}")
    print(f"Max sequence length: {ss.model.max_seq_length}")


def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)
