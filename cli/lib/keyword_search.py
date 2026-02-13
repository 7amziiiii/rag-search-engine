from lib.search_utils import load_movies, load_stop_words, CACHE_PATH, BM25_K1, BM25_B
import string
from nltk.stem import PorterStemmer
from collections import Counter, defaultdict
import os
import pickle
import math


class InvertedIndex:
    def __init__(self):
        self.index = defaultdict(set)  # token -> set of doc_ids
        self.docmap = {}  # doc_id -> movie dict
        self.doc_lengths = {}

        # doc_id -> Count of token frequencies
        self.term_frequencies = defaultdict(Counter)

        self.index_path = CACHE_PATH / 'index.pkl'
        self.docmap_path = CACHE_PATH / 'docmap.pkl'
        self.term_frequencies_path = CACHE_PATH / 'term_frequencies.pkl'
        self.doc_lengths_path = CACHE_PATH / "doc_lengths.pkl"

    def _add_document(self, doc_id: int, text: str):
        tokens = tokenize_text(clean_text(text))
        for token in set(tokens):
            self.index[token].add(doc_id)
        self.term_frequencies[doc_id].update(tokens)
        self.doc_lengths[doc_id] = len(tokens)

    def __get_avg_doc_length(self) -> float:
        n_docs = len(self.docmap)
        if n_docs == 0:
            return 0.0
        sum = 0
        for length in self.doc_lengths.values():
            sum += length
        return sum / n_docs

    def get_documents(self, term: str):
        return sorted(list(self.index[term]))

    def build(self):
        movies = load_movies()
        for movie in movies:
            doc_id = movie['id']
            text = f"{movie['title']} {movie['description']}"
            self._add_document(doc_id, text)
            self.docmap[doc_id] = movie

    def save(self):
        os.makedirs(CACHE_PATH, exist_ok=True)
        with open(self.index_path, 'wb') as f:
            pickle.dump(self.index, f)

        with open(self.docmap_path, 'wb') as f:
            pickle.dump(self.docmap, f)

        with open(self.term_frequencies_path, 'wb') as f:
            pickle.dump(self.term_frequencies, f)

        with open(self.doc_lengths_path, 'wb') as f:
            pickle.dump(self.doc_lengths, f)

    def load(self):
        with open(self.index_path, 'rb') as f:
            self.index = pickle.load(f)

        with open(self.docmap_path, 'rb') as f:
            self.docmap = pickle.load(f)

        with open(self.term_frequencies_path, 'rb') as f:
            self.term_frequencies = pickle.load(f)

        with open(self.doc_lengths_path, 'rb') as f:
            self.doc_lengths = pickle.load(f)

    def get_tf(self, doc_id, term):
        tokens = tokenize_text(term)
        if len(tokens) != 1:
            raise ValueError("Term must be a single token")
        token = tokens[0]
        return self.term_frequencies[doc_id][token]

    def get_idf(self, term):
        tokens = tokenize_text(term)
        if len(tokens) != 1:
            raise ValueError("Term must be a single token")
        token = tokens[0]
        total_docs_count = len(self.docmap)
        term_match_doc_count = len(self.index[token])
        return math.log((total_docs_count + 1) / (term_match_doc_count + 1))

    def get_bm25_idf(self, term: str) -> float:
        tokens = tokenize_text(term)
        if len(tokens) != 1:
            raise ValueError("Term must be a single token")
        token = tokens[0]
        total_docs_count = len(self.docmap)
        term_match_doc_count = len(self.index[token])
        return math.log((total_docs_count - term_match_doc_count + 0.5) / (term_match_doc_count + 0.5) + 1)

    def get_tf_idf(self, doc_id, term):
        tf = self.get_tf(doc_id, term)
        idf = self.get_idf(term)
        return tf * idf

    def get_bm25_tf(self, doc_id, term, k1=BM25_K1, b=BM25_B):
        tf = self.get_tf(doc_id, term)
        avg_len = self.__get_avg_doc_length()
        length_norm = 1 - b + b * (self.doc_lengths[doc_id] / avg_len)
        return (tf * (k1 + 1)) / (tf + k1 * length_norm)

    def bm25(self, doc_id, term):
        bm25_idf = self.get_bm25_idf(term)
        bm25_tf = self.get_bm25_tf(doc_id, term)
        return bm25_tf * bm25_idf

    def bm25_search(self, query, limit):
        tokens = tokenize_text(query)
        docs_scores = defaultdict(int)
        for token in tokens:
            for doc_id in self.index[token]:
                score = self.bm25(doc_id, token)
                docs_scores[doc_id] += score
        sorted_docs = sorted(docs_scores.items(), key=lambda x: x[1], reverse=True)
        stored_docs = sorted_docs[:limit]
  
        for i, (doc_id, score) in enumerate(stored_docs):
            print(f"{i+1}. ({doc_id}) {self.docmap[doc_id]['title']} - Score: {score:.2f}")

     
def bm25_search_command(query: str, n_results: int = 5):
    idx = InvertedIndex()
    idx.load()
    idx.bm25_search(query, n_results)

def bm25_tf_command(doc_id, term, k1, b):
    idx = InvertedIndex()
    idx.load()
    bm25_tf = idx.get_bm25_tf(doc_id, term, k1, b)
    print(f"BM25 TF score of '{term}' in document '{doc_id}': {bm25_tf:.2f}")


def bm25_idf_command(term):
    idx = InvertedIndex()
    idx.load()
    idf = idx.get_bm25_idf(term)
    print(f"BM25 IDF score of '{term}': {idf:.2f}")


def tfidf_command(doc_id, term):
    idx = InvertedIndex()
    idx.load()
    tfidf = idx.get_tf_idf(doc_id, term)
    print(f"TF-IDF score of '{term}' in document '{doc_id}': {tfidf:.2f}")


def idf_command(term):
    idx = InvertedIndex()
    idx.load()
    idf = idx.get_idf(term)
    print(f"Inverse document frequency of '{term}': {idf:.2f}")


def get_tf_command(doc_id, term):
    idx = InvertedIndex()
    idx.load()
    print(idx.get_tf(doc_id, term))


def build_command():
    index = InvertedIndex()
    index.build()
    index.save()


def search_command(query: str, n_results: int = 5):
    movies = load_movies()
    idx = InvertedIndex()
    idx.load()

    seen, results = set(), []
    query_tokens = tokenize_text(query)

    for token in query_tokens:
        doc_ids = idx.get_documents(token)
        for doc_id in doc_ids:
            if doc_id not in seen:
                seen.add(doc_id)
                results.append(idx.docmap[doc_id])
            if len(results) == n_results:
                break
    return results


def clean_text(text: str):
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text


def tokenize_text(text: str):
    text = clean_text(text)
    tokens = [tok for tok in text.split() if tok]
    stop_words = load_stop_words()
    stemmer = PorterStemmer()
    tokens = [token for token in tokens if token not in stop_words]
    tokens = [stemmer.stem(token) for token in tokens]
    return tokens


def has_matching_token(query_tokens: list[str], movie_tokens: list[str]):

    for query_tok in query_tokens:
        for movie_tok in movie_tokens:
            if query_tok in movie_tok:
                return True
    return False
