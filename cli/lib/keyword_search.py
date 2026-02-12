from lib.search_utils import load_movies, load_stop_words, CACHE_PATH
import string
from nltk.stem import PorterStemmer
from collections import Counter, defaultdict
import os
import pickle

class InvertedIndex:
    def __init__(self):
        self.index = defaultdict(set)
        self.docmap = {}
        self.term_frequencies = defaultdict(Counter)
        self.index_path = CACHE_PATH / 'index.pkl'
        self.docmap_path = CACHE_PATH / 'docmap.pkl'
        self.term_frequencies_path = CACHE_PATH / 'term_frequencies.pkl'

    def _add_document(self, doc_id: int, text: str):
        tokens = tokenize_text(clean_text(text))
        for token in set(tokens):
            self.index[token].add(doc_id)
        self.term_frequencies[doc_id].update(tokens)
    
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

    def load(self):
        with open(self.index_path, 'rb') as f:
            self.index = pickle.load(f)
        with open(self.docmap_path, 'rb') as f:
            self.docmap = pickle.load(f)
        with open(self.term_frequencies_path, 'rb') as f:
            self.term_frequencies = pickle.load(f)

    def get_tf(self,doc_id, term):
        tokens = tokenize_text(term)
        if len(tokens) != 1:
            raise ValueError("Term must be a single token")
        token = tokens[0]
        return self.term_frequencies[doc_id][token]

def get_tf_command(doc_id, term):
    idx = InvertedIndex()
    idx.load()
    print(idx.get_tf(doc_id, term))

def build_command():
    index = InvertedIndex()
    index.build()
    index.save()
    # docs = index.get_documents('merida')
    # print(f"First document for token 'merida' = {docs[0]}")

def clean_text(text: str):
    text = text.lower()
    text  = text.translate(str.maketrans("","", string.punctuation))
    return text


def tokenize_text(text: str):
    text = clean_text(text)
    tokens = [tok for tok in text.split() if tok]
    stop_words = load_stop_words()
    stemmer = PorterStemmer()
    tokens = [token for token in tokens if token not in stop_words]
    tokens = [stemmer.stem(token) for token in tokens ]
    return tokens



# def clean_tokens(tokens: list):
    

def has_matching_token(query_tokens:list[str], movie_tokens:list[str]):
    
    for query_tok in query_tokens:
        for movie_tok in movie_tokens:
            if query_tok in movie_tok:
                return True
    return False



def search_command(query: str,n_results: int = 5):
    movies = load_movies()
    idx = InvertedIndex()
    idx.load()
    
    seen, results =set(), []
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
        
    # for movie in movies:
    #     movie_tokens = tokenize_text(movie['title'])
    #     if has_matching_token(query_tokens,movie_tokens):
    #         results.append(movie)
    #     if len(results) == n_results:
    #         break
    # return results