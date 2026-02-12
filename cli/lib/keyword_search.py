from lib.search_utils import load_movies, load_stop_words
import string
from nltk.stem import PorterStemmer

def clean_text(text: str):
    text = text.lower()
    text  = text.translate(str.maketrans("","", string.punctuation))
    return text


def tokenize_text(text: str):
    text = clean_text(text)
    tokens = [tok for tok in text.split() if tok]
    return tokens


def clean_tokens(tokens: list):
    stop_words = load_stop_words()
    stemmer = PorterStemmer()
    stop_words = [stemmer.stem(token) for token in stop_words ]
    tokens = [stemmer.stem(token) for token in tokens ]
    tokens = [token for token in tokens if token not in stop_words]

    return tokens


def has_matching_token(query_tokens:list[str], movie_tokens:list[str]):
    
    for query_tok in clean_tokens(query_tokens):
        for movie_tok in clean_tokens(movie_tokens):
            if query_tok in movie_tok:
                return True
    return False


def search_command(query: str,n_results: int):
    movies = load_movies()
    results = []
    query_tokens = tokenize_text(query)
    for movie in movies:
        movie_tokens = tokenize_text(movie['title'])
        if has_matching_token(query_tokens,movie_tokens):
            results.append(movie)
        if len(results) == n_results:
            break
    return results