from lib.search_utils import load_movies
import string
def clean_text(text: str):
    text = text.lower()
    text  = text.translate(str.maketrans("","", string.punctuation))
    return text

def tokenize_text(text: str):
    text = clean_text(text)
    tokens = [tok for tok in text.split() if tok]
    return tokens


def has_matching_token(query_tokens:list[str], movie_tokens:list[str]):
    for query_tok in query_tokens:
        for movie_tok in movie_tokens:
            if query_tok in movie_tok:
                return True
    return False


def search_command(query: str,n_results: int):
    movies = load_movies()
    results = []
    query = tokenize_text(query)
    for movie in movies:
        if has_matching_token(query,tokenize_text(movie['title'])):
            results.append(movie)
        if len(results) == n_results:
            break
    return results