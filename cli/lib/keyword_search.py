from lib.search_utils import load_movies

def search_command(query,n_results):
    movies = load_movies()
    results = []
    for movie in movies:
        if query in movie['title']:
            results.append(movie)
        if len(results) == n_results:
            break
    return results