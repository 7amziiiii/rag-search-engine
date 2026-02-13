#!/usr/bin/env python3

import argparse
from lib.keyword_search import bm25_idf_command, bm25_search_command, get_tf_command, search_command, build_command, idf_command, tfidf_command, bm25_tf_command
from lib.search_utils import BM25_B, BM25_K1


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(
        dest="command", help="Available commands")

    search_parser = subparsers.add_parser(
        "search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")
    search_parser = subparsers.add_parser(
        "build", help="Build the inverted index")

    tf_parser = subparsers.add_parser(
        "tf", help="Get term frequency for a document")
    tf_parser.add_argument("doc_id", type=int, help="Document ID")
    tf_parser.add_argument("term", type=str, help="Term to get frequency for")

    idf_parser = subparsers.add_parser(
        "idf", help="Get inverse document frequency for a term")
    idf_parser.add_argument("term", type=str, help="Term to get IDF for")

    tfidf_parser = subparsers.add_parser(
        "tfidf", help="Get TF-IDF score for a term in a document")
    tfidf_parser.add_argument("doc_id", type=int, help="Document ID")
    tfidf_parser.add_argument(
        "term", type=str, help="Term to get TF-IDF score for")

    bm25_idf_parser = subparsers.add_parser(
        "bm25idf", help="Get BM25 IDF score for a given term")
    bm25_idf_parser.add_argument(
        "term", type=str, help="Term to get BM25 IDF score for")

    bm25_tf_parser = subparsers.add_parser(
        "bm25tf", help="Get BM25 TF score for a given document ID and term")
    bm25_tf_parser.add_argument("doc_id", type=int, help="Document ID")
    bm25_tf_parser.add_argument(
        "term", type=str, help="Term to get BM25 TF score for")
    bm25_tf_parser.add_argument(
        "k1", type=float, nargs='?', default=BM25_K1, help="Tunable BM25 K1 parameter")
    bm25_tf_parser.add_argument("b", type=float, nargs='?',
                                default=BM25_B, help="Tunable BM25 b parameter")
    bm25search_parser = subparsers.add_parser(
        "bm25search", help="Search movies using full BM25 scoring")
    bm25search_parser.add_argument("query", type=str, help="Search query")
    args = parser.parse_args()

    match args.command:
        case "search":
            # print the search query here
            print(f"Searching for: {args.query}")
            results = search_command(args.query, 5)
            for i, result in enumerate(results):
                print(f"{i+1}. {result['title']}")
        case "tf":
            get_tf_command(args.doc_id, args.term)
        case "build":
            build_command()
        case "idf":
            idf_command(args.term)
        case "tfidf":
            tfidf_command(args.doc_id, args.term)
        case "bm25idf":
            bm25_idf_command(args.term)
        case "bm25tf":
            bm25_tf_command(args.doc_id, args.term, args.k1, args.b)

        case "bm25search":
            bm25_search_command(args.query, 5)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
