#!/usr/bin/env python3

import argparse
from lib.keyword_search import  get_tf_command, search_command,build_command, idf_command, tfidf_command

def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")
    search_parser = subparsers.add_parser("build", help="Build the inverted index")
    
    tf_parser = subparsers.add_parser("tf", help="Get term frequency for a document")
    tf_parser.add_argument("doc_id", type=int, help="Document ID")
    tf_parser.add_argument("term", type=str, help="Term to get frequency for")
    
    idf_parser = subparsers.add_parser("idf", help="Get inverse document frequency for a term")
    idf_parser.add_argument("term", type=str, help="Term to get IDF for")
    
    tfidf_parser = subparsers.add_parser("tfidf", help="Get TF-IDF score for a term in a document")
    tfidf_parser.add_argument("doc_id", type=int, help="Document ID")
    tfidf_parser.add_argument("term", type=str, help="Term to get TF-IDF score for")
    
    args = parser.parse_args()

    match args.command:
        case "search":
            # print the search query here
            print(f"Searching for: {args.query}")
            results  = search_command(args.query,5)
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
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()