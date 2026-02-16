#!/usr/bin/env python3

import argparse
from lib.semantic_search import (embedding_command, verify_model,
                                 verify_embeddings_command, embed_query_text_command,
                                 search_command, chunk_text_command)


def main():
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    subparsers.add_parser("verify", help="Verify that the model loads correctly")
    embedding_parser = subparsers.add_parser("embed_text", help="Generate embedding for input text")
    embedding_parser.add_argument("text", type=str, help="Input text to generate embedding for")
    subparsers.add_parser("verify_embeddings",
                          help="Verify that embeddings are generated and cached correctly")
    query_parser = subparsers.add_parser("embedquery", help="Generate embedding for a query")
    query_parser.add_argument("query", type=str, help="Input query to generate embedding for")
    search_parser = subparsers.add_parser("search", help="Search for similar documents")
    search_parser.add_argument("query", type=str, help="Input query to search for")
    search_parser.add_argument("--limit", type=int, default=5,
                               help="Number of top results to return")
    chunk_parser = subparsers.add_parser("chunk", help="Chunk text into fixed-size pieces")
    chunk_parser.add_argument("text", type=str, help="Input text to chunk")
    chunk_parser.add_argument("--chunk-size", type=int, default=200,
                              help="Chunk size in characters")
    args = parser.parse_args()

    match args.command:

        case "chunk":
            chunk_text_command(args.text, args.chunk_size)
        case "search":
            search_command(args.query, args.limit)
        case "embedquery":
            embed_query_text_command(args.query)
        case "verify_embeddings":
            verify_embeddings_command()
        case "embed_text":
            embedding_command(args.text)
        case "verify":
            verify_model()
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
