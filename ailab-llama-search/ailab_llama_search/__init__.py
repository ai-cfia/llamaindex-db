import logging

import dpath
from llama_index.core import VectorStoreIndex
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
from llama_index.vector_stores.postgres import PGVectorStore


class AilabLlamaSearchError(Exception):
    """Generic Ailab llama search error."""


def transform(node_dict: dict, paths: dict):
    if not paths:
        return node_dict

    return {key: dpath.get(node_dict, path) for key, path in paths.items()}


def search(
    query: str,
    index: VectorStoreIndex,
    search_params: dict = {},
    trans_paths: dict = {},
):
    if not query:
        logging.error("Empty search query received")
        raise AilabLlamaSearchError("search query cannot be empty.")

    retriever = index.as_retriever(**search_params)
    nodes = retriever.retrieve(query)
    return [transform(n.dict(), trans_paths) for n in nodes]


def create_index_object(embed_model_params: dict, vector_store_params: dict):
    embed_model = AzureOpenAIEmbedding(**embed_model_params)
    vector_store = PGVectorStore.from_params(**vector_store_params)
    return VectorStoreIndex.from_vector_store(vector_store, embed_model)
