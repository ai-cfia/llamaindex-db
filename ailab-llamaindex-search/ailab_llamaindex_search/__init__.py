import logging

import dpath
from llama_index.core import VectorStoreIndex
from llama_index.core.schema import NodeWithScore
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
from llama_index.vector_stores.postgres import PGVectorStore


class AilabLlamaIndexSearchError(Exception):
    """Generic Ailab LlamaIndex search error."""


def select_highest_scored_nodes_by_url(nodes: list[NodeWithScore]):
    best_nodes: dict[str, NodeWithScore] = {}
    for node in nodes:
        url: str = node.metadata["url"]
        if url not in best_nodes or best_nodes[url].score < node.score:
            best_nodes[url] = node
    return list(best_nodes.values())


def transform(node_dict: dict, paths: dict):
    if not paths:
        return node_dict

    return {key: dpath.get(node_dict, path) for key, path in paths.items()}


def search(
    query: str,
    index: VectorStoreIndex,
    similarity_top_k: int = 10,
    trans_paths: dict = {},
):
    if not query:
        logging.error("Empty search query received")
        raise AilabLlamaIndexSearchError("search query cannot be empty.")
    retriever = index.as_retriever(similarity_top_k=similarity_top_k * 2)
    nodes = retriever.retrieve(query)
    best_nodes = select_highest_scored_nodes_by_url(nodes)
    return [transform(node.dict(), trans_paths) for node in best_nodes]


def create_index_object(embed_model_params: dict, vector_store_params: dict):
    embed_model = AzureOpenAIEmbedding(**embed_model_params)
    vector_store = PGVectorStore.from_params(**vector_store_params)
    return VectorStoreIndex.from_vector_store(vector_store, embed_model)
