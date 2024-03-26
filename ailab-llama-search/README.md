# AI Lab - Llama Search Package

## Overview

The `ailab-llama-search` package facilitates querying our custom index built using LlamaIndex and PostgresSQL.

## Installation

```bash
pip install git+https://github.com/ai-cfia/llamaindex-db.git@main#subdirectory=ailab-llama-search
```

## Usage

```python
from ailab_llama_search import create_index_object, search

# adapt these parameters to your needs
embed_model_params = {
    "azure_endpoint": "azure_openai_endpoint",
    "api_key": "azure_openai_api_key",
    "api_version": "2023-07-01-preview",
    "model": "embed_model_name",
    "deployment_name": "embed_model_deployment_name",
}
vector_store_params = {
    "host": "postgres_host",
    "user": "postgres_user",
    "password": "postgres_password",
    "database": "postgres_db_name",
    "port": "5432",
    "embed_dim": 1536,
}
trans_paths = {
    "id": "node/metadata/id",
    "chunk_id": "node/metadata/chunk_id",
    "url": "node/metadata/url",
    "title": "node/metadata/title",
    "subtitle": "node/metadata/subtitle",
    "tokens_count": "node/metadata/tokens_count",
    "last_updated": "node/metadata/last_updated",
    "score": "node/metadata/score",
    "llama_id": "node/id_",
    "llama_score": "score",
    "content": "node/text",
}

index = create_index_object(embed_model_params, vector_store_params)
search_results = search("your query", index, trans_paths=trans_paths)

for result in search_results:
    print(result)
```

## Exceptions

- `AilabLlamaSearchError`: Triggered if the search query string is empty or
  `None`.

## Functions

- `search`: Executes search queries against a LlamaIndex `VectorStoreIndex`.
- `create_index_object`: Generates a `VectorStoreIndex` object needed to use the
  `search` function.
- `transform`: Modifies search result data based on predefined mappings.
