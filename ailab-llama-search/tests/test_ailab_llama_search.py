import unittest
from unittest.mock import MagicMock, patch

from ailab_llama_search import (
    AilabLlamaSearchError,
    VectorStoreIndex,
    create_index_object,
    search,
    transform,
)


class TestAilabLlamaTransform(unittest.TestCase):
    def test_transform(self):
        source = {
            "id": "123",
            "nested": {"key": "value", "list": [1, 2, 3]},
            "list": ["a", "b", "c"],
        }
        paths = {
            "new_id": "/id",
            "nested_value": "/nested/key",
            "first_list_item": "/list/0",
        }
        expected = {
            "new_id": "123",
            "nested_value": "value",
            "first_list_item": "a",
        }
        self.assertEqual(transform(source, paths), expected)

    def test_transform_with_empty_or_none_path_map(self):
        source_dict = {"id": "123", "nested": {"key": "value"}}
        self.assertEqual(transform(source_dict, {}), source_dict)
        self.assertEqual(transform(source_dict, None), source_dict)

    def test_transform_error(self):
        source_dict = {"id": "123"}
        invalid_path_map = {"invalid_key": "/nonexistent/path"}
        with self.assertRaises((KeyError, ValueError)):
            transform(source_dict, invalid_path_map)


class TestAilabLlamaSearch(unittest.TestCase):
    def setUp(self):
        self.mock_index = MagicMock(spec=VectorStoreIndex)
        self.mock_retriever = MagicMock()
        self.mock_retriever.retrieve.return_value = [MagicMock(dict=MagicMock(return_value={'id': 1, 'name': 'Test Node'}))]
        self.mock_index.as_retriever.return_value = self.mock_retriever
    
    def test_search_with_empty_query_error(self):
        with self.assertRaises(AilabLlamaSearchError):
            search("", self.mock_index)

    @patch('ailab_llama_search.transform')
    def test_search_calls_transform_on_results(self, mock_transform):
        mock_transform.return_value = {'id': 1, 'name': 'Transformed Node'}
        results = search("test query", self.mock_index)
        self.assertTrue(mock_transform.called)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0], {'id': 1, 'name': 'Transformed Node'})

    @patch('ailab_llama_search.AzureOpenAIEmbedding')
    @patch('ailab_llama_search.PGVectorStore.from_params')
    @patch('ailab_llama_search.VectorStoreIndex.from_vector_store')
    def test_create_index_object_initializes_correctly(self, mock_from_vector_store, mock_from_params, mock_azure_openai_embedding):
        mock_embed_model = MagicMock()
        mock_azure_openai_embedding.return_value = mock_embed_model
        mock_vector_store = MagicMock()
        mock_from_params.return_value = mock_vector_store
        mock_index_object = MagicMock()
        mock_from_vector_store.return_value = mock_index_object
        embed_model_params = {'param1': 'value1'}
        vector_store_params = {'param2': 'value2'}
        result = create_index_object(embed_model_params, vector_store_params)
        mock_azure_openai_embedding.assert_called_once_with(**embed_model_params)
        mock_from_params.assert_called_once_with(**vector_store_params)
        mock_from_vector_store.assert_called_once_with(mock_vector_store, mock_embed_model)
        self.assertEqual(result, mock_index_object)
