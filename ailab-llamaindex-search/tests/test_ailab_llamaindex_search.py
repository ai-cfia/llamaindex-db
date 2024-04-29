import unittest
from unittest.mock import MagicMock, call, patch

from ailab_llamaindex_search import (
    AilabLlamaIndexSearchError,
    VectorStoreIndex,
    create_index_object,
    search,
    select_highest_scored_nodes_by_url,
    transform,
)
from llama_index.core.schema import NodeWithScore, TextNode


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
        self.mock_index.as_retriever.return_value = self.mock_retriever

    def test_search_with_empty_query_error(self):
        with self.assertRaises(AilabLlamaIndexSearchError):
            search("", self.mock_index)

    @patch("ailab_llamaindex_search.transform")
    @patch("ailab_llamaindex_search.select_highest_scored_nodes_by_url")
    def test_search_calls_the_right_functions(self, mock_select, mock_transform):
        d1 = {"id_": "1", "metadata": {"url": "https://example.com"}}
        d2 = {"id_": "2", "metadata": {"url": "https://example.com"}}
        node1 = NodeWithScore(node=TextNode.from_dict(d1), score=0.8)
        node2 = NodeWithScore(node=TextNode.from_dict(d2), score=0.9)
        nodes = [node1, node2]
        selected_nodes = [node2]
        transformed_nodes = node2.dict()
        self.mock_retriever.retrieve.return_value = nodes
        mock_select.return_value = selected_nodes
        mock_transform.side_effect = lambda node_dict, _: node_dict

        results = search("test query", self.mock_index)
        mock_select.assert_called_once_with(nodes)
        calls = [call(node.dict(), {}) for node in selected_nodes]
        mock_transform.assert_has_calls(calls, any_order=True)
        self.assertTrue(results[0] == transformed_nodes)

    @patch("ailab_llamaindex_search.select_highest_scored_nodes_by_url")
    @patch("ailab_llamaindex_search.transform")
    def test_retriever_similarity_top_k_parameter(self, mock_transform, mock_select):
        self.mock_index.as_retriever = MagicMock()
        similarity_top_k = 10
        search("valid query", self.mock_index, similarity_top_k=similarity_top_k)
        self.mock_index.as_retriever.assert_called_once_with(
            similarity_top_k=similarity_top_k * 2
        )

    @patch("ailab_llamaindex_search.AzureOpenAIEmbedding")
    @patch("ailab_llamaindex_search.PGVectorStore.from_params")
    @patch("ailab_llamaindex_search.VectorStoreIndex.from_vector_store")
    def test_create_index_object_initializes_correctly(
        self, mock_from_vector_store, mock_from_params, mock_azure_openai_embedding
    ):
        mock_embed_model = MagicMock()
        mock_azure_openai_embedding.return_value = mock_embed_model
        mock_vector_store = MagicMock()
        mock_from_params.return_value = mock_vector_store
        mock_index_object = MagicMock()
        mock_from_vector_store.return_value = mock_index_object
        embed_model_params = {"param1": "value1"}
        vector_store_params = {"param2": "value2"}
        result = create_index_object(embed_model_params, vector_store_params)
        mock_azure_openai_embedding.assert_called_once_with(**embed_model_params)
        mock_from_params.assert_called_once_with(**vector_store_params)
        mock_from_vector_store.assert_called_once_with(
            mock_vector_store, mock_embed_model
        )
        self.assertEqual(result, mock_index_object)


class TestSelectHighestScoredNodesByURL(unittest.TestCase):

    def test_empty_input(self):
        self.assertEqual(select_highest_scored_nodes_by_url([]), [])

    def test_single_node(self):
        node_data = {"id_": "1", "metadata": {"url": "https://example.com"}}
        node = NodeWithScore(node=TextNode.from_dict(node_data), score=1.0)
        self.assertEqual(select_highest_scored_nodes_by_url([node]), [node])

    def test_multiple_nodes_one_url(self):
        node_data1 = {"id_": "1", "metadata": {"url": "https://example.com"}}
        node_data2 = {"id_": "2", "metadata": {"url": "https://example.com"}}
        node1 = NodeWithScore(node=TextNode.from_dict(node_data1), score=1.0)
        node2 = NodeWithScore(node=TextNode.from_dict(node_data2), score=2.0)
        self.assertEqual(select_highest_scored_nodes_by_url([node1, node2]), [node2])

    def test_multiple_nodes_multiple_urls(self):
        node_data1 = {"id_": "1", "metadata": {"url": "https://example.com"}}
        node_data2 = {"id_": "2", "metadata": {"url": "https://example.com"}}
        node_data3 = {"id_": "3", "metadata": {"url": "https://example2.com"}}
        node1 = NodeWithScore(node=TextNode.from_dict(node_data1), score=1.0)
        node2 = NodeWithScore(node=TextNode.from_dict(node_data2), score=2.0)
        node3 = NodeWithScore(node=TextNode.from_dict(node_data3), score=3.0)
        result = select_highest_scored_nodes_by_url([node1, node2, node3])
        self.assertIn(node2, result)
        self.assertIn(node3, result)
        self.assertEqual(len(result), 2)

    def test_nodes_with_same_score(self):
        node_data1 = {"id_": "1", "metadata": {"url": "https://example.com"}}
        node_data2 = {"id_": "2", "metadata": {"url": "https://example.com"}}
        node1 = NodeWithScore(node=TextNode.from_dict(node_data1), score=1.0)
        node2 = NodeWithScore(node=TextNode.from_dict(node_data2), score=1.0)
        result = select_highest_scored_nodes_by_url([node1, node2])
        self.assertIn(node1, result)
        self.assertEqual(len(result), 1)
