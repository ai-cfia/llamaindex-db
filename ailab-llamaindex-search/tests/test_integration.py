import json
import os
import unittest

from ailab_llamaindex_search import create_index_object, search
from dotenv import load_dotenv


class AilabLlamaSearchIntegrationTests(unittest.TestCase):

    def setUp(self):
        load_dotenv()
        self.embed_model_params = json.loads(
            os.getenv("LLAMAINDEX_DB_EMBED_MODEL_PARAMS")
        )
        self.vector_store_params = json.loads(
            os.getenv("LLAMAINDEX_DB_VECTOR_STORE_PARAMS")
        )
        self.trans_paths = json.loads(os.getenv("LLAMAINDEX_DB_TRANS_PATHS"))
        self.index = create_index_object(
            self.embed_model_params, self.vector_store_params
        )

    def test_search(self):
        query = "steps and considerations of the sampling procedures for food safety"
        results = search(query, self.index, 10, self.trans_paths)
        for result in results:
            for key in self.trans_paths.keys():
                self.assertIn(key, result)
