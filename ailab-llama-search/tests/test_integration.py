import json
import os
import time
import unittest

from ailab_llama_search import create_index_object, search
from dotenv import load_dotenv


class AilabLlamaSearchIntegrationTests(unittest.TestCase):

    def setUp(self):
        load_dotenv()
        self.embed_model_params = json.loads(os.getenv("EMBED_MODEL_PARAMS"))
        self.vector_store_params = json.loads(os.getenv("VECTOR_STORE_PARAMS"))
        self.trans_paths = json.loads(os.getenv("TRANS_PATHS"))
        self.search_params = {"similarity_top_k": 5}
        self.index = create_index_object(
            self.embed_model_params, self.vector_store_params
        )

    def test_search(self):
        query = "steps and considerations of the sampling procedures for food safety"
        start_time = time.time()
        results = search(query, self.index, self.search_params, self.trans_paths)
        end_time = time.time()
        duration = (end_time - start_time) * 1000
        n = self.search_params["similarity_top_k"]
        self.assertLess(duration, 2000)
        self.assertEqual(len(results), n)
        for result in results:
            for key in self.trans_paths.keys():
                self.assertIn(key, result)
