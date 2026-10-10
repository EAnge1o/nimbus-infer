import json
import threading
import unittest
import urllib.parse
import urllib.request

from nimbus_infer.engine import Engine
from nimbus_infer.server import create_server


class TestServer(unittest.TestCase):
    """Boot the real HTTP server on an ephemeral port and hit its endpoints."""

    @classmethod
    def setUpClass(cls):
        cls.engine = Engine.demo("tiny", seed=0)
        cls.httpd = create_server(cls.engine, host="127.0.0.1", port=0)
        cls.port = cls.httpd.server_address[1]
        cls.thread = threading.Thread(target=cls.httpd.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()

    def _get(self, path: str):
        url = f"http://127.0.0.1:{self.port}{path}"
        with urllib.request.urlopen(url, timeout=30) as response:
            return response.status, response.read()

    def test_index_serves_html(self):
        status, body = self._get("/")
        self.assertEqual(status, 200)
        self.assertIn(b"nimbus-infer", body)
        self.assertIn(b"EventSource", body)

    def test_info_endpoint(self):
        _, body = self._get("/api/info")
        data = json.loads(body)
        self.assertEqual(data["parameters"], self.engine.n_params)
        self.assertGreater(data["kv_cache_bytes"], 0)

    def test_attention_endpoint_returns_square_matrix(self):
        prompt = urllib.parse.quote("the engine caches keys")
        _, body = self._get(f"/api/attention?prompt={prompt}")
        data = json.loads(body)
        n = len(data["tokens"])
        self.assertGreater(n, 0)
        self.assertEqual(len(data["matrix"]), n)
        self.assertTrue(all(len(row) == n for row in data["matrix"]))

    def test_stream_endpoint_emits_tokens_then_done(self):
        prompt = urllib.parse.quote("the engine")
        _, body = self._get(f"/api/stream?prompt={prompt}&max_new_tokens=3&temperature=0")
        text = body.decode("utf-8")
        self.assertIn('"type": "token"', text)
        self.assertIn('"type": "done"', text)

    def test_stream_rejects_empty_prompt(self):
        with self.assertRaises(urllib.error.HTTPError) as ctx:
            self._get("/api/stream?prompt=%20")
        self.assertEqual(ctx.exception.code, 400)


if __name__ == "__main__":
    unittest.main()