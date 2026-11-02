import os
import sys
import types
import unittest
from unittest.mock import patch

from openevo_webshop.inference import GatewayBackend


class GatewayTeacherBackendTests(unittest.TestCase):
    def test_gateway_backend_records_usage_and_auth(self):
        calls = []

        class Response:
            def raise_for_status(self):
                return None

            def json(self):
                return {
                    "choices": [{"message": {"content": "<action>click[x]</action>"}}],
                    "usage": {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18},
                }

        def post(url, **kwargs):
            calls.append((url, kwargs))
            return Response()

        with patch.dict(sys.modules, {"httpx": types.SimpleNamespace(post=post)}), patch.dict(
            os.environ, {"OPENAI_API_KEY": "test-key"}
        ):
            backend = GatewayBackend(
                "gpt-test",
                "https://gateway.example/v1",
                device="cuda",
            )

            self.assertEqual(
                backend.generate("prompt", max_new=32, temperature=0.0, generation_seed=9),
                "<action>click[x]</action>",
            )
            self.assertEqual(calls[0][0], "https://gateway.example/v1/chat/completions")
            self.assertEqual(calls[0][1]["headers"], {"Authorization": "Bearer test-key"})
            self.assertEqual(calls[0][1]["json"]["seed"], 9)
            self.assertEqual(
                backend.usage_totals,
                {"requests": 1, "prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18},
            )


    def test_gateway_backend_fails_closed_without_endpoint(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "OPENAI_BASE_URL not set"):
                GatewayBackend("gpt-test").generate("prompt")


if __name__ == "__main__":
    unittest.main()
