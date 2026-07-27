import hashlib
import json
import os
import pickle
import unittest
from unittest.mock import patch

from requests.exceptions import RequestException

from lm_eval.api.instance import Instance
from lm_eval.models.gguf import GGUFLM, get_result


base_url = "https://matthoffner-ggml-llm-api.hf.space"


def gguf_completion_mock(base_url=None, **kwargs):
    # Generate a hash from the parameters
    hash_kwargs = {"base_url": base_url, **kwargs}
    parameters_hash = hashlib.sha256(
        json.dumps(hash_kwargs, sort_keys=True).encode("utf-8")
    ).hexdigest()

    fname = f"./tests/testdata/gguf_test_{parameters_hash}.pkl"

    if os.path.exists(fname):
        with open(fname, "rb") as fh:
            return pickle.load(fh)  # noqa: S301 - trusted local test fixture
    else:
        print("The file does not exist, attempting to write...")
        if "stop" in kwargs and kwargs["stop"] is not None:
            result = {
                "choices": [
                    {
                        "text": f"generated text until {kwargs['stop']}",
                        "finish_reason": "stop",
                    }
                ]
            }
        else:
            # modern logprobs format as returned by llama.cpp's
            # /v1/completions endpoint since PR #10783:
            # curl -X POST 'http://localhost:8080/v1/completions' \
            #   -H 'Content-Type: application/json' \
            #   -d '{"prompt": "str", "grammar": "root ::= \"ing\"", "logprobs": 10, "temperature": 0.0, "max_tokens": 3}'
            result = {
                "id": "chatcmpl-abc123",
                "object": "text_completion",
                "created": 1785175015,
                "model": "Qwen3.6-35B-A3B-Q8_0",
                "choices": [
                    {
                        "text": "ing",
                        "index": 0,
                        "logprobs": {
                            "content": [
                                {
                                    "id": 287,
                                    "token": "ing",
                                    "bytes": [105, 110, 103],
                                    "logprob": -1.033263319857306,
                                    "top_logprobs": [
                                        {
                                            "id": 287,
                                            "token": "ing",
                                            "bytes": [105, 110, 103],
                                            "logprob": -1.033263319857306,
                                        },
                                        {
                                            "id": 279,
                                            "token": "ed",
                                            "bytes": [101, 100],
                                            "logprob": -2.6530743779017394,
                                        },
                                    ],
                                }
                            ]
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 2,
                    "completion_tokens": 1,
                    "total_tokens": 3,
                },
            }

        try:
            os.makedirs(os.path.dirname(fname), exist_ok=True)
            print("Writing file at", fname)
            with open(fname, "wb") as fh:
                pickle.dump(result, fh)
            print("File written successfully")
        except Exception as e:  # noqa: BLE001 - best-effort fixture write
            print("File writing failed:", e)

        return result


class GGUFLMTest(unittest.TestCase):
    @patch(
        "lm_eval.models.gguf.GGUFLM.gguf_completion", side_effect=gguf_completion_mock
    )
    def test_loglikelihood(self, gguf_completion_mock):
        lm = GGUFLM(base_url, parallel=1)

        # Test loglikelihood
        requests = [
            Instance(
                request_type="loglikelihood",
                doc=args,
                arguments=args,
                idx=i,
            )
            for i, args in enumerate([("str", "ing"), ("str", "ing")])
        ]
        res = lm.loglikelihood(requests)

        # Assert the loglikelihood response is correct
        expected_res = [(-1.033263319857306, True), (-1.033263319857306, True)]
        self.assertEqual(res, expected_res)

    @patch(
        "lm_eval.models.gguf.GGUFLM.gguf_completion", side_effect=gguf_completion_mock
    )
    def test_loglikelihood_empty_continuation(self, gguf_completion_mock):
        lm = GGUFLM(base_url, parallel=1)

        requests = [
            Instance(
                request_type="loglikelihood",
                doc=args,
                arguments=args,
                idx=i,
            )
            for i, args in enumerate([("str", ""), ("str", "ing")])
        ]
        res = lm.loglikelihood(requests)

        # An empty continuation is scored (0.0, True) without a server call
        expected_res = [(0.0, True), (-1.033263319857306, True)]
        self.assertEqual(res, expected_res)

    @patch(
        "lm_eval.models.gguf.GGUFLM.gguf_completion", side_effect=gguf_completion_mock
    )
    def test_generate_until(self, gguf_completion_mock):
        lm = GGUFLM(base_url, parallel=1)

        # Test generate_until
        requests = [
            Instance(
                request_type="generate_until",
                doc={"input": doc},
                arguments=(doc, {"until": stop}),
                idx=i,
            )
            for i, (doc, stop) in enumerate([("input1", "stop1"), ("input2", "stop2")])
        ]

        res = lm.generate_until(requests)

        # Assert the generate_until response is correct
        expected_res = ["generated text until stop1", "generated text until stop2"]
        self.assertEqual(res, expected_res)

    def test_get_result(self):
        def make_content(tokens):
            return [
                {
                    "token": tok,
                    "logprob": lp,
                    "top_logprobs": [
                        {"token": t, "logprob": l} for t, l in top.items()
                    ],
                }
                for tok, lp, top in tokens
            ]

        # greedy: every forced token is the argmax
        logprob, is_greedy = get_result(
            make_content(
                [
                    (" Paris", -0.5, {" Paris": -0.5, " a": -2.0}),
                    (",", -0.7, {",": -0.7, ".": -0.8}),
                ]
            ),
            " Paris,",
        )
        self.assertAlmostEqual(logprob, -1.2)
        self.assertTrue(is_greedy)

        # non-greedy: "," is not the argmax of its position
        logprob, is_greedy = get_result(
            make_content(
                [
                    (" Paris", -0.5, {" Paris": -0.5, " a": -2.0}),
                    (",", -0.7, {".": -0.1, ",": -0.7}),
                ]
            ),
            " Paris,",
        )
        self.assertAlmostEqual(logprob, -1.2)
        self.assertFalse(is_greedy)

        # llama.cpp emits a trailing empty-token entry after the grammar has
        # fully matched; it must be ignored
        logprob, is_greedy = get_result(
            make_content(
                [
                    ("ing", -1.0, {"ing": -1.0}),
                    ("", -15.0, {" renowned": -0.6}),
                ]
            ),
            "ing",
        )
        self.assertEqual(logprob, -1.0)
        self.assertTrue(is_greedy)

        # empty continuation
        self.assertEqual(get_result([], ""), (0.0, True))

        # generated tokens diverging from the continuation raise
        with self.assertRaises(ValueError):
            get_result(make_content([("xyz", -1.0, {"xyz": -1.0})]), "ing")

        # running out of tokens before matching the continuation raises
        with self.assertRaises(ValueError):
            get_result(make_content([("in", -1.0, {"in": -1.0})]), "ing")

    def test_parallel_mapping_preserves_order(self):
        lm = GGUFLM(base_url, parallel=3)
        items = list(range(20))
        res = lm._map_requests(lambda x: x * 2, items, disable_tqdm=True)
        self.assertEqual(res, [x * 2 for x in items])

    def test_detect_total_slots(self):
        lm = GGUFLM(base_url)
        with patch("lm_eval.models.gguf.requests.get") as mock_get:
            mock_get.return_value.json.return_value = {"total_slots": 4}
            self.assertEqual(lm._detect_total_slots(), 4)
            self.assertEqual(lm._resolve_parallel(), 4)

        lm = GGUFLM(base_url)
        with patch(
            "lm_eval.models.gguf.requests.get",
            side_effect=RequestException("no server"),
        ):
            self.assertIsNone(lm._detect_total_slots())
            # falls back to serial requests
            self.assertEqual(lm._resolve_parallel(), 1)

    def test_completions_url_derivation(self):
        for base, expected in [
            ("http://localhost:8080", "http://localhost:8080/v1/completions"),
            ("http://localhost:8080/", "http://localhost:8080/v1/completions"),
            ("http://localhost:8080/v1", "http://localhost:8080/v1/completions"),
            (
                "http://localhost:8080/v1/completions",
                "http://localhost:8080/v1/completions",
            ),
        ]:
            lm = GGUFLM(base, parallel=1)
            self.assertEqual(lm.completions_url, expected)
            self.assertEqual(lm.server_url, "http://localhost:8080")


if __name__ == "__main__":
    unittest.main()
