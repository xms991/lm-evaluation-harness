import logging
import time

import requests
from requests.exceptions import RequestException
from tqdm import tqdm

from lm_eval.api.model import LM
from lm_eval.api.registry import register_model


logger = logging.getLogger(__name__)


def _escape_grammar_literal(text: str) -> str:
    """Escape a string for use as a double-quoted literal in a GBNF grammar."""
    return (
        text.replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("\n", "\\n")
        .replace("\r", "\\r")
        .replace("\t", "\\t")
    )


def _continuation_grammar(continuation: str) -> str:
    """Build a GBNF grammar that forces generation of exactly `continuation`."""
    return f'root ::= "{_escape_grammar_literal(continuation)}"'


def get_result(content_logprobs, continuation):
    """Parse the modern (chat-completions style) logprobs format returned by
    llama.cpp's `/v1/completions` endpoint:

        "logprobs": {
            "content": [
                {
                    "token": " Paris",
                    "logprob": -0.55,
                    "top_logprobs": [{"token": " Paris", "logprob": -0.55}, ...],
                    ...
                },
                ...
            ]
        }

    Because llama.cpp never returns logprobs for prompt tokens (and silently
    ignores `echo`), the continuation is grammar-forced instead. llama.cpp
    reports pre-sampling probabilities for grammar-constrained tokens, so the
    per-token `logprob` values are the true teacher-forced logprobabilities.

    Returns (logprob_sum, is_greedy) for `continuation`.
    """
    logprob_sum = 0.0
    is_greedy = True
    matched = ""
    for item in content_logprobs:
        token = item.get("token") or ""
        if token == "":
            # llama.cpp can emit a trailing empty-token entry after the
            # grammar has fully matched; skip it.
            continue
        matched += token
        if not continuation.startswith(matched):
            raise ValueError(
                f"Generated tokens {matched!r} diverge from continuation {continuation!r}"
            )
        logprob_sum += item["logprob"]
        top_logprobs = item.get("top_logprobs") or []
        if is_greedy and top_logprobs:
            top_item = max(top_logprobs, key=lambda x: x["logprob"])
            if top_item["token"] != token:
                is_greedy = False
        if matched == continuation:
            return logprob_sum, is_greedy
    if continuation == "":
        return logprob_sum, is_greedy
    raise ValueError(
        f"Ran out of logprobs before matching continuation {continuation!r} "
        f"(only matched {matched!r})"
    )


@register_model("gguf", "ggml")
class GGUFLM(LM):
    """Evaluate GGUF models served by a llama.cpp server (llama-server)
    through its OpenAI-compatible `/v1/completions` endpoint.

    Requires a llama.cpp version that returns logprobs in the modern
    OpenAI format (`logprobs.content`, see llama.cpp PR #10783; any release
    from December 2024 onward). The deprecated legacy format
    (`token_logprobs`/`text_offset`) is not supported.

    Pass `base_url` pointing at the server, e.g.
    `--model gguf --model_args base_url=http://127.0.0.1:8080`.
    When the server runs in router mode (multiple models), also pass
    `model=<name-or-alias>` so requests are routed correctly.
    """

    def __init__(
        self,
        base_url=None,
        model=None,
        max_length=2048,
        timeout=300,
        logprobs=10,
        temperature=0.0,
        **kwargs,
    ):
        super().__init__()
        assert base_url, "must pass `base_url` to use GGUF LM!"
        base_url = base_url.rstrip("/")
        if base_url.endswith("/completions"):
            self.completions_url = base_url
        elif base_url.endswith("/v1"):
            self.completions_url = base_url + "/completions"
        else:
            self.completions_url = base_url + "/v1/completions"
        self.model = model
        self.logprobs = logprobs
        self.temperature = temperature
        self.max_length = max_length
        self.timeout = timeout

    def gguf_completion(
        self,
        context,
        continuation=None,
        stop=None,
        max_tokens=None,
        retries=3,
        delay=5,
        **kwargs,
    ):
        for _ in range(retries):
            try:
                request = {
                    "prompt": context,
                    "temperature": self.temperature,
                }
                if self.model is not None:
                    request["model"] = self.model
                if continuation is not None:
                    # llama.cpp ignores `echo` and never returns logprobs for
                    # prompt tokens, so the continuation cannot be scored by
                    # echoing prompt+continuation. Instead, force the exact
                    # continuation with a GBNF grammar; llama.cpp reports
                    # pre-sampling probabilities for grammar-constrained
                    # tokens, which are exactly the teacher-forced
                    # logprobabilities we need.
                    request.update(
                        {
                            "grammar": _continuation_grammar(continuation),
                            "logprobs": self.logprobs,
                            # upper bound: every token is at least one byte
                            "max_tokens": max(len(continuation.encode("utf-8")), 1),
                        }
                    )
                elif max_tokens is not None:
                    request["max_tokens"] = max_tokens
                if stop is not None:
                    request["stop"] = stop
                response = requests.post(
                    self.completions_url, json=request, timeout=self.timeout
                )
                response.raise_for_status()
                return response.json()
            except RequestException as e:
                logger.error(f"RequestException: {e}")
                time.sleep(delay)  # wait before retrying
        raise RuntimeError(f"Failed to get a valid response after {retries} retries.")

    def loglikelihood(self, requests, disable_tqdm: bool = False):
        if not requests:
            return []
        res = []
        for context, continuation in tqdm(
            [req.args for req in requests], disable=disable_tqdm
        ):
            if continuation == "":
                res.append((0.0, True))
                continue
            response = self.gguf_completion(context=context, continuation=continuation)
            if response and "choices" in response and response["choices"]:
                choice = response["choices"][0]
                logprobs = choice.get("logprobs")
                if logprobs and "content" in logprobs and logprobs["content"]:
                    try:
                        res.append(get_result(logprobs["content"], continuation))
                    except ValueError as e:
                        logger.error(
                            f"Could not parse logprobs for continuation "
                            f"{continuation!r}: {e}"
                        )
                        res.append((float("-inf"), False))
                else:
                    logger.error(
                        "Invalid logprobs data. Expected 'logprobs' to contain a "
                        "'content' list (the modern OpenAI logprobs format used by "
                        "llama.cpp since PR #10783). Is the server a recent llama.cpp "
                        f"llama-server? Response: {response}"
                    )
                    res.append((float("-inf"), False))
            else:
                logger.error(
                    f"Invalid response for loglikelihood. Response: {response}"
                )
                res.append((float("-inf"), False))
        return res

    def generate_until(self, requests, disable_tqdm: bool = False):
        if not requests:
            return []

        res = []
        for request in tqdm([req.args for req in requests], disable=disable_tqdm):
            inp = request[0]
            request_args = request[1]
            until = request_args.get("until", ["</s>"])
            max_gen_toks = request_args.get("max_gen_toks", None)
            response = self.gguf_completion(
                context=inp, stop=until, max_tokens=max_gen_toks
            )
            if response and "choices" in response and response["choices"]:
                choice = response["choices"][0]
                if "text" in choice:
                    generated_text = choice["text"].strip()
                    res.append(generated_text)
                else:
                    logger.error(
                        f"Invalid response for greedy_until. Response: {response}"
                    )
                    res.append(None)  # Add default value in case of error
            else:
                logger.error(f"Invalid response for greedy_until. Response: {response}")
                res.append(None)  # Add default value in case of error
        return res

    def loglikelihood_rolling(self, requests, disable_tqdm: bool = False):
        raise NotImplementedError(
            "loglikelihood_rolling not yet supported for GGUF models"
        )
