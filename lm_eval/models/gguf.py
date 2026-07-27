import logging
import time
from concurrent.futures import ThreadPoolExecutor

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

    Requests are issued concurrently with a thread pool. By default the
    degree of parallelism is auto-detected from the server's `/props`
    endpoint (`total_slots`, i.e. llama-server's `--parallel` setting);
    override it with `parallel=<N>`. Note that llama-server only processes
    as many requests simultaneously as it has slots, so values above the
    server's slot count mostly add queueing on the server side.
    """

    def __init__(
        self,
        base_url=None,
        model=None,
        max_length=2048,
        timeout=300,
        logprobs=10,
        temperature=0.0,
        parallel=None,
        **kwargs,
    ):
        super().__init__()
        assert base_url, "must pass `base_url` to use GGUF LM!"
        base_url = base_url.rstrip("/")
        # derive the server root so /props can be queried for auto-detection
        server_url = base_url
        for suffix in ("/v1/completions", "/completions", "/v1"):
            if server_url.endswith(suffix):
                server_url = server_url[: -len(suffix)]
                break
        self.server_url = server_url
        self.completions_url = server_url + "/v1/completions"
        self.model = model
        self.logprobs = logprobs
        self.temperature = temperature
        self.max_length = max_length
        self.timeout = timeout
        self.parallel = parallel
        self._resolved_parallel = None

    def _detect_total_slots(self):
        """Query the server's /props endpoint for its slot count
        (llama-server's `--parallel` setting). Returns None if unavailable.
        """
        try:
            params = {"model": self.model} if self.model is not None else None
            response = requests.get(
                f"{self.server_url}/props", params=params, timeout=10
            )
            response.raise_for_status()
            total_slots = response.json().get("total_slots")
            if isinstance(total_slots, int) and total_slots > 0:
                return total_slots
        except (RequestException, ValueError) as e:
            logger.debug(f"Could not query /props for slot count: {e}")
        return None

    def _resolve_parallel(self):
        if self._resolved_parallel is None:
            if self.parallel is not None:
                self._resolved_parallel = max(1, int(self.parallel))
            else:
                total_slots = self._detect_total_slots()
                self._resolved_parallel = total_slots or 1
                if total_slots:
                    logger.info(
                        f"Auto-detected {total_slots} llama.cpp server slots; "
                        f"issuing up to {total_slots} concurrent requests. "
                        "Override with `parallel=<N>`."
                    )
        return self._resolved_parallel

    def _map_requests(self, fn, items, disable_tqdm):
        """Apply fn to each item, preserving order, using a thread pool when
        parallelism is enabled.
        """
        parallel = self._resolve_parallel()
        if parallel <= 1 or len(items) <= 1:
            return [fn(item) for item in tqdm(items, disable=disable_tqdm)]
        with ThreadPoolExecutor(max_workers=parallel) as executor:
            return list(
                tqdm(
                    executor.map(fn, items),
                    total=len(items),
                    disable=disable_tqdm,
                )
            )

    def gguf_completion(
        self,
        context,
        continuation=None,
        stop=None,
        max_tokens=None,
        id_slot=None,
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
                if id_slot is not None:
                    # pin the request to a specific server slot so that
                    # prompts sharing a prefix hit the same per-slot KV cache
                    request["id_slot"] = id_slot
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

    def _loglikelihood_one(self, item):
        args, id_slot = item
        context, continuation = args
        if continuation == "":
            return (0.0, True)
        response = self.gguf_completion(
            context=context, continuation=continuation, id_slot=id_slot
        )
        if response and "choices" in response and response["choices"]:
            choice = response["choices"][0]
            logprobs = choice.get("logprobs")
            if logprobs and "content" in logprobs and logprobs["content"]:
                try:
                    return get_result(logprobs["content"], continuation)
                except ValueError as e:
                    logger.error(
                        f"Could not parse logprobs for continuation "
                        f"{continuation!r}: {e}"
                    )
                    return (float("-inf"), False)
            else:
                logger.error(
                    "Invalid logprobs data. Expected 'logprobs' to contain a "
                    "'content' list (the modern OpenAI logprobs format used by "
                    "llama.cpp since PR #10783). Is the server a recent llama.cpp "
                    f"llama-server? Response: {response}"
                )
                return (float("-inf"), False)
        else:
            logger.error(f"Invalid response for loglikelihood. Response: {response}")
            return (float("-inf"), False)

    @staticmethod
    def _assign_slots_by_context(args_list, parallel):
        """Assign a server slot id to each (context, continuation) pair.

        Consecutive requests sharing the same context (e.g. the candidate
        continuations of one multiple-choice question) are pinned to the same
        slot, so all but the first reuse the slot's cached prompt prefix.
        Groups are round-robined across slots to keep them busy in parallel.
        """
        slots = []
        group_idx = -1
        prev_context = None
        for context, _ in args_list:
            if context != prev_context:
                group_idx += 1
                prev_context = context
            slots.append(group_idx % parallel)
        return slots

    def loglikelihood(self, requests, disable_tqdm: bool = False):
        if not requests:
            return []
        parallel = self._resolve_parallel()
        args_list = [req.args for req in requests]
        slots = self._assign_slots_by_context(args_list, parallel)
        return self._map_requests(
            self._loglikelihood_one,
            list(zip(args_list, slots, strict=True)),
            disable_tqdm,
        )

    def _generate_one(self, args):
        inp, request_args = args
        until = request_args.get("until", ["</s>"])
        max_gen_toks = request_args.get("max_gen_toks", None)
        # no id_slot pinning here: generation lengths vary widely, so the
        # server's dynamic idle-slot assignment load-balances better than a
        # static assignment (measured ~30% slower with pinning on gsm8k)
        response = self.gguf_completion(
            context=inp, stop=until, max_tokens=max_gen_toks
        )
        if response and "choices" in response and response["choices"]:
            choice = response["choices"][0]
            if "text" in choice:
                return choice["text"].strip()
            else:
                logger.error(f"Invalid response for greedy_until. Response: {response}")
                return None  # Add default value in case of error
        else:
            logger.error(f"Invalid response for greedy_until. Response: {response}")
            return None  # Add default value in case of error

    def generate_until(self, requests, disable_tqdm: bool = False):
        if not requests:
            return []

        return self._map_requests(
            self._generate_one,
            [req.args for req in requests],
            disable_tqdm,
        )

    def loglikelihood_rolling(self, requests, disable_tqdm: bool = False):
        raise NotImplementedError(
            "loglikelihood_rolling not yet supported for GGUF models"
        )
