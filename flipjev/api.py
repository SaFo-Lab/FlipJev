"""Client for the OpenRouter Decisions API that serves Jev.

A request carries the model name, the untrusted content (`state`) and a map of named typed questions. The response
holds one answer per question: a probability for a Noul question, a label with probabilities for a Choice question,
and probabilities with an expected level for a Score question.
"""

import json
import math
import os
import time
import urllib.error
import urllib.request

ENDPOINT = "https://openrouter.ai/api/alpha/decisions"
MODEL = "typesafe/jev-1.13"
API_KEY_VARIABLE = "OPENROUTER_API_KEY"
AUTH_SCHEME = "Bearer"   # the OpenRouter API authenticates with an "Authorization: <scheme> <key>" header
RETRY_STATUSES = (429, 500, 502, 503, 504)


def api_key():
    """The API key from the environment; stop with a message when it is missing."""
    key = os.environ.get(API_KEY_VARIABLE)
    if not key:
        raise SystemExit(f"{API_KEY_VARIABLE} is not set.")
    return key


def assemble_request(state, question_map, model=MODEL):
    return {"model": model, "state": state, "questions": question_map}


def redact(text, key):
    return str(text).replace(key, "[REDACTED]") if key else str(text)


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        # Never forward the credentials to another destination.
        return None


def post(payload, key, timeout=45, retries=2):
    """Send one request; retry rate limits, server errors and transport errors, and never raise."""
    started = time.monotonic()
    attempts = []
    opener = urllib.request.build_opener(_NoRedirect())
    for attempt in range(retries + 1):
        request = urllib.request.Request(
            ENDPOINT, data=json.dumps(payload, ensure_ascii=False).encode(),
            headers={"Authorization": AUTH_SCHEME + " " + key, "Content-Type": "application/json"},
        )
        try:
            with opener.open(request, timeout=timeout) as response:
                data = json.loads(redact(response.read().decode(), key))
            if "error" in data:
                return {"status": "api_error", "response": data, "attempts": attempts,
                        "latency_s": time.monotonic() - started}
            return {"status": "ok", "response": data, "attempts": attempts,
                    "latency_s": time.monotonic() - started}
        except urllib.error.HTTPError as exc:
            detail = redact(exc.read().decode(errors="replace")[:4000], key)
            attempts.append({"http_status": exc.code, "detail": detail})
            if exc.code not in RETRY_STATUSES or attempt == retries:
                return {"status": "http_error", "http_status": exc.code,
                        "attempts": attempts, "latency_s": time.monotonic() - started}
            retry_after = exc.headers.get("Retry-After", "")
            delay = min(float(retry_after), 15) if retry_after.isdigit() else 2 ** attempt
            time.sleep(delay)
        except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
            attempts.append({"error": redact(f"{type(exc).__name__}: {exc}", key)})
            if attempt == retries:
                return {"status": "transport_error", "attempts": attempts,
                        "latency_s": time.monotonic() - started}
            time.sleep(2 ** attempt)


def probability(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("Probability is not a number")
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("Probability is outside [0,1]")
    return float(value)


def parse_answer(answer, primitive):
    """A Noul answer or a yes/no Choice answer -> {"prediction", "p_yes", "threshold_tie"}; raises ValueError."""
    if not isinstance(answer, dict) or answer.get("type") != primitive:
        raise ValueError("Missing or mismatched answer type")
    if primitive == "noul":
        p_yes = probability(answer["noul"])
        return {"prediction": "yes" if p_yes >= 0.5 else "no", "p_yes": p_yes,
                "threshold_tie": p_yes == 0.5}
    prediction = answer["choice"]
    if prediction not in ("yes", "no"):
        raise ValueError("Unexpected choice label")
    p_yes = probability(answer["probabilities"]["yes"])
    p_no = probability(answer["probabilities"]["no"])
    if abs(p_yes + p_no - 1) > 0.025:
        raise ValueError("Choice probabilities do not approximately sum to 1")
    if (prediction == "yes" and p_yes < p_no) or (prediction == "no" and p_no < p_yes):
        raise ValueError("Choice contradicts the most likely label")
    return {"prediction": prediction, "p_yes": p_yes, "threshold_tie": p_yes == p_no}
