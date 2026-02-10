import re
import math
import json
import ollama
import platform
import itertools

import multiprocess as mp

from typing import Any, Dict, List, Optional, Callable, Tuple


JSON = Dict[str, Any]
_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)
QueryFn = Callable[[str, str, Optional[Dict[str, Any]], Optional[int]], str]

def _worker(qin: mp.Queue, qout: mp.Queue, default_model: str):
    """
    Receives items like:
      {"job_id": int, "prompt": str, "model": str|None, "format": "json"|dict|None, "options": dict|None}
    Returns:
      {"job_id": int, "response": str}  (or {"job_id": int, "error": "..."} on failure)
    """
    while True:
        item = qin.get()
        if item is None:  # sentinel for clean shutdown
            break

        job_id = item.get("job_id")
        prompt = item.get("prompt", "")
        model = item.get("model") or default_model
        fmt = item.get("format", None)  # can be "json" or a JSON schema dict
        options = item.get("options") or {}

        try:
            resp = ollama.generate(
                model=model,
                prompt=prompt,
                stream=False,
                format=fmt,
                options=options,
            )
            qout.put({"job_id": job_id, "response": resp["response"]})
        except Exception as e:
            qout.put({"job_id": job_id, "error": repr(e)})

# -------- engine --------

class MPBatchOllamaEngine:
    def __init__(self, *, model: str, units: int = 4):
        self.model = model
        self.units = units

        if platform.system() == "Darwin":
            mp.set_start_method("spawn", force=True)

        self._qin: mp.Queue = mp.Queue()
        self._qout: mp.Queue = mp.Queue()
        self._procs: List[mp.Process] = []
        self._job_counter = itertools.count()

        for _ in range(units):
            p = mp.Process(target=_worker, args=(self._qin, self._qout, self.model))
            p.daemon = True
            p.start()
            self._procs.append(p)

    def close(self):
        # ask workers to exit
        for _ in self._procs:
            self._qin.put(None)
        for p in self._procs:
            p.join(timeout=2)

        # hard-stop any stragglers
        for p in self._procs:
            if p.is_alive():
                p.kill()

    def generate_batch_requests(self, requests: List[JSON]) -> List[str]:
        """
        requests: list of dicts with keys prompt/model/format/options
        Returns list[str] of responses in the same order.
        """
        tagged: List[JSON] = []
        for r in requests:
            job_id = next(self._job_counter)
            rr = dict(r)
            rr["job_id"] = job_id
            tagged.append(rr)
            self._qin.put(rr)

        results: Dict[int, str] = {}
        errors: Dict[int, str] = {}

        for _ in tagged:
            out = self._qout.get()
            jid = out.get("job_id")
            if "error" in out:
                errors[jid] = out["error"]
                results[jid] = ""  # placeholder
            else:
                results[jid] = out.get("response", "")

        # keep input order
        ordered = [results[r["job_id"]] for r in tagged]

        # optional: raise if *everything* errored
        if errors and len(errors) == len(tagged):
            raise RuntimeError(f"All batch items failed. Example error: {next(iter(errors.values()))}")

        return ordered

def _extract_json(raw: str) -> JSON:
    """
    Try strict json.loads first. If the model wraps text around JSON, extract the first {...}.
    """
    raw = raw.strip()
    try:
        obj = json.loads(raw)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    m = _JSON_OBJECT_RE.search(raw)
    if not m:
        raise ValueError("No JSON object found in response.")
    obj = json.loads(m.group(0))
    if not isinstance(obj, dict):
        raise ValueError("Extracted JSON is not an object.")
    return obj

def _is_number(x: Any) -> bool:
    return isinstance(x, (int, float)) and not (isinstance(x, float) and (math.isnan(x) or math.isinf(x)))

def _normalize_string(s: str) -> str:
    return " ".join(s.strip().lower().split())

def _value_key(v: Any) -> Tuple[str, Any]:
    """
    Hashable key for voting.
    """
    if isinstance(v, bool):
        return ("bool", v)
    if _is_number(v):
        # round for stable voting (tweak if you need more precision)
        return ("num", round(float(v), 6))
    if isinstance(v, str):
        return ("str", _normalize_string(v))
    return ("other", str(v))

def _validate_candidate(out: JSON, expected: List[str]) -> Optional[JSON]:
    """
    Return cleaned dict with only expected keys if valid; otherwise None.
    """
    if not isinstance(out, dict):
        return None
    cleaned: JSON = {}
    for k in expected:
        if k not in out:
            return None
        v = out[k]
        if isinstance(v, (str, bool)) or _is_number(v):
            cleaned[k] = v
        else:
            return None
    return cleaned

def consensus_merge(candidates: List[JSON], expected: List[str]) -> JSON:
    """
    Field-wise self-consistency: for each key, take the most common value (with normalization).
    Returns a dict over expected keys.
    """
    merged: JSON = {}
    for k in expected:
        votes: Dict[Tuple[str, Any], Tuple[int, Any]] = {}  # key -> (count, representative_value)
        for c in candidates:
            v = c[k]
            kk = _value_key(v)
            if kk in votes:
                votes[kk] = (votes[kk][0] + 1, votes[kk][1])
            else:
                votes[kk] = (1, v)

        # pick most common; ties break by earliest inserted (dict preserves insertion in Py3.7+)
        best_key = max(votes.keys(), key=lambda kk: votes[kk][0])
        merged[k] = votes[best_key][1]
    return merged

def pick_best_candidate(candidates: List[JSON], expected: List[str]) -> JSON:
    """
    Alternative strategy: compute consensus, then return the candidate that matches it the most.
    """
    cons = consensus_merge(candidates, expected)

    def score(c: JSON) -> int:
        s = 0
        for k in expected:
            if _value_key(c[k]) == _value_key(cons[k]):
                s += 1
        return s

    best = max(candidates, key=score)
    return best

def make_batched_query_fn(
    engine: MPBatchOllamaEngine,
    *,
    n: int = 8,
    strategy: str = "consensus",  # "consensus" or "pick_best"
) -> QueryFn:
    """
    Returns a query_fn(prompt, model, schema, seed)->str that:
      - runs n parallel samples
      - parses/validates them
      - self-consistency merges outputs
      - returns JSON string
    """

    def _query(prompt: str, model: str, schema: Optional[JSON], seed: Optional[int]) -> str:
        # expected keys come from schema if provided
        expected: List[str] = []
        if schema and isinstance(schema.get("properties"), dict):
            expected = list(schema["properties"].keys())

        requests: List[JSON] = []
        for i in range(n):
            requests.append(
                {
                    "prompt": prompt,
                    "model": model,
                    "format": schema,
                }
            )

        raws = engine.generate_batch_requests(requests)

        valid: List[JSON] = []
        for raw in raws:
            try:
                obj = _extract_json(raw)
                if expected:
                    cleaned = _validate_candidate(obj, expected)
                    if cleaned is not None:
                        valid.append(cleaned)
                else:
                    # if no expected keys, accept any dict
                    if isinstance(obj, dict):
                        valid.append(obj)
            except Exception:
                continue

        if not valid:
            # Helpful debugging info: show a couple raw samples
            preview = "\n---\n".join(r[:400] for r in raws[:2])
            raise ValueError(f"No valid JSON candidates out of {n} samples.\nPreview:\n{preview}")

        if expected:
            if strategy == "pick_best":
                final = pick_best_candidate(valid, expected)
            else:
                final = consensus_merge(valid, expected)
        else:
            final = valid[0]

        return json.dumps(final, ensure_ascii=False)

    return _query

