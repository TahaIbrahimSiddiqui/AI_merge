import json
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional, Tuple
from tqdm.auto import tqdm

try:
    import google.generativeai as genai
except Exception:
    genai = None

DEFAULT_PROMPT = """You are a strict matching engine.
Task: Given a query string and a candidate list, pick the best single candidate by semantic similarity, spelling closeness, naming variations, ocr errors, phonentic errors and time period context .
Return a JSON object strictly:
{"best_index": int, "best_candidate": string, "confidence": int}
Rules:
- best_index is 0-based in the given candidates; -1 if none is plausible.
- confidence 0-100.
- Use only provided candidates.

Query: "{query}"

Candidates:
{candidates}
"""

def _extract_json(text: str) -> dict:
    try:
        return json.loads(text)
    except Exception:
        m = re.search(r"\{.*\}", text, flags=re.S)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                pass
    return {"best_index": -1, "best_candidate": None, "confidence": 0}

def gemini_select_best(
    query: str,
    candidates: List[str],
    api_key: Optional[str] = None,
    model_name: str = "gemini-2.5-pro",
    temperature: float = 0.2,
) -> Tuple[int, Optional[str], int]:
    if genai is None:
        raise RuntimeError("google-generativeai not installed. pip install google-generativeai")
    api_key = api_key or os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("Set GEMINI_API_KEY or pass api_key.")
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name)
    prompt = DEFAULT_PROMPT.format(
        query=query,
        candidates="\n".join(f"{i}. {c}" for i, c in enumerate(candidates))
    )
    resp = model.generate_content(prompt, generation_config={"temperature": temperature})
    text = getattr(resp, "text", "") or ""
    data = _extract_json(text)
    best_index = data.get("best_index", -1)
    conf = data.get("confidence", 0)
    if not isinstance(best_index, int):
        best_index = -1
    if not isinstance(conf, int):
        try:
            conf = int(conf)
        except Exception:
            conf = 0
    best_candidate = candidates[best_index] if 0 <= best_index < len(candidates) else None
    return best_index, best_candidate, conf

def gemini_batch_select(
    queries: List[str],
    candidate_lists: List[List[str]],
    api_key: Optional[str] = None,
    model_name: str = "gemini-2.5-pro",
    temperature: float = 0.2,
    max_workers: int = 8,
    show_progress: bool = True,
) -> List[Tuple[int, Optional[str], int]]:
    # Parallel batched calls with progress bar
    results = [None] * len(queries)
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {
            ex.submit(
                gemini_select_best, queries[i], candidate_lists[i],
                api_key, model_name, temperature
            ): i
            for i in range(len(queries))
        }
        it = as_completed(futures)
        if show_progress:
            it = tqdm(it, total=len(futures), desc="Gemini matching", unit="row")
        for fut in it:
            i = futures[fut]
            try:
                results[i] = fut.result()
            except Exception:
                results[i] = (-1, None, 0)
    return results