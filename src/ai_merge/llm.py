import json
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional, Tuple
from tqdm.auto import tqdm
import logging
try:
    import google.generativeai as genai
except Exception:
    genai = None

DEFAULT_PROMPT = """You are a highly advanced name matching engine. Your task is to find the best match for a given query from a list of candidates. You can do many to many matches, one to many matches, and many to one matches based on semantic, lexical, phonetic, and contextual similarities.
You must evaluate candidates based on a combination of the following factors:
Semantic Similarity: The meaning and context of the words.
Lexical & Phonetic Similarity: Consider spelling closeness (typos), as well as how the names sound, even when spelled differently across languages (e.g., "Marseille" vs. "Marsella").
Naming Variations: Intelligently recognize and expand abbreviations, acronyms, and short forms into their full names (e.g., "JFK" for "John F. Kennedy", "St. Petersburg" for "Saint Petersburg").
Contextual Knowledge: Apply your understanding of cultural and historical context, such as places that have changed names over time (e.g., "Bombay" and "Mumbai" refer to the same city).
Output Rules:
Your output must be a single, raw JSON object and nothing else.
The JSON object must strictly adhere to the following structure:
code
JSON
{
  "best_index": <int>,
  "best_candidate": "<string>",
  "confidence": <int>,
  "explanation": "<string>"
}
Field Definitions:
best_index: The 0-based index of the best matching candidate from the provided list. Use -1 if no plausible match is found.
best_candidate: The exact string of the best candidate from the list. If no match is found, this should be null.
confidence: An integer from 0 to 100 representing your confidence in the match.
Constraints:
You must STRICTLY only choose a candidate from the provided list.
Prioritize high-confidence matches. If no candidate is a strong match, it is better to return an index of -1 than to make a low-confidence guess.
Return with given id for both sides of the dataset

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
    try:
        conf = int(conf)
    except Exception:
        conf = 0
    if 0 <= best_index < len(candidates):
       best_candidate = candidates[best_index]
    else:
       if best_index != -1:  # Only log if not intentional "no match"
        logging.warning(f"LLM returned invalid index {best_index} for {len(candidates)} candidates")
        best_candidate = None
    return best_index, best_candidate, conf

def gemini_batch_select(
    queries: List[str],
    candidate_lists: List[List[str]],
    api_key: Optional[str] = None,
    model_name: str = "gemini-2.5-pro",
    temperature: float = 0.2,
    max_workers: int = 8,
    show_progress: bool = True,
):
    results = [None] * len(queries)
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {
            ex.submit(gemini_select_best, queries[i], candidate_lists[i], api_key, model_name, temperature): i
            for i in range(len(queries))
        }
        iterator = as_completed(futures)
        if show_progress:
            iterator = tqdm(iterator, total=len(futures), desc="Gemini matching", unit="row")
        for fut in iterator:
            i = futures[fut]
            try:
                results[i] = fut.result()
            except Exception as e:
                if show_progress:
                    tqdm.write(f"Error processing query {i}: {str(e)}")
                results[i] = (-1, None, 0)
    return results