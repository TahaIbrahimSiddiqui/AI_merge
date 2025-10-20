import numpy as np
from rapidfuzz import process, fuzz

def cdist_scores(queries: list[str], choices: list[str], scorer=fuzz.token_set_ratio) -> np.ndarray:
    # Returns matrix (len(queries), len(choices)) of scores
    if len(queries) == 0 or len(choices) == 0:
        return np.zeros((len(queries), len(choices)), dtype=np.float32)
    M = process.cdist(queries, choices, scorer=scorer)
    # Ensure float32 for memory
    return np.asarray(M, dtype=np.float32)

def best_per_query(M: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # For each row in M, return best index and score
    if M.size == 0:
        return np.array([], dtype=int), np.array([], dtype=np.float32)
    best_idx = np.argmax(M, axis=1)
    best_scores = M[np.arange(M.shape[0]), best_idx]
    return best_idx.astype(int), best_scores

def topk_indices(M: np.ndarray, k: int) -> np.ndarray:
    # Returns indices shape (n_queries, k) of top-k per row (unsorted order), then sorted by score desc
    n, m = M.shape
    if n == 0 or m == 0:
        return np.zeros((n, 0), dtype=int)
    k = min(k, m)
    # argpartition gives top-k unsorted
    part = np.argpartition(-M, kth=k-1, axis=1)[:, :k]
    # sort each row's top-k by score desc
    row_idx = np.arange(n)[:, None]
    row_scores = M[row_idx, part]
    order = np.argsort(-row_scores, axis=1)
    sorted_part = part[row_idx, order]
    return sorted_part