from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from .preprocess import add_normalized_columns, build_key_series
from .llm import gemini_batch_select

@dataclass
class ExactStage:
    on: List[Tuple[str, str]]  # [(left_col, right_col), ...]

@dataclass
class LLMBackend:
    model: str = "gemini-2.5-pro"
    api_key: Optional[str] = None
    temperature: float = 0.2
    max_workers: int = 8
    show_progress: bool = True
    min_confidence: int = 70
    max_candidates_per_group: int = 50
    strict_candidate_cap: bool = False

def _ensure_output_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c in ["match_type", "match_stage", "match_confidence", "matched_right_index", "matched_value"]:
        if c not in df.columns:
            df[c] = pd.NA
    return df

def _normalize_for_cols(
    left: pd.DataFrame, right: pd.DataFrame,
    left_cols: List[str], right_cols: List[str],
    left_corr: Optional[Dict[str, Dict[str, str]]],
    right_corr: Optional[Dict[str, Dict[str, str]]],
):
    left_n = add_normalized_columns(left, left_cols, corrections=left_corr)
    right_n = add_normalized_columns(right, right_cols, corrections=right_corr)
    return left_n, right_n

def _apply_exact_stage(
    left: pd.DataFrame,
    right: pd.DataFrame,
    stage: ExactStage,
    right_cols_to_return: List[str],
    left_corr: Optional[Dict[str, Dict[str, str]]],
    right_corr: Optional[Dict[str, Dict[str, str]]],
    stage_idx: int,
    show_progress: bool,
) -> pd.DataFrame:
    left = left.copy()
    
    # Prepare
    left_cols = [lc for lc, _ in stage.on]
    right_cols = [rc for _, rc in stage.on]
    
    left_n, right_n = _normalize_for_cols(left, right, left_cols, right_cols, left_corr, right_corr)
    
    # Build norm col maps
    left_norm_map = {lc: f"{lc}__norm" for lc in left_cols}
    right_norm_map = {rc: f"{rc}__norm" for rc in right_cols}
    
    # Prepare frames for merge
    lmerge = left_n.assign(_left_index=left_n.index)
    rmerge = right_n.assign(_right_index=right_n.index)
    
    # Join on normalized columns
    l_on = [left_norm_map[lc] for lc in left_cols]
    r_on = [right_norm_map[rc] for rc in right_cols]
    
    # Fixed: Remove walrus operator and duplicate columns
    merged = lmerge.merge(
        rmerge[[*r_on, "_right_index", *right_cols_to_return]],
        left_on=l_on, 
        right_on=r_on,
        how="left",
        suffixes=("", "__rdupe"),
    )
    
    # Keep first right match per left row (deterministic)
    merged.sort_values(by=["_left_index", "_right_index"], inplace=True)
    first_match = merged.drop_duplicates(subset="_left_index", keep="first").set_index("_left_index")
    
    # Write outputs for rows that matched
    matched = first_match["_right_index"].notna()
    if matched.any():
        left.loc[matched.index, "matched_right_index"] = first_match.loc[matched.index, "_right_index"].astype("Int64")
        left.loc[matched.index, "match_type"] = "exact"
        left.loc[matched.index, "match_stage"] = int(stage_idx)
        left.loc[matched.index, "match_confidence"] = pd.NA
        
        # Pull selected right columns
        for col in right_cols_to_return:
            if col in right.columns:
                # Fixed: Initialize column if doesn't exist
                if col not in left.columns:
                    left[col] = pd.NA
                left.loc[matched.index, col] = right.loc[first_match.loc[matched.index, "_right_index"], col].values
    
    return left

def merge_two_step(
    left_df: pd.DataFrame,
    right_df: pd.DataFrame,
    exact_stages: List[ExactStage],
    llm_match: Tuple[str, str],  # (left_col, right_col) for the LLM comparison
    right_cols_to_return: List[str],
    llm_backend: LLMBackend,
    left_corrections: Optional[Dict[str, Dict[str, str]]] = None,
    right_corrections: Optional[Dict[str, Dict[str, str]]] = None,
    llm_candidate_keys: Optional[List[Tuple[str, str]]] = None,  # default: use last exact stage keys
    keep_unmatched: bool = True,
):
    left = _ensure_output_columns(left_df)
    right = right_df.copy()
    
    # 1) Exact stages in order
    stages_iter = tqdm(exact_stages, desc="Exact stages", unit="stage") if llm_backend.show_progress else exact_stages
    for s_idx, stage in enumerate(stages_iter, start=1):
        unmatched_mask = left["matched_right_index"].isna()
        if not unmatched_mask.any():
            break
        left_unmatched = left.loc[unmatched_mask].copy()
        updated = _apply_exact_stage(
            left_unmatched, right, stage, right_cols_to_return,
            left_corr=left_corrections, right_corr=right_corrections,
            stage_idx=s_idx, show_progress=llm_backend.show_progress
        )
        left.loc[updated.index, updated.columns] = updated
    
    # 2) One LLM pass for remaining unmatched
    rem_mask = left["matched_right_index"].isna()
    if rem_mask.any():
        lcol, rcol = llm_match
        
        # Determine candidate scoping keys
        cand_pairs = llm_candidate_keys if llm_candidate_keys is not None else (exact_stages[-1].on if exact_stages else [])
        l_keys = [lc for lc, _ in cand_pairs]
        r_keys = [rc for _, rc in cand_pairs]
        
        # Normalize needed columns (keys + match cols)
        left_needed = list(dict.fromkeys([*l_keys, lcol]))
        right_needed = list(dict.fromkeys([*r_keys, rcol]))
        
        left_n, right_n = _normalize_for_cols(left.loc[rem_mask], right, left_needed, right_needed, left_corrections, right_corrections)
        
        # Build key groups
        l_norm_cols = [f"{c}__norm" for c in l_keys]
        r_norm_cols = [f"{c}__norm" for c in r_keys]
        
        l_keys_series = build_key_series(left_n, l_norm_cols) if l_norm_cols else pd.Series([""], index=left_n.index, dtype="string")
        r_keys_series = build_key_series(right_n, r_norm_cols) if r_norm_cols else pd.Series([""], index=right_n.index, dtype="string")
        
        left_groups = left_n.groupby(l_keys_series, sort=False).indices
        right_groups = right_n.groupby(r_keys_series, sort=False).indices
        
        shared_keys = [k for k in left_groups.keys() if k in right_groups]
        
        group_iter = tqdm(shared_keys, desc="LLM groups", unit="group") if llm_backend.show_progress else shared_keys
        
        for k in group_iter:
            lidx = left_groups[k]
            ridx = right_groups[k]
            
            lrows = left_n.loc[lidx]
            rrows = right_n.loc[ridx]
            
            if lrows.empty or rrows.empty:
                continue
            
            # Build candidate list for this group (dedup, cap)
            cand_map = {}
            for ri, val in zip(rrows.index, rrows[rcol].astype("string").fillna("").tolist()):
                if val not in cand_map:
                    cand_map[val] = ri
            
            candidates = list(cand_map.keys())
            
            if len(candidates) > llm_backend.max_candidates_per_group:
                if llm_backend.strict_candidate_cap:
                    raise ValueError(f"LLM group '{k}' has {len(candidates)} candidates (> cap {llm_backend.max_candidates_per_group}).")
                candidates = sorted(candidates)[: llm_backend.max_candidates_per_group]
                cand_map = {c: cand_map[c] for c in candidates}
            
            if not candidates:
                continue
            
            # Queries for this group
            queries = left.loc[lrows.index, lcol].astype("string").fillna("").tolist()
            cand_lists = [candidates] * len(queries)
            
            results = gemini_batch_select(
                queries, cand_lists,
                api_key=llm_backend.api_key,
                model_name=llm_backend.model,
                temperature=llm_backend.temperature,
                max_workers=llm_backend.max_workers,
                show_progress=False,
            )
            
            for (best_rel, best_str, conf), left_row_index in zip(results, lrows.index):
                if best_rel is not None and best_rel >= 0 and best_rel < len(candidates) and conf >= llm_backend.min_confidence:
                    chosen_cand = candidates[best_rel]
                    right_idx = cand_map.get(chosen_cand)
                    if right_idx is not None:
                        left.at[left_row_index, "matched_right_index"] = int(right_idx)
                        left.at[left_row_index, "matched_value"] = chosen_cand
                        left.at[left_row_index, "match_confidence"] = int(conf)
                        left.at[left_row_index, "match_type"] = "llm"
                        left.at[left_row_index, "match_stage"] = "llm"
                        
                        # bring right cols now for this row
                        for col in right_cols_to_return:
                            if col in right.columns:
                                left.at[left_row_index, col] = right.at[right_idx, col]
    
    if not keep_unmatched:
        left = left[left["matched_right_index"].notna()].copy()
    
    # Stats
    matched_mask = left["matched_right_index"].notna()
    
    # Fixed: Handle empty dataframe case for highest_confidence
    llm_matches = left["match_type"] == "llm"
    highest_conf = 0
    if llm_matches.any():
        conf_vals = pd.to_numeric(left.loc[llm_matches, "match_confidence"], errors="coerce")
        if not conf_vals.isna().all():
            highest_conf = int(conf_vals.max())
    
    stats = {
        "total_left_rows": int(len(left_df)),
        "exact_matches": int(((left["match_type"] == "exact") & matched_mask).sum()),
        "llm_matches": int((llm_matches & matched_mask).sum()),
        "match_count": int(matched_mask.sum()),
        "matched_ratio": float(matched_mask.mean()),
        "highest_confidence": highest_conf,
        "stages": len(exact_stages),
    }
    
    return left, stats