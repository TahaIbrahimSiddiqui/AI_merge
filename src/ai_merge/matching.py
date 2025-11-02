from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from .preprocess import add_normalized_columns, build_key_series, ensure_unique_id, check_id_uniqueness
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
    
    # Merge
    merged = lmerge.merge(
        rmerge[[*r_on, "_right_index", *right_cols_to_return]],
        left_on=l_on, 
        right_on=r_on,
        how="left",
        suffixes=("", "__rdupe"),
        indicator=True
    )
    
    # Only keep rows that actually matched
    matched_rows = merged[merged["_merge"] == "both"].copy()
    
    if matched_rows.empty:
        return left  # No matches in this stage
    
    # For many-to-one matches, keep first right match per left row
    matched_rows.sort_values(by=["_left_index", "_right_index"], inplace=True)
    first_match = matched_rows.drop_duplicates(subset="_left_index", keep="first").set_index("_left_index")
    
    # Check if we have many-to-many matches
    matches_per_left = matched_rows.groupby("_left_index").size()
    if (matches_per_left > 1).any():
        if show_progress:
            max_matches = matches_per_left.max()
            print(f"  Warning: Stage {stage_idx} has many-to-many matches (up to {max_matches} per row)")
    
    # Write outputs for rows that matched
    matched_indices = first_match.index
    if len(matched_indices) > 0:
        # FIXED: Store right indices as a series first
        right_indices = first_match["_right_index"].astype("Int64")
        
        left.loc[matched_indices, "matched_right_index"] = right_indices
        left.loc[matched_indices, "match_type"] = "exact"
        left.loc[matched_indices, "match_stage"] = int(stage_idx)
        left.loc[matched_indices, "match_confidence"] = pd.NA
        
        # FIXED: Pull selected right columns using proper indexing
        for col in right_cols_to_return:
            if col in right.columns:
                if col not in left.columns:
                    left[col] = pd.NA
                # Create a mapping from left index to right value
                for left_idx in matched_indices:
                    right_idx = int(first_match.loc[left_idx, "_right_index"])
                    left.at[left_idx, col] = right.at[right_idx, col]
    
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
    left_id: Optional[str] = None,  # NEW: left dataframe ID variable
    right_id: Optional[str] = None,  # NEW: right dataframe ID variable
    validate_ids: bool = True,  # NEW: validate ID uniqueness
    auto_create_ids: bool = True,  # NEW: auto-create IDs if needed
):
    """
    Perform two-step fuzzy merge: exact matching stages followed by LLM matching.
    
    Parameters
    ----------
    left_df : pd.DataFrame
        Left dataframe to merge
    right_df : pd.DataFrame
        Right dataframe to merge
    exact_stages : List[ExactStage]
        List of exact matching stages
    llm_match : Tuple[str, str]
        (left_col, right_col) for LLM comparison
    right_cols_to_return : List[str]
        Columns from right dataframe to include in output
    llm_backend : LLMBackend
        LLM backend configuration
    left_corrections : Optional[Dict[str, Dict[str, str]]]
        Corrections for left dataframe columns
    right_corrections : Optional[Dict[str, Dict[str, str]]]
        Corrections for right dataframe columns
    llm_candidate_keys : Optional[List[Tuple[str, str]]]
        Keys for LLM candidate grouping (default: last exact stage keys)
    keep_unmatched : bool, default True
        Keep unmatched rows in output
    left_id : Optional[str]
        ID variable for left dataframe. If None or not unique, will create one.
    right_id : Optional[str]
        ID variable for right dataframe. If None or not unique, will create one.
    validate_ids : bool, default True
        Validate ID uniqueness before processing
    auto_create_ids : bool, default True
        Automatically create IDs if validation fails
    
    Returns
    -------
    tuple
        (merged_df, stats_dict, id_info_dict)
    """
    
    # NEW: Validate and ensure unique IDs
    id_info = {}
    
    if validate_ids:
        if llm_backend.show_progress:
            print("\n=== ID Validation ===")
        
        # Handle left dataframe ID
        left_df, left_id_used, left_stats = ensure_unique_id(
            left_df,
            id_var=left_id,
            new_id_name="_left_merge_id",
            force_create=False,
            verbose=llm_backend.show_progress
        )
        id_info["left"] = {
            "original_id": left_id,
            "id_used": left_id_used,
            "stats": left_stats
        }
        
        # Handle right dataframe ID
        right_df, right_id_used, right_stats = ensure_unique_id(
            right_df,
            id_var=right_id,
            new_id_name="_right_merge_id",
            force_create=False,
            verbose=llm_backend.show_progress
        )
        id_info["right"] = {
            "original_id": right_id,
            "id_used": right_id_used,
            "stats": right_stats
        }
        
        if llm_backend.show_progress:
            print(f"\nLeft ID: {left_id_used}")
            print(f"Right ID: {right_id_used}")
            print("=" * 40 + "\n")
    else:
        # If not validating, use IDs as provided or create default
        if left_id is None:
            left_df = left_df.copy()
            left_df["_left_merge_id"] = range(1, len(left_df) + 1)
            left_id_used = "_left_merge_id"
        else:
            left_id_used = left_id
            
        if right_id is None:
            right_df = right_df.copy()
            right_df["_right_merge_id"] = range(1, len(right_df) + 1)
            right_id_used = "_right_merge_id"
        else:
            right_id_used = right_id
        
        id_info = {
            "left": {"id_used": left_id_used, "validated": False},
            "right": {"id_used": right_id_used, "validated": False}
        }
    
    # Store original IDs for tracking
    left = _ensure_output_columns(left_df)
    right = right_df.copy()
    
    # Add ID columns to output columns if not already there
    if left_id_used not in left.columns:
        left[left_id_used] = left_df[left_id_used]
    if right_id_used not in right_cols_to_return and right_id_used in right.columns:
        right_cols_to_return = [right_id_used] + list(right_cols_to_return)
    
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
                        
                        # bring right cols now for this
                                                # bring right cols now for this row
                        for col in right_cols_to_return:
                            if col in right.columns:
                                if col not in left.columns:
                                    left[col] = pd.NA
                                left.at[left_row_index, col] = right.at[right_idx, col]
    
    if not keep_unmatched:
        left = left[left["matched_right_index"].notna()].copy()
    
    # 3) Calculate comprehensive statistics
    matched_mask = left["matched_right_index"].notna()
    
    # Fixed: Handle empty dataframe case for highest_confidence
    llm_matches = left["match_type"] == "llm"
    highest_conf = 0
    if llm_matches.any():
        conf_vals = pd.to_numeric(left.loc[llm_matches, "match_confidence"], errors="coerce")
        if not conf_vals.isna().all():
            highest_conf = int(conf_vals.max())
    
    # Calculate stage-wise statistics
    stage_stats = {}
    for s_idx in range(1, len(exact_stages) + 1):
        stage_matches = (left["match_stage"] == s_idx) & (left["match_type"] == "exact")
        stage_stats[f"stage_{s_idx}"] = int(stage_matches.sum())
    
    # Calculate match rate by stage
    total_left = len(left_df)
    exact_match_count = int(((left["match_type"] == "exact") & matched_mask).sum())
    llm_match_count = int((llm_matches & matched_mask).sum())
    total_matched = int(matched_mask.sum())
    
    stats = {
        "total_left_rows": total_left,
        "total_right_rows": int(len(right_df)),
        "exact_matches": exact_match_count,
        "llm_matches": llm_match_count,
        "match_count": total_matched,
        "unmatched_count": total_left - total_matched,
        "matched_ratio": float(matched_mask.mean()) if total_left > 0 else 0.0,
        "exact_match_ratio": float(exact_match_count / total_left) if total_left > 0 else 0.0,
        "llm_match_ratio": float(llm_match_count / total_left) if total_left > 0 else 0.0,
        "highest_confidence": highest_conf,
        "stages": len(exact_stages),
        "stage_stats": stage_stats,
        "left_id_used": id_info["left"]["id_used"],
        "right_id_used": id_info["right"]["id_used"],
    }
    
    # Add confidence distribution for LLM matches
    if llm_match_count > 0:
        conf_series = pd.to_numeric(left.loc[llm_matches & matched_mask, "match_confidence"], errors="coerce")
        stats["llm_confidence_stats"] = {
            "min": int(conf_series.min()) if not conf_series.isna().all() else 0,
            "max": int(conf_series.max()) if not conf_series.isna().all() else 0,
            "mean": float(conf_series.mean()) if not conf_series.isna().all() else 0.0,
            "median": float(conf_series.median()) if not conf_series.isna().all() else 0.0,
        }
    
    return left, stats, id_info


def validate_merge_inputs(
    left_df: pd.DataFrame,
    right_df: pd.DataFrame,
    exact_stages: List[ExactStage],
    llm_match: Tuple[str, str],
    right_cols_to_return: List[str],
    left_id: Optional[str] = None,
    right_id: Optional[str] = None,
    verbose: bool = True,
) -> Dict:
    """
    Validate inputs before performing merge.
    
    Parameters
    ----------
    left_df : pd.DataFrame
        Left dataframe
    right_df : pd.DataFrame
        Right dataframe
    exact_stages : List[ExactStage]
        Exact matching stages
    llm_match : Tuple[str, str]
        (left_col, right_col) for LLM matching
    right_cols_to_return : List[str]
        Columns to return from right dataframe
    left_id : Optional[str]
        Left dataframe ID variable
    right_id : Optional[str]
        Right dataframe ID variable
    verbose : bool, default True
        Print validation results
    
    Returns
    -------
    Dict
        Validation results with any issues found
    """
    issues = []
    warnings = []
    
    # Check dataframes are not empty
    if left_df.empty:
        issues.append("Left dataframe is empty")
    if right_df.empty:
        issues.append("Right dataframe is empty")
    
    # Check exact stage columns exist
    for stage_idx, stage in enumerate(exact_stages, 1):
        for left_col, right_col in stage.on:
            if left_col not in left_df.columns:
                issues.append(f"Stage {stage_idx}: Left column '{left_col}' not found")
            if right_col not in right_df.columns:
                issues.append(f"Stage {stage_idx}: Right column '{right_col}' not found")
    
    # Check LLM match columns exist
    left_llm_col, right_llm_col = llm_match
    if left_llm_col not in left_df.columns:
        issues.append(f"LLM match: Left column '{left_llm_col}' not found")
    if right_llm_col not in right_df.columns:
        issues.append(f"LLM match: Right column '{right_llm_col}' not found")
    
    # Check right columns to return exist
    for col in right_cols_to_return:
        if col not in right_df.columns:
            warnings.append(f"Column to return '{col}' not found in right dataframe")
    
    # Validate IDs if specified
    id_validation = {}
    if left_id is not None:
        left_id_stats = check_id_uniqueness(left_df, left_id)
        id_validation["left"] = left_id_stats
        if not left_id_stats.get("exists", False):
            issues.append(f"Left ID '{left_id}' not found in dataframe")
        elif not left_id_stats.get("is_unique", False):
            warnings.append(f"Left ID '{left_id}' is not unique ({left_id_stats.get('n_duplicates', 0)} duplicates)")
    
    if right_id is not None:
        right_id_stats = check_id_uniqueness(right_df, right_id)
        id_validation["right"] = right_id_stats
        if not right_id_stats.get("exists", False):
            issues.append(f"Right ID '{right_id}' not found in dataframe")
        elif not right_id_stats.get("is_unique", False):
            warnings.append(f"Right ID '{right_id}' is not unique ({right_id_stats.get('n_duplicates', 0)} duplicates)")
    
    # Check for duplicate stage definitions
    stage_keys = []
    for stage in exact_stages:
        key = tuple(sorted(stage.on))
        if key in stage_keys:
            warnings.append(f"Duplicate exact stage definition found: {stage.on}")
        stage_keys.append(key)
    
    validation_result = {
        "valid": len(issues) == 0,
        "issues": issues,
        "warnings": warnings,
        "id_validation": id_validation,
        "n_left_rows": len(left_df),
        "n_right_rows": len(right_df),
        "n_stages": len(exact_stages),
    }
    
    if verbose:
        print("\n=== Merge Input Validation ===")
        print(f"Left rows: {len(left_df)}")
        print(f"Right rows: {len(right_df)}")
        print(f"Exact stages: {len(exact_stages)}")
        
        if issues:
            print(f"\n❌ Found {len(issues)} issue(s):")
            for issue in issues:
                print(f"  - {issue}")
        else:
            print("\n✓ No critical issues found")
        
        if warnings:
            print(f"\n⚠ Found {len(warnings)} warning(s):")
            for warning in warnings:
                print(f"  - {warning}")
        
        print("=" * 40 + "\n")
    
    return validation_result


def print_merge_summary(stats: Dict, id_info: Dict, verbose: bool = True):
    """
    Print a formatted summary of merge results.
    
    Parameters
    ----------
    stats : Dict
        Statistics dictionary from merge_two_step
    id_info : Dict
        ID information dictionary from merge_two_step
    verbose : bool, default True
        Print detailed information
    """
    if not verbose:
        return
    
    print("\n" + "=" * 60)
    print("MERGE SUMMARY")
    print("=" * 60)
    
    # Input information
    print(f"\nInput:")
    print(f"  Left rows:  {stats['total_left_rows']:,}")
    print(f"  Right rows: {stats['total_right_rows']:,}")
    print(f"  Left ID:    {id_info['left']['id_used']}")
    print(f"  Right ID:   {id_info['right']['id_used']}")
    
    # Match results
    print(f"\nMatch Results:")
    print(f"  Total matched:     {stats['match_count']:,} ({stats['matched_ratio']:.1%})")
    print(f"  Exact matches:     {stats['exact_matches']:,} ({stats['exact_match_ratio']:.1%})")
    print(f"  LLM matches:       {stats['llm_matches']:,} ({stats['llm_match_ratio']:.1%})")
    print(f"  Unmatched:         {stats['unmatched_count']:,}")
    
    # Stage breakdown
    if stats.get('stage_stats'):
        print(f"\nExact Match by Stage:")
        for stage_name, count in stats['stage_stats'].items():
            stage_num = stage_name.split('_')[1]
            pct = (count / stats['total_left_rows'] * 100) if stats['total_left_rows'] > 0 else 0
            print(f"  Stage {stage_num}: {count:,} ({pct:.1f}%)")
    
    # LLM confidence
    if stats['llm_matches'] > 0 and 'llm_confidence_stats' in stats:
        conf = stats['llm_confidence_stats']
        print(f"\nLLM Confidence:")
        print(f"  Min:    {conf['min']}")
        print(f"  Mean:   {conf['mean']:.1f}")
        print(f"  Median: {conf['median']:.1f}")
        print(f"  Max:    {conf['max']}")
    
    # ID warnings
    if id_info['left'].get('stats', {}).get('action') == 'created':
        print(f"\n⚠ Created new left ID: {id_info['left']['id_used']}")
    if id_info['right'].get('stats', {}).get('action') == 'created':
        print(f"⚠ Created new right ID: {id_info['right']['id_used']}")
    
    print("=" * 60 + "\n")


# Convenience function for common use case
def ai_merge(
    left_df: pd.DataFrame,
    right_df: pd.DataFrame,
    on: List[Tuple[str, str]],  # matching columns for exact stage
    llm_on: Tuple[str, str],  # matching columns for LLM
    right_cols: Optional[List[str]] = None,  # columns to return from right
    left_id: Optional[str] = None,
    right_id: Optional[str] = None,
    api_key: Optional[str] = None,
    model: str = "gemini-2.5-pro",
    min_confidence: int = 70,
    keep_unmatched: bool = True,
    validate_first: bool = True,
    show_progress: bool = True,
) -> Tuple[pd.DataFrame, Dict, Dict]:
    """
    Simplified interface for fuzzy merge with single exact stage.
    
    Parameters
    ----------
    left_df : pd.DataFrame
        Left dataframe
    right_df : pd.DataFrame
        Right dataframe
    on : List[Tuple[str, str]]
        List of (left_col, right_col) tuples for exact matching
    llm_on : Tuple[str, str]
        (left_col, right_col) for LLM matching
    right_cols : Optional[List[str]]
        Columns to return from right dataframe (default: all)
    left_id : Optional[str]
        Left dataframe ID variable
    right_id : Optional[str]
        Right dataframe ID variable
    api_key : Optional[str]
        API key for LLM service
    model : str
        LLM model name
    min_confidence : int
        Minimum confidence threshold for LLM matches
    keep_unmatched : bool
        Keep unmatched rows in output
    validate_first : bool
        Validate inputs before merging
    show_progress : bool
        Show progress bars
    
    Returns
    -------
    Tuple[pd.DataFrame, Dict, Dict]
        (merged_df, stats, id_info)
    """
    # Default: return all right columns
    if right_cols is None:
        right_cols = [col for col in right_df.columns if col != right_id]
    
    # Validate inputs
    if validate_first:
        validation = validate_merge_inputs(
            left_df, right_df, 
            [ExactStage(on=on)], 
            llm_on, 
            right_cols,
            left_id, 
            right_id,
            verbose=show_progress
        )
        if not validation["valid"]:
            raise ValueError(f"Validation failed: {validation['issues']}")
    
    # Configure backend
    backend = LLMBackend(
        model=model,
        api_key=api_key,
        min_confidence=min_confidence,
        show_progress=show_progress,
    )
    
    # Run merge
    result_df, stats, id_info = merge_two_step(
        left_df=left_df,
        right_df=right_df,
        exact_stages=[ExactStage(on=on)],
        llm_match=llm_on,
        right_cols_to_return=right_cols,
        llm_backend=backend,
        keep_unmatched=keep_unmatched,
        left_id=left_id,
        right_id=right_id,
    )
    
    # Print summary
    print_merge_summary(stats, id_info, verbose=show_progress)
    
    return result_df, stats, id_info