import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Tuple
import logging

# Import from other files in your package
from .preprocess import (
    add_normalized_columns,
    build_key_series,
    ensure_unique_id
)
from .llm import gemini_batch_select


def hierarchical_llm_match(
    df_left: pd.DataFrame,
    df_right: pd.DataFrame,
    match_cols: List[str],
    hierarchy: Optional[List[List[str]]] = None,
    left_id: Optional[str] = None,
    right_id: Optional[str] = None,
    use_llm: bool = True,
    llm_display_cols: Optional[List[str]] = None,
    # NEW: allow explicitly choosing the columns the LLM should use for matching
    llm_match_cols: Optional[List[str]] = None,
    # NEW: allow different columns per side
    llm_match_cols_left: Optional[List[str]] = None,
    llm_match_cols_right: Optional[List[str]] = None,
    # NEW: whether to use the normalized columns (with `suffix`) for LLM matching
    use_norm_for_llm: bool = True,
    use_norm_for_llm_left: Optional[bool] = None,
    use_norm_for_llm_right: Optional[bool] = None,
    # NEW: threshold for accepting an LLM match (0-100)
    llm_confidence_threshold: int = 70,
    api_key: Optional[str] = None,
    model_name: str = "gemini-2.5-pro",
    temperature: float = 0.2,
    max_workers: int = 8,
    corrections_left: Optional[Dict[str, Dict]] = None,
    corrections_right: Optional[Dict[str, Dict]] = None,
    suffix: str = "__norm",
    verbose: bool = True,
) -> Tuple[pd.DataFrame, Dict]:
    """
    Perform hierarchical exact matching followed by LLM matching for unmatched records.

    Parameters
    ----------
    df_left : pd.DataFrame
        Left dataframe
    df_right : pd.DataFrame
        Right dataframe
    match_cols : List[str]
        Columns to use for matching (must exist in both dataframes)
    hierarchy : List[List[str]], optional
        List of column subsets defining matching hierarchy from most to least specific.
        Example: [['country', 'city', 'unit'], ['country', 'city'], ['country']]
        If None, will use all match_cols as single level.
    left_id : str, optional
        ID column in left dataframe. If None or not unique, will create one.
    right_id : str, optional
        ID column in right dataframe. If None or not unique, will create one.
    use_llm : bool, default True
        Whether to use LLM matching for unmatched records after exact matching
    llm_display_cols : List[str], optional
        Additional columns to display to LLM for context (beyond match_cols)
    llm_match_cols : List[str], optional
        Columns the LLM should use for matching on BOTH sides. If side-specific
        args (llm_match_cols_left/right) are provided they override this.
    llm_match_cols_left/right : List[str], optional
        Side-specific columns for the left/right dataframes. If None, falls back
        to `llm_match_cols` or `match_cols` in that order.
    use_norm_for_llm : bool, default True
        Use normalized columns (with `suffix`) as the inputs to the LLM when available.
    use_norm_for_llm_left/right : Optional[bool]
        Side-specific overrides for whether to use normalized columns. If None,
        falls back to `use_norm_for_llm`.
    llm_confidence_threshold : int, default 70
        Confidence threshold (0-100) to accept an LLM match.
    api_key : str, optional
        Gemini API key (or use GEMINI_API_KEY environment variable)
    model_name : str, default "gemini-2.5-pro"
        Gemini model to use
    temperature : float, default 0.2
        Temperature for LLM generation
    max_workers : int, default 8
        Number of parallel workers for LLM matching
    corrections_left : Dict[str, Dict], optional
        Corrections mapping for left dataframe normalization
    corrections_right : Dict[str, Dict], optional
        Corrections mapping for right dataframe normalization
    suffix : str, default "__norm"
        Suffix for normalized columns
    verbose : bool, default True
        Print matching progress and statistics

    Returns
    -------
    Tuple[pd.DataFrame, Dict]
        - Matched dataframe with columns from both sides
        - Statistics dictionary with matching details
    """

    # Setup logging
    if verbose:
        logging.basicConfig(level=logging.INFO)

    # Validate match_cols exist in both dataframes
    missing_left = [c for c in match_cols if c not in df_left.columns]
    missing_right = [c for c in match_cols if c not in df_right.columns]
    if missing_left or missing_right:
        raise ValueError(
            f"Match columns not found - Left: {missing_left}, Right: {missing_right}"
        )

    # Setup hierarchy
    if hierarchy is None:
        hierarchy = [match_cols]

    # Validate hierarchy
    for level in hierarchy:
        if not all(c in match_cols for c in level):
            raise ValueError(f"Hierarchy level {level} contains columns not in match_cols")

    # Ensure unique IDs
    df_left, left_id_col, left_stats = ensure_unique_id(
        df_left, id_var=left_id, new_id_name="_left_id", verbose=verbose
    )
    df_right, right_id_col, right_stats = ensure_unique_id(
        df_right, id_var=right_id, new_id_name="_right_id", verbose=verbose
    )

    # Add normalized columns
    if verbose:
        print("\nNormalizing columns...")
    df_left = add_normalized_columns(
        df_left, match_cols, corrections=corrections_left, suffix=suffix
    )
    df_right = add_normalized_columns(
        df_right, match_cols, corrections=corrections_right, suffix=suffix
    )

    # Initialize tracking
    matched_pairs = []
    unmatched_left = set(df_left[left_id_col])
    unmatched_right = set(df_right[right_id_col])
    match_stats = {
        "total_left": len(df_left),
        "total_right": len(df_right),
        "exact_matches_by_level": {},
        "llm_matches": 0,
        "unmatched_left": 0,
        "unmatched_right": 0,
    }

    # Hierarchical exact matching
    if verbose:
        print("\n" + "="*60)
        print("HIERARCHICAL EXACT MATCHING")
        print("="*60)

    for level_idx, level_cols in enumerate(hierarchy):
        if verbose:
            print(f"\nLevel {level_idx + 1}: {level_cols}")

        # Build keys for this level
        norm_cols = [f"{c}{suffix}" for c in level_cols]
        left_keys = build_key_series(df_left, norm_cols)
        right_keys = build_key_series(df_right, norm_cols)

        # Create lookup for right side
        right_lookup = (
            df_right[[right_id_col]]
            .assign(key=right_keys)
            .query(f"key != ''")
        )
        right_lookup = right_lookup[right_lookup[right_id_col].isin(unmatched_right)]
        right_dict = right_lookup.groupby('key')[right_id_col].apply(list).to_dict()

        # Match left side
        left_match = (
            df_left[[left_id_col]]
            .assign(key=left_keys)
            .query(f"key != ''")
        )
        left_match = left_match[left_match[left_id_col].isin(unmatched_left)]

        level_matches = 0
        for _, row in left_match.iterrows():
            lid = row[left_id_col]
            key = row['key']

            if key in right_dict and right_dict[key]:
                # Take first available match (could be modified for 1:many)
                rid = right_dict[key].pop(0)
                matched_pairs.append({
                    left_id_col: lid,
                    right_id_col: rid,
                    'match_method': f'exact_level_{level_idx + 1}',
                    'match_confidence': 100
                })
                unmatched_left.discard(lid)
                unmatched_right.discard(rid)
                level_matches += 1

        match_stats['exact_matches_by_level'][f'level_{level_idx + 1}'] = level_matches
        if verbose:
            print(f"  Matches: {level_matches}")
            print(f"  Remaining unmatched - Left: {len(unmatched_left)}, Right: {len(unmatched_right)}")

    # LLM matching for remaining unmatched records
    if use_llm and unmatched_left and unmatched_right:
        if verbose:
            print("\n" + "="*60)
            print("LLM-BASED MATCHING")
            print("="*60)
            print(f"Processing {len(unmatched_left)} unmatched left records...")

        # Resolve side-specific LLM column choices
        # Priority: side-specific -> llm_match_cols -> match_cols
        if llm_match_cols_left is not None:
            left_cols_for_llm = llm_match_cols_left
        elif llm_match_cols is not None:
            left_cols_for_llm = llm_match_cols
        else:
            left_cols_for_llm = match_cols

        if llm_match_cols_right is not None:
            right_cols_for_llm = llm_match_cols_right
        elif llm_match_cols is not None:
            right_cols_for_llm = llm_match_cols
        else:
            right_cols_for_llm = match_cols

        # Resolve normalization flags per side
        if use_norm_for_llm_left is None:
            use_norm_left = use_norm_for_llm
        else:
            use_norm_left = use_norm_for_llm_left

        if use_norm_for_llm_right is None:
            use_norm_right = use_norm_for_llm
        else:
            use_norm_right = use_norm_for_llm_right

        # Build normalized or raw column names per side
        if use_norm_left:
            llm_source_cols_left = [f"{c}{suffix}" for c in left_cols_for_llm]
        else:
            llm_source_cols_left = left_cols_for_llm.copy()

        if use_norm_right:
            llm_source_cols_right = [f"{c}{suffix}" for c in right_cols_for_llm]
        else:
            llm_source_cols_right = right_cols_for_llm.copy()

        # Prepare display/context columns: keep these as original, non-normalized names
        display_cols = match_cols.copy()
        if llm_display_cols:
            display_cols.extend([c for c in llm_display_cols if c not in display_cols])

        # Get unmatched records
        left_unmatched = df_left[df_left[left_id_col].isin(unmatched_left)]
        right_unmatched = df_right[df_right[right_id_col].isin(unmatched_right)]

        # Prepare candidates (right side)
        right_candidates = []
        right_id_list = []
        for _, row in right_unmatched.iterrows():
            parts = []
            # Add right-side match fields (may be normalized names or raw)
            for col in llm_source_cols_right:
                if col in row and pd.notna(row[col]) and str(row[col]).strip() != "":
                    parts.append(f"{col}: {row[col]}")
            # Add context/display columns from original names (if present)
            for col in display_cols:
                if col in row and pd.notna(row[col]) and str(row[col]).strip() != "":
                    parts.append(f"{col}: {row[col]}")
            candidate_str = " | ".join(parts)
            right_candidates.append(candidate_str)
            right_id_list.append(row[right_id_col])

        # Prepare queries (left side)
        queries = []
        left_id_list = []
        for _, row in left_unmatched.iterrows():
            parts = []
            for col in llm_source_cols_left:
                if col in row and pd.notna(row[col]) and str(row[col]).strip() != "":
                    parts.append(f"{col}: {row[col]}")
            for col in display_cols:
                if col in row and pd.notna(row[col]) and str(row[col]).strip() != "":
                    parts.append(f"{col}: {row[col]}")
            query_str = " | ".join(parts)
            queries.append(query_str)
            left_id_list.append(row[left_id_col])

        # Run LLM matching
        candidate_lists = [right_candidates] * len(queries)
        results = gemini_batch_select(
            queries=queries,
            candidate_lists=candidate_lists,
            api_key=api_key,
            model_name=model_name,
            temperature=temperature,
            max_workers=max_workers,
            show_progress=verbose,
        )

        # Process LLM results
        llm_matches = 0
        for i, (best_idx, best_candidate, confidence) in enumerate(results):
            if best_idx >= 0 and confidence >= llm_confidence_threshold:
                lid = left_id_list[i]
                rid = right_id_list[best_idx]
                matched_pairs.append({
                    left_id_col: lid,
                    right_id_col: rid,
                    'match_method': 'llm',
                    'match_confidence': confidence
                })
                unmatched_left.discard(lid)
                unmatched_right.discard(rid)
                llm_matches += 1

        match_stats['llm_matches'] = llm_matches
        if verbose:
            print(f"  LLM Matches: {llm_matches}")

    # Final statistics
    match_stats['unmatched_left'] = len(unmatched_left)
    match_stats['unmatched_right'] = len(unmatched_right)
    match_stats['total_matches'] = len(matched_pairs)
    match_stats['match_rate_left'] = (
        match_stats['total_matches'] / match_stats['total_left'] * 100
        if match_stats['total_left'] > 0 else 0
    )

    # Create matched dataframe
    if matched_pairs:
        matches_df = pd.DataFrame(matched_pairs)
        result = (
            matches_df
            .merge(df_left, on=left_id_col, how='left', suffixes=('', '_left'))
            .merge(df_right, on=right_id_col, how='left', suffixes=('', '_right'))
        )
    else:
        result = pd.DataFrame()

    # Print final statistics
    if verbose:
        print("\n" + "="*60)
        print("MATCHING SUMMARY")
        print("="*60)
        print(f"Total records - Left: {match_stats['total_left']}, Right: {match_stats['total_right']}")
        print(f"Total matches: {match_stats['total_matches']}")
        print(f"Match rate (left): {match_stats['match_rate_left']:.1f}%")
        print(f"Unmatched - Left: {match_stats['unmatched_left']}, Right: {match_stats['unmatched_right']}")

        print("\nBreakdown by method:")
        for level, count in match_stats['exact_matches_by_level'].items():
            print(f"  {level}: {count}")
        if use_llm:
            print(f"  LLM: {match_stats['llm_matches']}")

    return result, match_stats
