# ============================================================
# src/ai_merge/__init__.py
# ============================================================

"""
AI Merge Package - Hierarchical exact and LLM-based record matching
"""

from .preprocess import (
    normalize_series,
    add_normalized_columns,
    build_key_series,
    check_id_uniqueness,
    create_sequential_id,
    ensure_unique_id
)

from .llm import (
    gemini_select_best,
    gemini_batch_select
)

from .matching import hierarchical_llm_match

__version__ = "0.1.0"

__all__ = [
    # Main matching function
    'hierarchical_llm_match',
    
    # Preprocessing utilities
    'normalize_series',
    'add_normalized_columns',
    'build_key_series',
    'check_id_uniqueness',
    'create_sequential_id',
    'ensure_unique_id',
    
    # LLM utilities
    'gemini_select_best',
    'gemini_batch_select',
]