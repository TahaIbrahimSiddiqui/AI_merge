import pandas as pd
import os
from ai_merge import ai_merge, merge_two_step, ExactStage, LLMBackend

# ============================================
# Set your API key here
# ============================================
GEMINI_API_KEY = ""  # Replace with your actual key

left_df = pd.DataFrame({
    'company': ['Apple Store', 'Microsoft Office', 'Google Campus', 'Amazon Hub'],
    'city': ['Cupertino', 'Redmond', 'Mountain View', 'Seattle'],
    'state': ['CA', 'WA', 'CA', 'WA'],
    'country': ['USA', 'USA', 'USA', 'USA']
})

right_df = pd.DataFrame({
    'official_name': ['Apple Inc.', 'Microsoft Corp', 'Google LLC', 'Amazon.com'],
    'location_city': ['Cupertino', 'Redmond', 'Mountain View', 'Seattle'],
    'location_state': ['CA', 'WA', 'CA', 'WA'],
    'location_country': ['USA', 'USA', 'USA', 'USA']
})

# Method 1: Using merge_two_step with multiple exact stages
result_df, stats, id_info = merge_two_step(
    left_df=left_df,
    right_df=right_df,
    exact_stages=[
        # Stage 1: Match on country + state + city (most specific)
        ExactStage(on=[
            ('country', 'location_country'),
            ('state', 'location_state'),
            ('city', 'location_city')
        ]),
    ],
    llm_match=('company', 'official_name'),  # LLM stage for remaining
    right_cols_to_return=['official_name', 'location_city'],
    llm_backend=LLMBackend(
        api_key="your-api-key",
        show_progress=True
    )
)

print(result_df)