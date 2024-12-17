"""
Data loader module for focus group analysis
"""

import pandas as pd
from pathlib import Path

def load_focus_group_data():
    """Load and prepare focus group discussion data."""
    # Load data from processed directory
    processed_dir = Path('data/processed')
    
    # Load all processed session files
    all_responses = []
    for file in processed_dir.glob('*_processed.csv'):
        df = pd.read_csv(file)
        # Filter out moderator (Speaker 1)
        student_responses = df[df['Speaker'] != 'Speaker 1']
        all_responses.extend(student_responses['cleaned_text'].tolist())
    
    return all_responses 