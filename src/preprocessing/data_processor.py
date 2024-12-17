"""
Data Processing Module

This module handles the cleaning, validation, and standardization of focus group transcripts.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Union
import re
from datetime import datetime

class DataProcessor:
    def __init__(self, raw_dir: Union[str, Path]):
        """Initialize the data processor."""
        self.raw_dir = Path(raw_dir)
        self.processed_data = {}
        
    def load_all_sessions(self) -> Dict[str, pd.DataFrame]:
        """Load all session files from the raw directory."""
        sessions = {}
        for file_path in self.raw_dir.glob('*.csv'):
            session_id = self._extract_session_id(file_path.name)
            sessions[session_id] = pd.read_csv(file_path)
        return sessions
    
    def process_all_sessions(self) -> Dict[str, pd.DataFrame]:
        """Process all session files."""
        raw_sessions = self.load_all_sessions()
        
        for session_id, df in raw_sessions.items():
            self.processed_data[session_id] = self.process_session(df)
            
        return self.processed_data
    
    def process_session(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and process a single session DataFrame."""
        # Create a copy to avoid modifying original
        df = df.copy()
        
        # Clean and standardize columns
        df = self._clean_columns(df)
        
        # Clean text
        df['cleaned_text'] = df['Text'].apply(self._clean_text)
        
        # Standardize speaker labels
        df['Speaker'] = df['Speaker'].apply(self._standardize_speaker)
        
        # Process timestamps
        df = self._process_timestamps(df)
        
        # Add derived features
        df = self._add_features(df)
        
        return df
    
    def _clean_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure all required columns are present and properly named."""
        required_columns = {'In', 'Out', 'Duration', 'Text', 'Speaker', 'Status'}
        
        # Check for missing columns
        missing_cols = required_columns - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        return df
    
    def _clean_text(self, text: str) -> str:
        """Clean and standardize text content."""
        if pd.isna(text):
            return ""
            
        # Convert to string if not already
        text = str(text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s.,!?-]', '', text)
        
        return text.strip()
    
    def _standardize_speaker(self, speaker: str) -> str:
        """Standardize speaker labels."""
        if pd.isna(speaker):
            return "Unknown"
            
        # Remove any extra whitespace
        speaker = speaker.strip()
        
        # Ensure consistent format (e.g., "Speaker 1" instead of "speaker1")
        if re.match(r'^speaker\s*\d+$', speaker.lower()):
            num = re.search(r'\d+', speaker).group()
            return f"Speaker {num}"
            
        return speaker
    
    def _process_timestamps(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process and validate timestamps."""
        # Convert timestamps to datetime.time objects
        df['start_time'] = pd.to_datetime(df['In'], format='%H:%M:%S.%f').dt.time
        df['end_time'] = pd.to_datetime(df['Out'], format='%H:%M:%S.%f').dt.time
        
        # Calculate duration in seconds
        df['duration_seconds'] = pd.to_timedelta(df['Duration']).dt.total_seconds()
        
        return df
    
    def _add_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add derived features useful for analysis."""
        # Add word count
        df['word_count'] = df['cleaned_text'].str.split().str.len()
        
        # Calculate speaking rate (words per minute)
        df['speaking_rate'] = (df['word_count'] / df['duration_seconds']) * 60
        
        # Add turn number within session
        df['turn_number'] = range(1, len(df) + 1)
        
        # Add time since start of session (in seconds)
        df['time_from_start'] = pd.to_timedelta(df['In']).dt.total_seconds()
        
        return df
    
    def _extract_session_id(self, filename: str) -> str:
        """Extract a clean session ID from filename."""
        # Remove extension and common text
        session_id = filename.replace('.csv', '')
        session_id = session_id.replace('_Focus_Group_full', '')
        session_id = session_id.replace('_group_full', '')
        session_id = session_id.replace('_group__full', '')
        
        return session_id
    
    def save_processed_sessions(self, output_dir: Union[str, Path]) -> None:
        """Save processed sessions to CSV files."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for session_id, df in self.processed_data.items():
            output_path = output_dir / f"{session_id}_processed.csv"
            df.to_csv(output_path, index=False)
            
    def get_session_summaries(self) -> pd.DataFrame:
        """Generate summary statistics for each session."""
        summaries = []
        
        for session_id, df in self.processed_data.items():
            summary = {
                'session_id': session_id,
                'total_duration_minutes': df['duration_seconds'].sum() / 60,
                'total_turns': len(df),
                'unique_speakers': df['Speaker'].nunique(),
                'total_words': df['word_count'].sum(),
                'avg_turn_length_words': df['word_count'].mean(),
                'avg_speaking_rate': df['speaking_rate'].mean(),
                'most_active_speaker': df.groupby('Speaker')['word_count'].sum().idxmax()
            }
            summaries.append(summary)
            
        return pd.DataFrame(summaries) 