"""
Focus Group Transcript Processor

This module provides functionality for preprocessing focus group transcripts,
including text cleaning, speaker separation, and barrier analysis.
"""

import pandas as pd
import re
from pathlib import Path
from typing import Dict, List, Union, Tuple
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tokenize import sent_tokenize
from collections import defaultdict
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

class FocusGroupProcessor:
    def __init__(self, data_dir: Union[str, Path]):
        """Initialize the focus group processor."""
        self.data_dir = Path(data_dir)
        self.processed_data = None
        
        # Initialize NLTK components
        try:
            nltk.data.find('vader_lexicon')
            nltk.data.find('punkt')
        except LookupError:
            nltk.download('vader_lexicon')
            nltk.download('punkt')
        
        self.sia = SentimentIntensityAnalyzer()
        
        # Define barrier categories
        self.barrier_keywords = {
            'economic': ['money', 'salary', 'pay', 'debt', 'loan', 'cost', 'financial', 'afford', 'expensive'],
            'educational': ['school', 'degree', 'phd', 'masters', 'education', 'competitive', 'training', 'program'],
            'personal': ['stress', 'emotional', 'burden', 'toll', 'life', 'balance', 'time', 'difficult'],
            'knowledge': ['know', 'understand', 'aware', 'path', 'career', 'option', 'information'],
            'work_life': ['family', 'children', 'schedule', 'hours', 'flexibility', 'vacation', 'weekend']
        }
        
        # Define solution indicators
        self.solution_indicators = [
            'could', 'should', 'would', 'maybe', 'perhaps', 'suggest', 'recommend',
            'solution', 'help', 'improve', 'better', 'change', 'implement', 'try'
        ]
        
    def load_transcript(self, file_path: Union[str, Path]) -> pd.DataFrame:
        """Load a focus group transcript CSV file."""
        return pd.read_csv(file_path)
    
    def clean_text(self, text: str) -> str:
        """Clean text by removing special characters and normalizing whitespace."""
        text = re.sub(r'[^\w\s.,!?-]', '', text)
        return ' '.join(text.split()).strip()
    
    def get_descriptive_statistics(self) -> Dict:
        """Generate detailed descriptive statistics about the focus group."""
        if self.processed_data is None:
            raise ValueError("No processed data available. Run process_all_transcripts first.")
            
        stats = {
            'overview': {
                'total_duration_minutes': self.processed_data['duration_seconds'].sum() / 60,
                'total_turns': len(self.processed_data),
                'unique_speakers': self.processed_data['Speaker'].nunique(),
                'avg_turn_length_words': self.processed_data['word_count'].mean(),
                'total_words': self.processed_data['word_count'].sum()
            },
            'speaker_engagement': {
                'turns_per_speaker': self.processed_data.groupby('Speaker').size().to_dict(),
                'words_per_speaker': self.processed_data.groupby('Speaker')['word_count'].sum().to_dict(),
                'avg_sentiment_per_speaker': self.processed_data.groupby('Speaker')['sentiment_compound'].mean().to_dict()
            },
            'conversation_flow': {
                'avg_turn_duration': self.processed_data['duration_seconds'].mean(),
                'max_turn_duration': self.processed_data['duration_seconds'].max(),
                'speaking_rate_stats': {
                    'mean': self.processed_data['speaking_rate'].mean(),
                    'std': self.processed_data['speaking_rate'].std(),
                    'min': self.processed_data['speaking_rate'].min(),
                    'max': self.processed_data['speaking_rate'].max()
                }
            },
            'content_analysis': {
                'barrier_mentions': {
                    category: self.processed_data[f'barrier_{category}'].sum()
                    for category in self.barrier_keywords.keys()
                },
                'sentiment_distribution': {
                    'positive_turns': (self.processed_data['sentiment_compound'] > 0.05).sum(),
                    'neutral_turns': ((self.processed_data['sentiment_compound'] >= -0.05) & 
                                    (self.processed_data['sentiment_compound'] <= 0.05)).sum(),
                    'negative_turns': (self.processed_data['sentiment_compound'] < -0.05).sum()
                }
            }
        }
        
        return stats
    
    def classify_utterance(self, text: str) -> Dict[str, bool]:
        """Classify whether an utterance discusses barriers or solutions."""
        text_lower = text.lower()
        
        # Check for solution indicators
        is_solution = any(indicator in text_lower for indicator in self.solution_indicators)
        
        # Check for barrier mentions
        has_barrier = any(
            any(keyword in text_lower for keyword in keywords)
            for keywords in self.barrier_keywords.values()
        )
        
        return {
            'is_barrier': has_barrier,
            'is_solution': is_solution,
            'is_both': has_barrier and is_solution
        }
    
    def get_top_ngrams(self, n: int = 2, top_k: int = 20) -> Dict[str, List[Tuple[str, int]]]:
        """Get the most frequent n-grams from the transcripts."""
        vectorizer = CountVectorizer(ngram_range=(n, n), stop_words='english')
        ngrams = vectorizer.fit_transform(self.processed_data['cleaned_text'])
        
        # Get feature names and frequencies
        feature_names = vectorizer.get_feature_names_out()
        frequencies = ngrams.sum(axis=0).A1
        
        # Sort by frequency
        top_indices = frequencies.argsort()[-top_k:][::-1]
        
        return [
            (feature_names[i], frequencies[i])
            for i in top_indices
        ]
    
    def process_transcript(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process a transcript DataFrame."""
        # Existing processing
        df['cleaned_text'] = df['Text'].apply(self.clean_text)
        df['duration_seconds'] = pd.to_timedelta(df['Duration']).dt.total_seconds()
        df['word_count'] = df['cleaned_text'].str.split().str.len()
        df['speaking_rate'] = (df['word_count'] / df['duration_seconds']) * 60
        
        # Sentiment analysis
        sentiments = df['cleaned_text'].apply(self.analyze_sentiment)
        df['sentiment_compound'] = [s['compound'] for s in sentiments]
        df['sentiment_positive'] = [s['pos'] for s in sentiments]
        df['sentiment_negative'] = [s['neg'] for s in sentiments]
        
        # Barrier analysis
        barriers = df['cleaned_text'].apply(self.identify_barriers)
        df['barriers'] = barriers
        
        # Add barrier category columns
        for category in self.barrier_keywords.keys():
            df[f'barrier_{category}'] = df['barriers'].apply(
                lambda x: len(x.get(category, [])) if category in x else 0
            )
        
        # Add utterance classification
        classifications = df['cleaned_text'].apply(self.classify_utterance)
        df['is_barrier'] = [c['is_barrier'] for c in classifications]
        df['is_solution'] = [c['is_solution'] for c in classifications]
        df['is_both'] = [c['is_both'] for c in classifications]
        
        # Add sentence count
        df['sentence_count'] = df['cleaned_text'].apply(lambda x: len(sent_tokenize(x)))
        
        return df
    
    def get_speaker_statistics(self, df: pd.DataFrame) -> Dict[str, Dict]:
        """Calculate statistics for each speaker."""
        stats = {}
        for speaker in df['Speaker'].unique():
            speaker_df = df[df['Speaker'] == speaker]
            stats[speaker] = {
                'total_turns': len(speaker_df),
                'total_words': speaker_df['word_count'].sum(),
                'total_duration': speaker_df['duration_seconds'].sum(),
                'avg_speaking_rate': speaker_df['speaking_rate'].mean(),
                'avg_sentiment': speaker_df['sentiment_compound'].mean(),
                'barriers_mentioned': {
                    category: speaker_df[f'barrier_{category}'].sum()
                    for category in self.barrier_keywords.keys()
                },
                'utterance_types': {
                    'barrier_statements': speaker_df['is_barrier'].sum(),
                    'solution_statements': speaker_df['is_solution'].sum(),
                    'combined_statements': speaker_df['is_both'].sum()
                }
            }
        return stats
    
    def analyze_sentiment(self, text: str) -> Dict[str, float]:
        """Analyze sentiment of text using VADER."""
        return self.sia.polarity_scores(text)
    
    def identify_barriers(self, text: str) -> Dict[str, List[str]]:
        """Identify mentions of different types of barriers in text."""
        text_lower = text.lower()
        found_barriers = defaultdict(list)
        
        for category, keywords in self.barrier_keywords.items():
            for keyword in keywords:
                if keyword in text_lower:
                    found_barriers[category].append(keyword)
        
        return dict(found_barriers)
    
    def process_all_transcripts(self) -> None:
        """Process all transcript files in the data directory."""
        all_data = []
        for file_path in self.data_dir.glob('*.csv'):
            df = self.load_transcript(file_path)
            processed_df = self.process_transcript(df)
            processed_df['source_file'] = file_path.name
            all_data.append(processed_df)
        
        self.processed_data = pd.concat(all_data, ignore_index=True)
    
    def save_processed_data(self, output_path: Union[str, Path]) -> None:
        """Save processed data to a file."""
        if self.processed_data is not None:
            self.processed_data.to_csv(output_path, index=False)
        else:
            raise ValueError("No processed data available. Run process_all_transcripts first.") 