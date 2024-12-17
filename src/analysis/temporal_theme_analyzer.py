"""
Temporal Pattern and Theme Analysis

This module provides functionality for analyzing temporal patterns in focus group
conversations and generating themes using embeddings.
"""

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

class TemporalThemeAnalyzer:
    def __init__(self):
        """Initialize the analyzer with a sentence transformer model."""
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.embeddings = None
        self.processed_data = None
        
    def process_transcript(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process transcript with temporal information and embeddings."""
        # Convert time strings to datetime
        df['start_time'] = pd.to_datetime(df['In'], format='%H:%M:%S.%f').dt.time
        df['end_time'] = pd.to_datetime(df['Out'], format='%H:%M:%S.%f').dt.time
        
        # Generate embeddings for each utterance
        texts = df['Text'].tolist()
        self.embeddings = self.model.encode(texts)
        
        # Calculate similarity matrix
        similarity_matrix = cosine_similarity(self.embeddings)
        
        # Add average similarity with other utterances
        df['avg_similarity'] = similarity_matrix.mean(axis=1)
        
        self.processed_data = df
        return df
    
    def identify_themes(self, min_samples: int = 3, eps: float = 0.3) -> Dict:
        """Identify themes using DBSCAN clustering on embeddings."""
        if self.embeddings is None:
            raise ValueError("No embeddings available. Process transcript first.")
        
        # Cluster embeddings
        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(self.embeddings)
        
        # Add cluster labels to processed data
        self.processed_data['theme_cluster'] = clustering.labels_
        
        # Generate theme summaries
        themes = {}
        for cluster_id in set(clustering.labels_):
            if cluster_id == -1:  # Skip noise points
                continue
                
            cluster_texts = self.processed_data[
                self.processed_data['theme_cluster'] == cluster_id
            ]['Text'].tolist()
            
            # Find most central text in cluster
            cluster_embeddings = self.embeddings[clustering.labels_ == cluster_id]
            centroid = cluster_embeddings.mean(axis=0)
            distances = np.linalg.norm(cluster_embeddings - centroid, axis=1)
            representative_idx = distances.argmin()
            
            themes[f"Theme_{cluster_id}"] = {
                'size': len(cluster_texts),
                'representative_quote': cluster_texts[representative_idx],
                'utterance_count': len(cluster_texts),
                'speakers': self.processed_data[
                    self.processed_data['theme_cluster'] == cluster_id
                ]['Speaker'].unique().tolist()
            }
        
        return themes
    
    def analyze_temporal_patterns(self) -> Dict:
        """Analyze temporal patterns in the conversation."""
        if self.processed_data is None:
            raise ValueError("No processed data available. Process transcript first.")
        
        # Calculate time-based metrics
        patterns = {
            'turn_taking': self._analyze_turn_taking(),
            'speaker_activity': self._analyze_speaker_activity(),
            'theme_evolution': self._analyze_theme_evolution()
        }
        
        return patterns
    
    def _analyze_turn_taking(self) -> Dict:
        """Analyze turn-taking patterns between speakers."""
        # Calculate transition matrix
        speakers = self.processed_data['Speaker'].unique()
        transitions = pd.crosstab(
            self.processed_data['Speaker'],
            self.processed_data['Speaker'].shift(-1)
        )
        
        return {
            'transition_matrix': transitions.to_dict(),
            'avg_turn_duration': self.processed_data['duration_seconds'].mean(),
            'max_consecutive_turns': self._get_max_consecutive_turns()
        }
    
    def _analyze_speaker_activity(self) -> Dict:
        """Analyze speaker activity patterns over time."""
        # Calculate activity metrics in 5-minute windows
        self.processed_data['time_window'] = pd.to_datetime(
            self.processed_data['In'], 
            format='%H:%M:%S.%f'
        ).dt.floor('5T')
        
        activity = self.processed_data.groupby(['time_window', 'Speaker']).agg({
            'duration_seconds': 'sum',
            'Text': 'count'
        }).reset_index()
        
        return activity.to_dict(orient='records')
    
    def _analyze_theme_evolution(self) -> Dict:
        """Analyze how themes evolve over time."""
        if 'theme_cluster' not in self.processed_data.columns:
            self.identify_themes()
        
        # Track theme occurrence over time
        theme_evolution = self.processed_data.groupby(
            ['time_window', 'theme_cluster']
        ).size().reset_index(name='count')
        
        return theme_evolution.to_dict(orient='records')
    
    def _get_max_consecutive_turns(self) -> Dict:
        """Calculate maximum consecutive turns for each speaker."""
        max_consecutive = {}
        current_speaker = None
        current_count = 0
        
        for speaker in self.processed_data['Speaker']:
            if speaker == current_speaker:
                current_count += 1
            else:
                if current_speaker is not None:
                    max_consecutive[current_speaker] = max(
                        max_consecutive.get(current_speaker, 0),
                        current_count
                    )
                current_speaker = speaker
                current_count = 1
        
        return max_consecutive
    
    def plot_temporal_patterns(self, output_path: str):
        """Generate visualizations of temporal patterns."""
        patterns = self.analyze_temporal_patterns()
        
        # Create figure with subplots
        fig = plt.figure(figsize=(15, 10))
        gs = fig.add_gridspec(2, 2)
        
        # 1. Speaker activity over time
        ax1 = fig.add_subplot(gs[0, :])
        activity_df = pd.DataFrame(patterns['speaker_activity'])
        sns.lineplot(
            data=activity_df,
            x='time_window',
            y='duration_seconds',
            hue='Speaker',
            ax=ax1
        )
        ax1.set_title('Speaker Activity Over Time')
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Speaking Duration (seconds)')
        
        # 2. Turn-taking heatmap
        ax2 = fig.add_subplot(gs[1, 0])
        transition_df = pd.DataFrame(patterns['turn_taking']['transition_matrix'])
        sns.heatmap(
            transition_df,
            annot=True,
            fmt='d',
            cmap='YlOrRd',
            ax=ax2
        )
        ax2.set_title('Speaker Transition Patterns')
        
        # 3. Theme evolution
        ax3 = fig.add_subplot(gs[1, 1])
        theme_df = pd.DataFrame(patterns['theme_evolution'])
        sns.scatterplot(
            data=theme_df,
            x='time_window',
            y='theme_cluster',
            size='count',
            ax=ax3
        )
        ax3.set_title('Theme Evolution Over Time')
        
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
    
    def generate_theme_network(self, output_path: str):
        """Generate a network visualization of related themes."""
        if self.embeddings is None or 'theme_cluster' not in self.processed_data.columns:
            raise ValueError("Process transcript and identify themes first.")
        
        # Calculate theme centroids
        theme_centroids = {}
        for theme in self.processed_data['theme_cluster'].unique():
            if theme == -1:  # Skip noise points
                continue
            mask = self.processed_data['theme_cluster'] == theme
            theme_centroids[theme] = self.embeddings[mask].mean(axis=0)
        
        # Calculate theme similarities
        theme_similarities = pd.DataFrame(
            cosine_similarity([v for v in theme_centroids.values()]),
            index=theme_centroids.keys(),
            columns=theme_centroids.keys()
        )
        
        # Plot theme network
        plt.figure(figsize=(12, 8))
        sns.heatmap(
            theme_similarities,
            annot=True,
            fmt='.2f',
            cmap='viridis'
        )
        plt.title('Theme Similarity Network')
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close() 