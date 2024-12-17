#!/usr/bin/env python3
"""
Initial Analysis of Focus Group Data

This script generates descriptive analyses and visualizations of the focus group discussions.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from wordcloud import WordCloud
import re
from collections import Counter

def load_processed_data(processed_dir: Path) -> tuple:
    """Load all processed data files."""
    # Load session data
    sessions = {}
    for file in processed_dir.glob('*_processed.csv'):
        session_id = file.stem.replace('_processed', '')
        sessions[session_id] = pd.read_csv(file)
    
    # Load summary data
    summaries = pd.read_csv(processed_dir / 'session_summaries.csv')
    
    with open(processed_dir / 'overall_statistics.json', 'r') as f:
        overall_stats = json.load(f)
        
    return sessions, summaries, overall_stats

def plot_session_durations(summaries: pd.DataFrame, output_dir: Path):
    """Plot session durations."""
    plt.figure(figsize=(10, 6))
    bars = plt.bar(summaries['session_id'], summaries['total_duration_minutes'])
    plt.xticks(rotation=45, ha='right')
    plt.title('Duration of Each Focus Group Session')
    plt.xlabel('Session')
    plt.ylabel('Duration (minutes)')
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}m',
                ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'session_durations.png')
    plt.close()

def plot_participant_engagement(summaries: pd.DataFrame, output_dir: Path):
    """Plot participant engagement metrics."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Number of participants
    bars1 = ax1.bar(summaries['session_id'], summaries['unique_speakers'])
    ax1.set_title('Number of Participants per Session')
    ax1.set_xlabel('Session')
    ax1.set_ylabel('Number of Unique Speakers')
    ax1.tick_params(axis='x', rotation=45)
    
    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom')
    
    # Average turns per participant
    avg_turns = summaries['total_turns'] / summaries['unique_speakers']
    bars2 = ax2.bar(summaries['session_id'], avg_turns)
    ax2.set_title('Average Turns per Participant')
    ax2.set_xlabel('Session')
    ax2.set_ylabel('Average Number of Turns')
    ax2.tick_params(axis='x', rotation=45)
    
    # Add value labels
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}',
                ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'participant_engagement.png')
    plt.close()

def analyze_speaker_patterns(sessions: dict, output_dir: Path):
    """Analyze and visualize speaker patterns."""
    # Combine all sessions
    all_turns = []
    for session_id, df in sessions.items():
        turns = df.groupby('Speaker')['word_count'].agg(['count', 'sum', 'mean'])
        turns['session_id'] = session_id
        all_turns.append(turns)
    
    combined_turns = pd.concat(all_turns)
    
    # Plot average words per turn by speaker role
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=combined_turns.reset_index(), x='Speaker', y='mean')
    plt.title('Words per Turn by Speaker Role')
    plt.xlabel('Speaker')
    plt.ylabel('Average Words per Turn')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_dir / 'words_per_turn.png')
    plt.close()

def create_topic_wordcloud(sessions: dict, output_dir: Path):
    """Create word clouds for participant responses."""
    # Combine all participant (non-moderator) text
    participant_text = []
    for df in sessions.values():
        # Filter out moderator (Speaker 1) and clean text
        participant_responses = df[df['Speaker'] != 'Speaker 1']['cleaned_text']
        participant_text.extend(participant_responses)
    
    # Join all text
    text = ' '.join(participant_text)
    
    # Create and generate a word cloud image
    wordcloud = WordCloud(
        width=800, height=400,
        background_color='white',
        max_words=100,
        collocations=False
    ).generate(text)
    
    # Display the word cloud
    plt.figure(figsize=(20,10))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('Most Common Words in Participant Responses')
    plt.tight_layout(pad=0)
    plt.savefig(output_dir / 'response_wordcloud.png')
    plt.close()

def analyze_turn_taking(sessions: dict, output_dir: Path):
    """Analyze turn-taking patterns in discussions."""
    # Analyze turn transitions
    all_transitions = []
    
    for session_id, df in sessions.items():
        # Create pairs of consecutive speakers
        transitions = pd.DataFrame({
            'from_speaker': df['Speaker'].iloc[:-1].values,
            'to_speaker': df['Speaker'].iloc[1:].values,
            'session_id': session_id
        })
        all_transitions.append(transitions)
    
    combined_transitions = pd.concat(all_transitions)
    
    # Create transition matrix
    transition_matrix = pd.crosstab(
        combined_transitions['from_speaker'],
        combined_transitions['to_speaker']
    )
    
    # Plot heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(transition_matrix, annot=True, fmt='d', cmap='YlOrRd')
    plt.title('Speaker Transition Patterns')
    plt.tight_layout()
    plt.savefig(output_dir / 'turn_taking_patterns.png')
    plt.close()

def main():
    """Run all analyses."""
    # Set up directories
    processed_dir = Path('data/processed')
    output_dir = Path('outputs/initial_analysis')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    sessions, summaries, overall_stats = load_processed_data(processed_dir)
    
    # Generate plots
    plot_session_durations(summaries, output_dir)
    plot_participant_engagement(summaries, output_dir)
    analyze_speaker_patterns(sessions, output_dir)
    create_topic_wordcloud(sessions, output_dir)
    analyze_turn_taking(sessions, output_dir)
    
    # Save summary statistics
    with open(output_dir / 'analysis_summary.txt', 'w') as f:
        f.write("Focus Group Analysis Summary\n")
        f.write("==========================\n\n")
        
        f.write("Overall Statistics:\n")
        f.write("-----------------\n")
        for key, value in overall_stats.items():
            f.write(f"{key}: {value}\n")
        
        f.write("\nSession Details:\n")
        f.write("--------------\n")
        f.write(summaries.to_string())

if __name__ == "__main__":
    main() 