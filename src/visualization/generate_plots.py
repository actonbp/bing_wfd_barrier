#!/usr/bin/env python3
"""
Generate Visualizations

This script generates visualizations from the processed focus group data,
including barrier analysis and solution patterns.
"""

import argparse
import json
import logging
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from wordcloud import WordCloud

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def plot_barrier_solution_distribution(utterance_path: Path, output_path: Path):
    """Generate visualization of barrier vs solution distribution."""
    with open(utterance_path) as f:
        data = json.load(f)
    
    # Create summary for plotting
    summary_data = pd.DataFrame([
        {'Type': 'Barriers Only', 
         'Count': data['barrier_statements'] - data['combined_statements']},
        {'Type': 'Solutions Only', 
         'Count': data['solution_statements'] - data['combined_statements']},
        {'Type': 'Both', 
         'Count': data['combined_statements']}
    ])
    
    plt.figure(figsize=(10, 6))
    sns.barplot(data=summary_data, x='Type', y='Count')
    plt.title('Distribution of Barrier and Solution Statements')
    plt.ylabel('Number of Utterances')
    
    plt.tight_layout()
    plt.savefig(output_path / 'barrier_solution_distribution.png')
    plt.close()

def plot_barrier_types(grant_summary_path: Path, output_path: Path):
    """Generate visualization of barrier types."""
    with open(grant_summary_path) as f:
        data = json.load(f)
    
    barriers = pd.DataFrame.from_dict(
        data['barrier_distribution'], 
        orient='index', 
        columns=['count']
    ).reset_index()
    barriers.columns = ['Category', 'Count']
    
    plt.figure(figsize=(12, 6))
    sns.barplot(data=barriers, x='Category', y='Count')
    plt.title('Distribution of Barrier Types')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig(output_path / 'barrier_types.png')
    plt.close()

def plot_speaker_engagement(desc_stats_path: Path, output_path: Path):
    """Generate visualization of speaker engagement patterns."""
    with open(desc_stats_path) as f:
        data = json.load(f)
    
    # Create speaker engagement summary
    engagement = pd.DataFrame.from_dict(
        data['speaker_engagement']['turns_per_speaker'],
        orient='index',
        columns=['Turns']
    ).reset_index()
    engagement.columns = ['Speaker', 'Turns']
    
    words = pd.DataFrame.from_dict(
        data['speaker_engagement']['words_per_speaker'],
        orient='index',
        columns=['Words']
    ).reset_index()
    words.columns = ['Speaker', 'Words']
    
    # Plot turns and words
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    sns.barplot(data=engagement, x='Speaker', y='Turns', ax=ax1)
    ax1.set_title('Number of Speaking Turns by Speaker')
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45)
    
    sns.barplot(data=words, x='Speaker', y='Words', ax=ax2)
    ax2.set_title('Total Words Spoken by Speaker')
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45)
    
    plt.tight_layout()
    plt.savefig(output_path / 'speaker_engagement.png')
    plt.close()

def plot_ngram_wordcloud(ngram_path: Path, output_path: Path):
    """Generate word clouds for n-grams."""
    with open(ngram_path) as f:
        ngrams = json.load(f)
    
    for n_type, data in ngrams.items():
        # Create frequency dictionary
        freq_dict = {' '.join(item[0]): item[1] for item in data}
        
        # Generate word cloud
        wordcloud = WordCloud(
            width=800, height=400,
            background_color='white',
            max_words=50
        ).generate_from_frequencies(freq_dict)
        
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title(f'Most Common {n_type.capitalize()}')
        
        plt.tight_layout()
        plt.savefig(output_path / f'{n_type}_wordcloud.png')
        plt.close()

def plot_sentiment_by_topic(processed_data_path: Path, output_path: Path):
    """Generate visualization of sentiment distribution by topic."""
    df = pd.read_csv(processed_data_path)
    
    # Calculate average sentiment for different types of statements
    sentiment_data = pd.DataFrame({
        'Type': ['Barrier', 'Solution', 'Combined'],
        'Sentiment': [
            df[df['is_barrier'] & ~df['is_solution']]['sentiment_compound'].mean(),
            df[df['is_solution'] & ~df['is_barrier']]['sentiment_compound'].mean(),
            df[df['is_both']]['sentiment_compound'].mean()
        ]
    })
    
    plt.figure(figsize=(10, 6))
    sns.barplot(data=sentiment_data, x='Type', y='Sentiment')
    plt.title('Average Sentiment by Statement Type')
    plt.ylabel('Average Sentiment Score')
    
    plt.tight_layout()
    plt.savefig(output_path / 'sentiment_by_topic.png')
    plt.close()

def main(processed_data_path: str, desc_stats_path: str, utterance_path: str, 
         ngram_path: str, grant_summary_path: str, output_dir: str):
    """Generate all visualizations."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    logger.info("Generating barrier/solution distribution plot...")
    plot_barrier_solution_distribution(Path(utterance_path), output_path)
    
    logger.info("Generating barrier types plot...")
    plot_barrier_types(Path(grant_summary_path), output_path)
    
    logger.info("Generating speaker engagement plots...")
    plot_speaker_engagement(Path(desc_stats_path), output_path)
    
    logger.info("Generating n-gram word clouds...")
    plot_ngram_wordcloud(Path(ngram_path), output_path)
    
    logger.info("Generating sentiment analysis plots...")
    plot_sentiment_by_topic(Path(processed_data_path), output_path)
    
    logger.info(f"Visualizations saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate visualizations from analysis results")
    parser.add_argument(
        "--processed-data",
        type=str,
        default="../data/processed/processed_transcripts.csv",
        help="Path to processed transcript data"
    )
    parser.add_argument(
        "--desc-stats-path",
        type=str,
        default="../data/processed/descriptive_statistics.json",
        help="Path to descriptive statistics JSON file"
    )
    parser.add_argument(
        "--utterance-path",
        type=str,
        default="../data/processed/utterance_summary.json",
        help="Path to utterance summary JSON file"
    )
    parser.add_argument(
        "--ngram-path",
        type=str,
        default="../data/processed/ngram_analysis.json",
        help="Path to n-gram analysis JSON file"
    )
    parser.add_argument(
        "--grant-summary-path",
        type=str,
        default="../data/processed/grant_summary.json",
        help="Path to grant summary JSON file"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="../results/visualizations",
        help="Directory to save visualizations"
    )
    
    args = parser.parse_args()
    main(
        args.processed_data,
        args.desc_stats_path,
        args.utterance_path,
        args.ngram_path,
        args.grant_summary_path,
        args.output_dir
    ) 