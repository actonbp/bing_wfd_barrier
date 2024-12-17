#!/usr/bin/env python3
"""
Process Focus Group Transcripts

This script processes focus group transcripts and generates detailed analyses
including descriptive statistics, barrier analysis, n-gram analysis,
temporal patterns, and embedding-based themes.
"""

import argparse
from pathlib import Path
from focus_group_processor import FocusGroupProcessor
from temporal_theme_analyzer import TemporalThemeAnalyzer
import json
import logging
import pandas as pd

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def save_json(data: dict, path: Path, indent: int = 2):
    """Save dictionary as JSON file."""
    with open(path, 'w') as f:
        json.dump(data, f, indent=indent)

def main(data_dir: str, output_dir: str):
    """
    Process all focus group transcripts and save results.
    
    Args:
        data_dir: Directory containing focus group transcripts
        output_dir: Directory to save processed data and statistics
    """
    # Create output directory if it doesn't exist
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Processing transcripts from {data_dir}")
    
    # Initialize processors
    processor = FocusGroupProcessor(data_dir)
    theme_analyzer = TemporalThemeAnalyzer()
    
    # Process transcripts
    processor.process_all_transcripts()
    
    # Save processed data
    processed_data_path = output_path / "processed_transcripts.csv"
    processor.save_processed_data(processed_data_path)
    logger.info(f"Saved processed transcripts to {processed_data_path}")
    
    # Get and save descriptive statistics
    desc_stats = processor.get_descriptive_statistics()
    desc_stats_path = output_path / "descriptive_statistics.json"
    save_json(desc_stats, desc_stats_path)
    logger.info(f"Saved descriptive statistics to {desc_stats_path}")
    
    # Get and save speaker statistics
    stats = processor.get_speaker_statistics(processor.processed_data)
    stats_path = output_path / "speaker_statistics.json"
    save_json(stats, stats_path)
    logger.info(f"Saved speaker statistics to {stats_path}")
    
    # Get and save n-gram analysis
    ngrams = {
        'unigrams': processor.get_top_ngrams(n=1, top_k=30),
        'bigrams': processor.get_top_ngrams(n=2, top_k=30),
        'trigrams': processor.get_top_ngrams(n=3, top_k=30)
    }
    ngrams_path = output_path / "ngram_analysis.json"
    save_json(ngrams, ngrams_path)
    logger.info(f"Saved n-gram analysis to {ngrams_path}")
    
    # Perform temporal and theme analysis
    logger.info("Performing temporal and theme analysis...")
    theme_analyzer.process_transcript(processor.processed_data)
    
    # Generate and save themes
    themes = theme_analyzer.identify_themes()
    themes_path = output_path / "identified_themes.json"
    save_json(themes, themes_path)
    logger.info(f"Saved identified themes to {themes_path}")
    
    # Generate and save temporal patterns
    patterns = theme_analyzer.analyze_temporal_patterns()
    patterns_path = output_path / "temporal_patterns.json"
    save_json(patterns, patterns_path)
    logger.info(f"Saved temporal patterns to {patterns_path}")
    
    # Generate visualizations
    viz_dir = output_path / "visualizations"
    viz_dir.mkdir(exist_ok=True)
    
    # Generate temporal pattern visualizations
    theme_analyzer.plot_temporal_patterns(str(viz_dir / "temporal_patterns.png"))
    theme_analyzer.generate_theme_network(str(viz_dir / "theme_network.png"))
    logger.info(f"Saved temporal and theme visualizations to {viz_dir}")
    
    # Generate utterance type summary
    utterance_summary = {
        'total_utterances': len(processor.processed_data),
        'barrier_statements': processor.processed_data['is_barrier'].sum(),
        'solution_statements': processor.processed_data['is_solution'].sum(),
        'combined_statements': processor.processed_data['is_both'].sum(),
        'by_speaker': {
            speaker: {
                'barriers': group['is_barrier'].sum(),
                'solutions': group['is_solution'].sum(),
                'combined': group['is_both'].sum()
            }
            for speaker, group in processor.processed_data.groupby('Speaker')
        }
    }
    utterance_path = output_path / "utterance_summary.json"
    save_json(utterance_summary, utterance_path)
    logger.info(f"Saved utterance summary to {utterance_path}")
    
    # Save summary for grant figures
    grant_summary = {
        'overview': desc_stats['overview'],
        'barrier_distribution': desc_stats['content_analysis']['barrier_mentions'],
        'speaker_engagement': {
            'most_active_speaker': max(
                desc_stats['speaker_engagement']['turns_per_speaker'].items(),
                key=lambda x: x[1]
            )[0],
            'avg_turns_per_speaker': sum(desc_stats['speaker_engagement']['turns_per_speaker'].values()) / 
                                   len(desc_stats['speaker_engagement']['turns_per_speaker'])
        },
        'themes': {
            'total_themes': len(themes),
            'avg_utterances_per_theme': sum(t['utterance_count'] for t in themes.values()) / len(themes) if themes else 0,
            'theme_summaries': {
                theme_id: {
                    'representative_quote': info['representative_quote'],
                    'speaker_count': len(info['speakers'])
                }
                for theme_id, info in themes.items()
            }
        },
        'temporal_insights': {
            'avg_turn_duration': patterns['turn_taking']['avg_turn_duration'],
            'max_consecutive_turns': patterns['turn_taking']['max_consecutive_turns']
        }
    }
    grant_path = output_path / "grant_summary.json"
    save_json(grant_summary, grant_path)
    logger.info(f"Saved grant summary to {grant_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process focus group transcripts")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="../data/focus-groups",
        help="Directory containing focus group transcripts"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="../data/processed",
        help="Directory to save processed data"
    )
    
    args = parser.parse_args()
    main(args.data_dir, args.output_dir) 