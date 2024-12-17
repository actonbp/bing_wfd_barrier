#!/usr/bin/env python3
"""
Process Raw Focus Group Data

This script processes the raw focus group transcripts, cleaning and standardizing the data
for further analysis.
"""

import argparse
from pathlib import Path
import logging
from data_processor import DataProcessor
import json
import numpy as np

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy types."""
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                          np.int16, np.int32, np.int64, np.uint8,
                          np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def save_json(data: dict, path: Path, indent: int = 2):
    """Save dictionary as JSON file."""
    with open(path, 'w') as f:
        json.dump(data, f, indent=indent, cls=NumpyEncoder)

def main(raw_dir: str, output_dir: str):
    """
    Process all raw focus group transcripts.
    
    Args:
        raw_dir: Directory containing raw transcript files
        output_dir: Directory to save processed data
    """
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Processing transcripts from {raw_dir}")
    
    # Initialize and run processor
    processor = DataProcessor(raw_dir)
    processed_data = processor.process_all_sessions()
    
    # Save processed sessions
    processor.save_processed_sessions(output_path)
    logger.info(f"Saved processed sessions to {output_path}")
    
    # Generate and save session summaries
    summaries = processor.get_session_summaries()
    summaries.to_csv(output_path / "session_summaries.csv", index=False)
    logger.info("Saved session summaries")
    
    # Save overall statistics
    overall_stats = {
        'total_sessions': len(processed_data),
        'total_duration_minutes': float(summaries['total_duration_minutes'].sum()),
        'total_turns': int(summaries['total_turns'].sum()),
        'total_words': int(summaries['total_words'].sum()),
        'avg_speakers_per_session': float(summaries['unique_speakers'].mean()),
        'avg_duration_minutes': float(summaries['total_duration_minutes'].mean()),
        'avg_turns_per_session': float(summaries['total_turns'].mean())
    }
    
    save_json(overall_stats, output_path / "overall_statistics.json")
    logger.info("Saved overall statistics")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process raw focus group transcripts")
    parser.add_argument(
        "--raw-dir",
        type=str,
        default="data/raw",
        help="Directory containing raw transcripts"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/processed",
        help="Directory to save processed data"
    )
    
    args = parser.parse_args()
    main(args.raw_dir, args.output_dir) 