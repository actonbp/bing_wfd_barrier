#!/usr/bin/env python3
"""
Topic Modeling Analysis

This script performs topic modeling on processed focus group transcripts
using Latent Dirichlet Allocation (LDA) from gensim.
"""

import argparse
import logging
from pathlib import Path
import pandas as pd
import numpy as np
from gensim import corpora, models
from gensim.utils import simple_preprocess
import nltk
from nltk.corpus import stopwords
import json

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_data(input_path: Path) -> pd.DataFrame:
    """Load processed transcript data."""
    return pd.read_csv(input_path)

def preprocess_text(text: str, stop_words: set) -> list:
    """
    Preprocess text for topic modeling.
    
    Args:
        text: Input text string
        stop_words: Set of stop words to remove
        
    Returns:
        List of preprocessed tokens
    """
    # Tokenize and remove stop words
    tokens = simple_preprocess(text)
    return [token for token in tokens if token not in stop_words]

def create_corpus(texts: list) -> tuple:
    """
    Create gensim dictionary and corpus.
    
    Args:
        texts: List of preprocessed documents
        
    Returns:
        Tuple of (dictionary, corpus)
    """
    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]
    return dictionary, corpus

def train_lda_model(corpus: list, dictionary: corpora.Dictionary, num_topics: int) -> models.LdaModel:
    """
    Train LDA topic model.
    
    Args:
        corpus: Document-term matrix
        dictionary: Gensim dictionary
        num_topics: Number of topics to extract
        
    Returns:
        Trained LDA model
    """
    return models.LdaModel(
        corpus=corpus,
        id2word=dictionary,
        num_topics=num_topics,
        random_state=42,
        update_every=1,
        chunksize=100,
        passes=10,
        alpha='auto',
        per_word_topics=True
    )

def save_topics(model: models.LdaModel, dictionary: corpora.Dictionary, output_path: Path):
    """Save topic model results."""
    # Get topics with their top words and weights
    topics = {}
    for idx, topic in model.show_topics(formatted=False, num_words=10):
        topics[f"Topic_{idx}"] = [
            {"word": word, "weight": weight}
            for word, weight in topic
        ]
    
    # Save to JSON
    with open(output_path / "topics.json", 'w') as f:
        json.dump(topics, f, indent=2)

def main(input_path: str, output_dir: str, num_topics: int):
    """
    Run topic modeling analysis on processed transcripts.
    
    Args:
        input_path: Path to processed transcript data
        output_dir: Directory to save results
        num_topics: Number of topics to extract
    """
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Download required NLTK data
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')
    
    # Load stop words
    stop_words = set(stopwords.words('english'))
    
    # Load and preprocess data
    logger.info("Loading and preprocessing data...")
    df = load_data(input_path)
    texts = [preprocess_text(text, stop_words) for text in df['cleaned_text']]
    
    # Create corpus
    logger.info("Creating document corpus...")
    dictionary, corpus = create_corpus(texts)
    
    # Train model
    logger.info(f"Training LDA model with {num_topics} topics...")
    lda_model = train_lda_model(corpus, dictionary, num_topics)
    
    # Save results
    logger.info("Saving results...")
    save_topics(lda_model, dictionary, output_path)
    
    # Save model for later use
    model_path = output_path / "lda_model"
    lda_model.save(str(model_path))
    logger.info(f"Model saved to {model_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Perform topic modeling on focus group transcripts")
    parser.add_argument(
        "--input-path",
        type=str,
        default="../data/processed/processed_transcripts.csv",
        help="Path to processed transcript data"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="../results/topic_modeling",
        help="Directory to save results"
    )
    parser.add_argument(
        "--num-topics",
        type=int,
        default=5,
        help="Number of topics to extract"
    )
    
    args = parser.parse_args()
    main(args.input_path, args.output_dir, args.num_topics) 