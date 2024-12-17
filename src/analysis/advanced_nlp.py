"""
Advanced NLP Analysis Module

This module provides advanced text analysis capabilities using:
- BERT embeddings for semantic analysis
- UMAP for dimensionality reduction
- Topic modeling
- Sentiment analysis
- Response clustering
"""

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.cluster import HDBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import umap
from collections import Counter
import re

class AdvancedTextAnalyzer:
    def __init__(self, model_name='all-mpnet-base-v2'):
        """Initialize with specified embedding model."""
        self.model = SentenceTransformer(model_name)
        # Define topic categories for coloring
        self.topic_colors = {
            'career': '#FF6B6B',  # Warm red for career terms
            'health': '#4ECDC4',  # Teal for health terms
            'education': '#45B7D1',  # Blue for education terms
            'personal': '#96CEB4',  # Soft green for personal terms
            'support': '#FFEEAD',  # Soft yellow for support terms
        }
        
        # Define topic-related terms
        self.topic_terms = {
            'career': {'job', 'career', 'work', 'profession', 'field', 'counselor', 'position'},
            'health': {'mental', 'health', 'therapy', 'counseling', 'treatment', 'recovery'},
            'education': {'school', 'education', 'degree', 'training', 'certification', 'learn'},
            'personal': {'feel', 'life', 'experience', 'personal', 'family', 'friend'},
            'support': {'help', 'support', 'care', 'assist', 'guide', 'service'}
        }
        
        self.stop_words = {
            'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're",
            "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves',
            'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself',
            'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs',
            'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll",
            'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an',
            'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of',
            'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through',
            'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down',
            'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then',
            'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any',
            'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor',
            'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can',
            'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll',
            'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't",
            'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't",
            'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn',
            "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't",
            'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't",
            'like', 'yeah', 'think', 'know', 'going', 'really', 'want', 'something',
            'okay', 'well', 'got', 'ill', 'im', 'thats', 'would', 'could', 'maybe',
            'kind', 'lot', 'way', 'sure', 'right', 'mean', 'actually', 'probably',
            'thing', 'things', 'much', 'many', 'one', 'two', 'three', 'first', 'second',
            'get', 'getting', 'got', 'go', 'going', 'gone', 'say', 'said', 'saying',
            'says', 'just', 'thats', 'dont', 'doesnt', 'didnt', 'cant', 'couldnt',
            'wouldnt', 'wont', 'wasnt', 'werent', 'isnt', 'arent', 'youre', 'theyre',
            'ive', 'theyve', 'weve', 'youve'
        }

    def get_word_color(self, word):
        """Determine color for a word based on its topic category."""
        word = word.lower()
        for topic, terms in self.topic_terms.items():
            if word in terms:
                return self.topic_colors[topic]
        return '#666666'  # Default color for uncategorized words

    def create_enhanced_wordcloud(self, texts: list, title: str = None) -> plt.Figure:
        """Create an enhanced word cloud with better visual organization."""
        # Process texts and get word frequencies
        all_words = []
        for text in texts:
            words = self.tokenize(self.clean_text(text))
            all_words.extend(words)
        
        word_freq = Counter(all_words)
        
        # Create color function
        color_func = lambda word, *args, **kwargs: self.get_word_color(word)
        
        # Create word cloud with improved parameters
        wordcloud = WordCloud(
            width=2000,
            height=1200,
            background_color='white',
            max_words=100,
            min_font_size=12,
            max_font_size=150,
            color_func=color_func,
            prefer_horizontal=0.7,
            margin=20,
            relative_scaling=0.5,
            collocations=True
        ).generate_from_frequencies(word_freq)
        
        # Create figure with improved layout
        fig = plt.figure(figsize=(24, 12))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        
        if title:
            plt.title(title, fontsize=20, pad=20)
            
        # Add color legend
        legend_elements = [plt.Rectangle((0,0),1,1, facecolor=color, label=topic.title())
                         for topic, color in self.topic_colors.items()]
        plt.legend(handles=legend_elements, loc='center left', 
                  bbox_to_anchor=(1, 0.5), title='Topic Categories')
        
        plt.tight_layout(pad=3)
        return fig

    def plot_top_bigrams(self, texts: list, title: str = None, min_count: int = 3) -> plt.Figure:
        """Create an improved visualization of top bigrams."""
        # Get bigrams with counts
        bigrams = self.find_collocations(texts, min_count)
        bigram_counts = Counter([' '.join(bg) for bg in bigrams])
        
        # Convert to DataFrame for plotting
        df = pd.DataFrame.from_dict(bigram_counts, orient='index', columns=['count'])
        df = df.sort_values('count', ascending=True).tail(15)  # Show top 15
        
        # Create color mapping based on first word of bigram
        colors = [self.get_word_color(bigram.split()[0]) for bigram in df.index]
        
        # Create plot
        fig = plt.figure(figsize=(12, 8))
        ax = plt.gca()
        
        # Plot horizontal bars
        bars = ax.barh(range(len(df)), df['count'], color=colors)
        
        # Customize appearance
        ax.set_yticks(range(len(df)))
        ax.set_yticklabels(df.index)
        ax.invert_yaxis()
        
        # Add value labels
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax.text(width, bar.get_y() + bar.get_height()/2,
                   f'{int(width)}', ha='left', va='center', fontweight='bold')
        
        # Add titles and labels
        plt.title('Most Frequent Word Pairs in Discussions' if not title else title,
                 pad=20, fontsize=14)
        plt.xlabel('Frequency', fontsize=12)
        
        # Add grid for readability
        ax.grid(True, axis='x', linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        return fig

    def plot_semantic_clusters(self, texts: list, title: str = None) -> plt.Figure:
        """Create improved semantic clustering visualization."""
        # Generate embeddings
        embeddings = self.model.encode(texts)
        
        # Reduce dimensionality
        reducer = umap.UMAP(n_components=2, random_state=42)
        embedding_2d = reducer.fit_transform(embeddings)
        
        # Perform clustering
        clusterer = hdbscan.HDBSCAN(min_cluster_size=5, min_samples=3)
        clusters = clusterer.fit_predict(embedding_2d)
        
        # Create plot
        fig = plt.figure(figsize=(15, 10))
        
        # Create scatter plot with improved aesthetics
        scatter = plt.scatter(
            embedding_2d[:, 0],
            embedding_2d[:, 1],
            c=clusters,
            cmap='viridis',
            alpha=0.6,
            s=100
        )
        
        # Add cluster labels
        cluster_centers = {}
        for cluster_id in set(clusters):
            if cluster_id != -1:  # Skip noise points
                mask = clusters == cluster_id
                center_x = embedding_2d[mask, 0].mean()
                center_y = embedding_2d[mask, 1].mean()
                cluster_centers[cluster_id] = (center_x, center_y)
                plt.annotate(f'Cluster {cluster_id}',
                           (center_x, center_y),
                           xytext=(5, 5),
                           textcoords='offset points',
                           fontsize=12,
                           bbox=dict(facecolor='white', alpha=0.7))
        
        # Customize plot
        plt.title('Semantic Clustering of Responses\nSimilar responses appear closer together',
                 pad=20, fontsize=14)
        plt.xlabel('First Semantic Dimension', fontsize=12)
        plt.ylabel('Second Semantic Dimension', fontsize=12)
        
        # Add colorbar
        cbar = plt.colorbar(scatter)
        cbar.set_label('Cluster Assignment', fontsize=12)
        
        # Add grid
        plt.grid(True, linestyle='--', alpha=0.3)
        
        plt.tight_layout()
        return fig

    def clean_text(self, text: str) -> str:
        """Clean text for analysis."""
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters but keep sentence structure
        text = re.sub(r'[^a-z\s.,!?]', '', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
        
    def tokenize(self, text: str) -> list:
        """Simple word tokenization."""
        # Split on whitespace and punctuation
        words = re.findall(r'\b\w+\b', text.lower())
        return [w for w in words if w not in self.stop_words]
        
    def find_collocations(self, texts: list, min_count: int = 3) -> list:
        """Find word pairs that frequently occur together."""
        # Get all words
        all_words = []
        for text in texts:
            words = self.tokenize(self.clean_text(text))
            all_words.extend(words)
            
        # Find word pairs
        word_pairs = []
        for i in range(len(all_words) - 1):
            if all_words[i] and all_words[i+1]:  # Skip empty strings
                word_pairs.append((all_words[i], all_words[i+1]))
                
        # Count frequencies
        pair_counts = Counter(word_pairs)
        
        # Filter by frequency
        common_pairs = [(pair, count) for pair, count in pair_counts.items() 
                       if count >= min_count]
        
        # Sort by frequency
        common_pairs.sort(key=lambda x: x[1], reverse=True)
        
        return [pair for pair, _ in common_pairs[:20]]  # Return top 20 pairs
        
    def cluster_responses(self, texts: list, n_clusters: int = None) -> tuple:
        """Cluster responses using BERT embeddings and HDBSCAN."""
        # Generate embeddings
        print("Generating embeddings...")
        embeddings = self.model.encode(texts, show_progress_bar=True)
        
        # First UMAP reduction to 5 dimensions
        print("Performing dimensionality reduction...")
        umap_reducer = umap.UMAP(
            n_components=5,
            n_neighbors=15,
            min_dist=0.1,
            metric='cosine',
            random_state=42
        )
        embeddings_reduced = umap_reducer.fit_transform(embeddings)
        
        # Perform clustering
        print("Clustering responses...")
        clusterer = HDBSCAN(
            min_cluster_size=max(5, len(texts) // 20),  # Adaptive cluster size
            min_samples=3,
            metric='euclidean'
        )
        clusters = clusterer.fit_predict(embeddings_reduced)
        
        # Find representative texts for each cluster
        representative_texts = []
        unique_clusters = sorted(set(clusters))
        unique_clusters = [c for c in unique_clusters if c != -1]  # Remove noise cluster
        
        for cluster_id in unique_clusters:
            cluster_texts = [t for t, c in zip(texts, clusters) if c == cluster_id]
            if cluster_texts:
                # Find most central text in cluster
                cluster_embeddings = self.model.encode(cluster_texts)
                centroid = cluster_embeddings.mean(axis=0)
                distances = np.linalg.norm(cluster_embeddings - centroid, axis=1)
                representative_texts.append(cluster_texts[distances.argmin()])
            else:
                representative_texts.append("")
                
        return clusters, representative_texts
        
    def analyze_semantic_evolution(self, texts: list, timestamps: list) -> pd.DataFrame:
        """Analyze how topics evolve over time using improved embedding and clustering."""
        # Generate BERT embeddings
        print("Generating embeddings...")
        embeddings = self.model.encode(texts, show_progress_bar=True)
        
        # First UMAP reduction to 5 dimensions
        print("Performing initial dimensionality reduction...")
        umap_reducer_5d = umap.UMAP(
            n_components=5,
            n_neighbors=15,
            min_dist=0.1,
            metric='cosine',
            random_state=42
        )
        embeddings_5d = umap_reducer_5d.fit_transform(embeddings)
        
        # Perform clustering in 5D space
        print("Clustering in 5D space...")
        clusterer = HDBSCAN(
            min_cluster_size=5,
            min_samples=3,
            cluster_selection_epsilon=0.3,
            metric='euclidean'
        )
        clusters = clusterer.fit_predict(embeddings_5d)
        
        # Final reduction to 2D for visualization
        print("Reducing to 2D for visualization...")
        pca = PCA(n_components=2)
        embedding_2d = pca.fit_transform(embeddings_5d)
        
        # Create DataFrame with results
        df = pd.DataFrame({
            'x': embedding_2d[:, 0],
            'y': embedding_2d[:, 1],
            'cluster': clusters,
            'timestamp': timestamps,
            'text': texts
        })
        
        return df

    def plot_semantic_evolution(self, df: pd.DataFrame, title: str = "Evolution of Discussion Topics") -> plt.Figure:
        """Create an enhanced visualization of semantic evolution."""
        # Set up the plot
        plt.figure(figsize=(15, 10))
        
        # Create a custom colormap for clusters
        n_clusters = len(set(df['cluster'])) - (1 if -1 in df['cluster'] else 0)
        colors = plt.cm.viridis(np.linspace(0, 1, n_clusters))
        
        # Plot points
        scatter = plt.scatter(
            df['x'], df['y'],
            c=df['timestamp'],
            cmap='viridis',
            alpha=0.6,
            s=100
        )
        
        # Add cluster labels and boundaries
        for cluster_id in sorted(set(df['cluster'])):
            if cluster_id != -1:  # Skip noise points
                mask = df['cluster'] == cluster_id
                cluster_points = df[mask]
                
                # Calculate cluster center
                center_x = cluster_points['x'].mean()
                center_y = cluster_points['y'].mean()
                
                # Add cluster label
                plt.annotate(
                    f'Topic {cluster_id + 1}',
                    (center_x, center_y),
                    xytext=(5, 5),
                    textcoords='offset points',
                    bbox=dict(
                        facecolor='white',
                        alpha=0.7,
                        edgecolor='none',
                        pad=1
                    ),
                    fontsize=12,
                    fontweight='bold'
                )
        
        # Customize plot
        plt.title(title, fontsize=16, pad=20)
        plt.xlabel('First Principal Component', fontsize=12)
        plt.ylabel('Second Principal Component', fontsize=12)
        
        # Add colorbar for time progression
        cbar = plt.colorbar(scatter)
        cbar.set_label('Time in Session', fontsize=12)
        
        # Add grid and style
        plt.grid(True, linestyle='--', alpha=0.3)
        plt.tight_layout()
        
        return plt.gcf()
        
    def plot_semantic_space(self, texts: list, title: str = None) -> plt.Figure:
        """Plot texts in 2D semantic space."""
        # Generate embeddings
        embeddings = self.model.encode(texts)
        
        # Reduce dimensionality
        reducer = umap.UMAP(n_components=2, random_state=42)
        embedding_2d = reducer.fit_transform(embeddings)
        
        # Cluster points
        clusterer = HDBSCAN(
            min_cluster_size=5,
            min_samples=3,
            cluster_selection_epsilon=0.3,
            metric='euclidean'
        )
        clusters = clusterer.fit_predict(embedding_2d)
        
        # Create plot
        fig = plt.figure(figsize=(12, 8))
        scatter = plt.scatter(
            embedding_2d[:, 0],
            embedding_2d[:, 1],
            c=clusters,
            cmap='viridis',
            alpha=0.6,
            s=100
        )
        plt.colorbar(scatter)
        
        if title:
            plt.title(title, fontsize=14, pad=20)
            
        plt.xlabel('First UMAP Dimension', fontsize=12)
        plt.ylabel('Second UMAP Dimension', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.3)
        plt.tight_layout()
            
        return fig
        
    def analyze_response_similarities(self, texts: list) -> np.ndarray:
        """Compute pairwise similarities between responses."""
        embeddings = self.model.encode(texts)
        similarities = cosine_similarity(embeddings)
        return similarities