"""
Advanced NLP Analysis Module (Updated)

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
            'career': '#FF6B6B',      # Warm red for career terms
            'clinical': '#4ECDC4',     # Teal for clinical/health terms
            'education': '#45B7D1',    # Blue for education terms
            'barriers': '#96CEB4',     # Soft green for challenges/barriers
            'motivation': '#FFA07A'    # Light salmon orange for motivational factors
        }
        
        # Define topic-related terms with more specific categorization
        self.topic_terms = {
            'career': {
                'counselor', 'career', 'profession', 'job', 'position', 'salary', 'pay',
                'income', 'wage', 'employment', 'work', 'field', 'industry'
            },
            'clinical': {
                'substance', 'addiction', 'disorder', 'treatment', 'recovery', 'therapy',
                'mental', 'health', 'patient', 'clinical', 'counseling', 'therapeutic',
                'diagnosis', 'intervention', 'rehabilitation'
            },
            'education': {
                'degree', 'certification', 'license', 'training', 'education', 'program',
                'course', 'study', 'qualification', 'requirement', 'credential', 'exam',
                'school', 'university', 'college'
            },
            'barriers': {
                'challenge', 'barrier', 'difficulty', 'stress', 'burnout', 'workload',
                'cost', 'debt', 'time', 'balance', 'demanding', 'pressure', 'burden',
                'struggle', 'obstacle', 'problem', 'issue', 'concern'
            },
            'motivation': {
                'help', 'impact', 'difference', 'support', 'change', 'community',
                'passion', 'purpose', 'meaningful', 'rewarding', 'fulfilling',
                'contribute', 'service', 'care', 'dedication'
            }
        }

        # Additional words to remove beyond basic stop words
        self.additional_stops = {
            'guess', 'definitely', 'interesting', 'stuff', 'pretty', 'shes', 'trying',
            'different', 'especially', 'little', 'big', 'hard', 'feel', 'need',
            'want', 'make', 'take', 'day', 'way', 'thing', 'person', 'people',
            'important', 'supportive', 'working', 'interested', 'like', 'yeah',
            'just', 'know', 'think', 'going', 'really', 'would', 'could', 'much',
            'something', 'right', 'okay', 'well', 'kind', 'able', 'said', 'theres',
            'youre', 'dont', 'very', 'when', 'also', 'lot', 'good', 'time', 'get',
            'getting', 'got', 'actually', 'maybe', 'see', 'seeing', 'seen', 'look',
            'looking', 'looked', 'sure', 'come', 'coming', 'came', 'even', 'still',
            'back', 'mean', 'means', 'meant', 'say', 'says', 'said', 'saying',
            'thats', 'theyre', 'cant', 'wont', 'doesnt', 'didnt', 'isnt', 'arent'
        }
        
        # Initialize base stop words
        self.stop_words = self._initialize_stop_words()

    def _initialize_stop_words(self):
        """Initialize comprehensive stop words list."""
        # Start with basic stop words
        stops = {
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
            'once'
        }
        # Add additional domain-specific stops
        stops.update(self.additional_stops)
        return stops

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
        
        # Filter out low-frequency words and apply minimum length
        min_freq = 20  # Further increased minimum frequency
        word_freq = {
            word: freq for word, freq in word_freq.items() 
            if freq >= min_freq and len(word) > 3  # Filter out very short words
        }
        
        # Boost frequencies for domain-specific terms
        boost_multiplier = 2.5  # Increased boost for better contrast
        for category, terms in self.topic_terms.items():
            for word in terms:
                if word in word_freq:
                    word_freq[word] = int(word_freq[word] * boost_multiplier)
        
        # Create color function
        color_func = lambda word, *args, **kwargs: self.get_word_color(word)
        
        # Create word cloud with improved parameters
        wordcloud = WordCloud(
            width=2400,
            height=1600,
            background_color='white',
            max_words=20,          # Slightly reduced for even better focus
            min_font_size=32,      # Increased minimum font size
            max_font_size=500,     # Increased maximum font size
            color_func=color_func,
            prefer_horizontal=0.65, # Slightly more vertical words
            margin=6,              # Keep tight packing
            relative_scaling=1.0,   # Maximum scaling for better size differentiation
            collocations=False,    # Avoid bigrams in word cloud
            random_state=42
        ).generate_from_frequencies(word_freq)
        
        # Create figure with improved layout
        fig = plt.figure(figsize=(30, 15))
        
        # Create subplot with specific size ratios
        gs = plt.GridSpec(1, 20, figure=fig)
        ax = fig.add_subplot(gs[0, :18])
        
        # Plot word cloud
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        
        if title:
            ax.set_title(title, fontsize=24, pad=20)
        
        # Add color legend with improved categories
        legend_ax = fig.add_subplot(gs[0, 18:])
        legend_elements = [plt.Rectangle((0,0), 1, 1, facecolor=color, label=topic.title())
                         for topic, color in self.topic_colors.items()]
        legend = legend_ax.legend(handles=legend_elements,
                                title='Theme Categories',
                                title_fontsize=16,
                                fontsize=14,
                                loc='center',
                                bbox_to_anchor=(0.5, 0.5))
        legend_ax.axis('off')
        
        plt.tight_layout(pad=4)
        return fig

    def plot_top_bigrams(self, texts: list, title: str = None, min_count: int = 2) -> plt.Figure:
        """Create an improved visualization of top meaningful bigrams."""
        # Get bigrams with counts
        bigrams = self.find_collocations(texts, min_count)
        bigram_counts = Counter([' '.join(bg) for bg in bigrams])
        
        # Convert to DataFrame for plotting
        df = pd.DataFrame.from_dict(bigram_counts, orient='index', columns=['count'])
        df = df.sort_values('count', ascending=True).tail(12)  # Show top 12
        
        # Create color mapping based on topic categories
        colors = []
        for bigram in df.index:
            words = bigram.split()
            # Check first word in topics
            color = self.get_word_color(words[0])
            # If first word isn't in topics, try second word
            if color == '#666666':
                color = self.get_word_color(words[1])
            colors.append(color)
        
        # Create plot with improved dimensions
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
            ax.text(width + 0.1, bar.get_y() + bar.get_height()/2,
                   f'{int(width)}', ha='left', va='center', fontweight='bold')
        
        # Add titles and labels
        plt.title('Most Common Word Pairs by Frequency' if not title else title,
                 pad=20, fontsize=14)
        plt.xlabel('Frequency', fontsize=12)
        
        # Add grid for readability
        ax.grid(True, axis='x', linestyle='--', alpha=0.7)
        
        # Adjust layout
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
        """Find meaningful word pairs that frequently occur together."""
        # Get all words with better cleaning
        all_words = []
        for text in texts:
            # Clean and tokenize each text
            words = self.tokenize(self.clean_text(text))
            # Only keep meaningful words (longer than 2 chars and not in stop words)
            words = [w for w in words if len(w) > 2 and w not in self.stop_words]
            all_words.extend(words)
            
        # Find word pairs, excluding pairs with both words being filler
        word_pairs = []
        for i in range(len(all_words) - 1):
            w1, w2 = all_words[i], all_words[i+1]
            # Check if at least one word is from our topic terms
            is_meaningful = False
            for topic_terms in self.topic_terms.values():
                if w1 in topic_terms or w2 in topic_terms:
                    is_meaningful = True
                    break
            if is_meaningful:
                word_pairs.append((w1, w2))
                
        # Count frequencies
        pair_counts = Counter(word_pairs)
        
        # Filter by frequency and sort
        common_pairs = [(pair, count) for pair, count in pair_counts.items() 
                       if count >= min_count]
        common_pairs.sort(key=lambda x: x[1], reverse=True)
        
        return [pair for pair, _ in common_pairs]
        
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
        
    def analyze_response_similarities(self, texts: list) -> np.ndarray:
        """Compute pairwise similarities between responses."""
        embeddings = self.model.encode(texts)
        similarities = cosine_similarity(embeddings)
        return similarities 

    def analyze_thematic_relationships(self, texts: list, title: str = None) -> plt.Figure:
        """Analyze and visualize relationships between key themes in discussions."""
        # Process all texts
        all_words = []
        for text in texts:
            words = self.tokenize(self.clean_text(text))
            all_words.extend(words)

        # Create theme-based word groups
        theme_groups = {
            'Education Path': {'school', 'degree', 'program', 'training', 'certification', 'license'},
            'Career Benefits': {'salary', 'pay', 'benefits', 'career', 'job', 'opportunity'},
            'Clinical Practice': {'counseling', 'therapy', 'treatment', 'recovery', 'health', 'mental'},
            'Support Role': {'help', 'support', 'care', 'community', 'impact', 'difference'},
            'Challenges': {'barrier', 'difficult', 'challenge', 'stress', 'burnout', 'pressure'}
        }

        # Count co-occurrences within a window
        window_size = 5
        theme_connections = {theme: {other_theme: 0 for other_theme in theme_groups if other_theme != theme} 
                           for theme in theme_groups}
        
        # Analyze co-occurrences within windows
        for i in range(len(all_words)):
            window = all_words[max(0, i-window_size):min(len(all_words), i+window_size)]
            word_themes = []
            
            # Find themes present in window
            for theme, words in theme_groups.items():
                if any(word in words for word in window):
                    word_themes.append(theme)
            
            # Count theme co-occurrences
            for t1 in word_themes:
                for t2 in word_themes:
                    if t1 != t2:
                        theme_connections[t1][t2] += 1

        # Create visualization
        fig = plt.figure(figsize=(12, 8))
        
        # Calculate node positions (circular layout)
        n_themes = len(theme_groups)
        angles = np.linspace(0, 2*np.pi, n_themes, endpoint=False)
        radius = 0.4
        pos = {theme: (radius*np.cos(angle), radius*np.sin(angle)) 
               for theme, angle in zip(theme_groups.keys(), angles)}

        # Draw connections
        max_width = 3.0
        min_width = 0.5
        max_count = max(max(connections.values()) for connections in theme_connections.values())
        
        # Draw lines with varying thickness
        for theme1, connections in theme_connections.items():
            for theme2, count in connections.items():
                if count > 0:
                    x1, y1 = pos[theme1]
                    x2, y2 = pos[theme2]
                    width = min_width + (count/max_count)*(max_width - min_width)
                    alpha = 0.3 + 0.7*(count/max_count)
                    plt.plot([x1, x2], [y1, y2], 
                            color='gray', 
                            alpha=alpha,
                            linewidth=width,
                            zorder=1)

        # Draw nodes
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFA07A']
        for (theme, (x, y)), color in zip(pos.items(), colors):
            plt.scatter(x, y, s=1000, c=[color], alpha=0.6, zorder=2)
            plt.annotate(theme, (x, y), 
                        xytext=(0, 0), 
                        textcoords='offset points',
                        ha='center', va='center',
                        fontsize=11,
                        fontweight='bold')

        # Customize plot
        plt.title('Thematic Relationships in Discussions' if not title else title,
                 pad=20, fontsize=14)
        plt.axis('equal')
        plt.axis('off')
        
        # Add legend for connection strength
        legend_elements = [plt.Line2D([0], [0], color='gray', alpha=0.3, linewidth=min_width, label='Weak'),
                         plt.Line2D([0], [0], color='gray', alpha=0.7, linewidth=(max_width+min_width)/2, label='Medium'),
                         plt.Line2D([0], [0], color='gray', alpha=1.0, linewidth=max_width, label='Strong')]
        plt.legend(handles=legend_elements, 
                  title='Connection Strength',
                  loc='center left',
                  bbox_to_anchor=(1, 0.5))

        plt.tight_layout()
        return fig

    def discover_themes(self, texts: list, n_topics: int = 8) -> tuple:
        """Discover themes using BERT embeddings and clustering."""
        # Better preprocessing
        processed_texts = []
        for text in texts:
            # Clean text
            text = self.clean_text(text)
            # Filter out very short responses and common acknowledgments
            if len(text.split()) > 10 and not any(ack in text.lower() for ack in ['okay', 'gotcha', 'yup', 'yeah']):
                processed_texts.append(text)
        
        print(f"Processing {len(processed_texts)} substantive responses...")
        
        # Generate embeddings
        print("Generating embeddings for theme discovery...")
        embeddings = self.model.encode(processed_texts, show_progress_bar=False)
        
        # Reduce dimensionality for clustering
        umap_reducer = umap.UMAP(
            n_components=5,
            n_neighbors=30,  # Increased for better global structure
            min_dist=0.1,
            metric='cosine',
            random_state=42
        )
        reduced_embeddings = umap_reducer.fit_transform(embeddings)
        
        # Cluster to find topics
        clusterer = HDBSCAN(
            min_cluster_size=10,  # Increased for more robust clusters
            min_samples=5,
            metric='euclidean',
            cluster_selection_epsilon=0.5  # Added to be more inclusive
        )
        cluster_labels = clusterer.fit_predict(reduced_embeddings)
        
        # Get representative texts and keywords for each cluster
        themes = {}
        for cluster_id in set(cluster_labels):
            if cluster_id != -1:  # Skip noise
                # Get texts in this cluster
                cluster_texts = [text for text, label in zip(processed_texts, cluster_labels) 
                               if label == cluster_id]
                
                # Get all words in cluster texts
                cluster_words = []
                for text in cluster_texts:
                    # More aggressive word filtering
                    words = [w for w in self.tokenize(text) 
                           if len(w) > 3  # Filter very short words
                           and w not in self.additional_stops
                           and not w.isnumeric()]  # Filter numbers
                    cluster_words.extend(words)
                
                # Count word frequencies
                word_freq = Counter(cluster_words)
                
                # Get top keywords with better filtering
                keywords = []
                for word, count in word_freq.most_common(20):
                    # Check if word is in our topic terms
                    is_topic_term = False
                    for terms in self.topic_terms.values():
                        if word in terms:
                            is_topic_term = True
                            break
                    
                    if is_topic_term or count >= 3:  # Include if it's a topic term or frequent
                        keywords.append(word)
                        if len(keywords) == 5:  # Get top 5 keywords
                            break
                
                if keywords:  # Only include themes with meaningful keywords
                    # Find most representative text (closest to cluster center)
                    cluster_embeddings = self.model.encode(cluster_texts)
                    centroid = cluster_embeddings.mean(axis=0)
                    distances = np.linalg.norm(cluster_embeddings - centroid, axis=1)
                    representative_text = cluster_texts[distances.argmin()]
                    
                    themes[f"Theme {len(themes)+1}"] = {
                        'keywords': keywords,
                        'representative_text': representative_text,
                        'size': len(cluster_texts)
                    }
        
        return themes

    def visualize_discovered_themes(self, themes: dict) -> plt.Figure:
        """Create a visualization of discovered themes."""
        # Prepare data for visualization
        theme_names = list(themes.keys())
        theme_sizes = [theme['size'] for theme in themes.values()]
        total_responses = sum(theme_sizes)
        
        # Calculate percentages
        percentages = [size/total_responses * 100 for size in theme_sizes]
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8))
        
        # Create pie chart
        colors = plt.cm.Set3(np.linspace(0, 1, len(themes)))
        wedges, texts, autotexts = ax1.pie(percentages, 
                                         labels=theme_names,
                                         colors=colors,
                                         autopct='%1.1f%%',
                                         pctdistance=0.85)
        
        # Add center circle for donut chart
        centre_circle = plt.Circle((0,0), 0.70, fc='white')
        ax1.add_artist(centre_circle)
        
        # Add title to pie chart
        ax1.set_title('Distribution of Themes', pad=20)
        
        # Create keyword summary
        ax2.axis('off')
        summary_text = "Theme Keywords:\n\n"
        for theme_name, theme_info in themes.items():
            keywords_str = ", ".join(theme_info['keywords'])
            summary_text += f"{theme_name}:\n{keywords_str}\n\n"
        
        ax2.text(0, 0.5, summary_text, 
                verticalalignment='center',
                fontsize=10,
                fontfamily='monospace')
        
        plt.tight_layout()
        return fig

    def analyze_barriers(self, texts: list) -> dict:
        """Analyze specific barriers mentioned in discussions."""
        # Define barrier categories
        barrier_categories = {
            'education': {
                'cost', 'tuition', 'expensive', 'debt', 'loans', 'afford',
                'degree', 'certification', 'requirements', 'credits', 'classes',
                'training', 'education', 'school', 'university', 'college'
            },
            'work_life': {
                'stress', 'burnout', 'workload', 'hours', 'schedule', 'balance',
                'demanding', 'pressure', 'exhausting', 'overwhelming', 'difficult',
                'challenging', 'time', 'family', 'personal'
            },
            'emotional': {
                'emotional', 'draining', 'heavy', 'trauma', 'vicarious', 'burden',
                'tough', 'hard', 'intense', 'overwhelming', 'challenging',
                'difficult', 'stress', 'anxiety', 'depression'
            },
            'professional': {
                'salary', 'pay', 'money', 'income', 'compensation', 'benefits',
                'advancement', 'growth', 'opportunity', 'career', 'job', 'market',
                'demand', 'competition', 'experience'
            },
            'social': {
                'stigma', 'perception', 'reputation', 'respect', 'status',
                'recognition', 'support', 'isolation', 'community', 'family',
                'friends', 'social', 'network'
            }
        }

        # Initialize results structure
        results = {}
        
        # Process each category
        for category, terms in barrier_categories.items():
            # Initialize category results
            category_mentions = []
            category_contexts = []
            term_frequencies = Counter()
            
            # Process each text
            for text in texts:
                try:
                    # Clean and tokenize
                    cleaned_text = self.clean_text(text)
                    words = set(self.tokenize(cleaned_text))
                    
                    # Check for barrier terms
                    found_terms = words.intersection(terms)
                    if found_terms:
                        # Store the mention
                        category_mentions.append(text)
                        term_frequencies.update(found_terms)
                        
                        # Find relevant sentences
                        sentences = cleaned_text.split('.')
                        for sentence in sentences:
                            if any(term in sentence.lower() for term in found_terms):
                                category_contexts.append(sentence.strip())
                except Exception as e:
                    print(f"Error processing text: {str(e)}")
                    continue
            
            # Store results for this category
            results[category] = {
                'count': len(category_mentions),
                'frequency': (len(category_mentions) / len(texts)) * 100,
                'contexts': list(set(category_contexts))[:5],  # Top 5 unique contexts
                'common_terms': dict(term_frequencies.most_common(5))
            }
        
        return results

    def visualize_barriers(self, barrier_results: dict) -> plt.Figure:
        """Create visualization of barrier analysis results."""
        # Prepare data for plotting
        categories = list(barrier_results.keys())
        frequencies = [results['frequency'] for results in barrier_results.values()]
        
        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8))
        
        # Plot frequencies as bar chart
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFA07A']
        bars = ax1.bar(categories, frequencies, color=colors)
        
        # Customize first subplot
        ax1.set_title('Frequency of Barrier Categories', pad=20)
        ax1.set_xlabel('Barrier Category')
        ax1.set_ylabel('Percentage of Discussions')
        ax1.tick_params(axis='x', rotation=45)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}%',
                    ha='center', va='bottom')
        
        # Create summary text for second subplot
        ax2.axis('off')
        summary_text = "Barrier Analysis Summary:\n\n"
        for category, results in barrier_results.items():
            summary_text += f"\n{category.replace('_', ' ').title()}:\n"
            summary_text += f"- Mentioned in {results['count']} discussions\n"
            if results['common_terms']:
                summary_text += "- Common terms: " + ", ".join(results['common_terms'].keys()) + "\n"
            if results['contexts']:
                summary_text += f"- Example: '{results['contexts'][0][:100]}...'\n"
        
        ax2.text(0, 1.0, summary_text, 
                va='top', ha='left',
                fontsize=10,
                fontfamily='monospace',
                transform=ax2.transAxes)
        
        plt.tight_layout()
        return fig