import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def create_barrier_overview(results):
    """Create overview visualization of barrier categories."""
    # Prepare data
    categories = []
    frequencies = []
    mentions = []
    contexts = []
    
    for category, data in results.items():
        categories.append(category)
        frequencies.append(data['frequency'])
        mentions.append(data['total_mentions'])
        contexts.append(data['unique_contexts'])
    
    # Create figure with subplots
    fig = plt.figure(figsize=(15, 10))
    gs = plt.GridSpec(2, 2, figure=fig)
    
    # 1. Bar chart of frequencies
    ax1 = fig.add_subplot(gs[0, :])
    bars = ax1.bar(categories, frequencies)
    ax1.set_title('Frequency of Barrier Categories in Discussions', pad=20)
    ax1.set_ylabel('Percentage of Discussions')
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%',
                ha='center', va='bottom')
    
    # Rotate x-labels for better readability
    plt.xticks(rotation=45, ha='right')
    
    # 2. Comparison of mentions vs contexts
    ax2 = fig.add_subplot(gs[1, 0])
    x = np.arange(len(categories))
    width = 0.35
    
    ax2.bar(x - width/2, mentions, width, label='Total Mentions')
    ax2.bar(x + width/2, contexts, width, label='Unique Contexts')
    ax2.set_title('Mentions vs Unique Contexts')
    ax2.set_xticks(x)
    ax2.set_xticklabels(categories, rotation=45, ha='right')
    ax2.legend()
    
    # 3. Pie chart of relative frequencies
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.pie(frequencies, labels=categories, autopct='%1.1f%%')
    ax3.set_title('Distribution of Barrier Types')
    
    plt.tight_layout()
    return fig

def create_term_frequency_plots(results):
    """Create visualizations of term frequencies within each category."""
    # Create figure
    fig = plt.figure(figsize=(15, 12))
    
    # Calculate number of subplots needed
    n_categories = len(results)
    n_rows = (n_categories + 1) // 2  # Round up to nearest 2
    
    # Create subplots for each category
    for i, (category, data) in enumerate(results.items(), 1):
        ax = fig.add_subplot(n_rows, 2, i)
        
        # Get term frequencies
        terms = list(data['term_frequencies'].keys())
        freqs = list(data['term_frequencies'].values())
        
        # Sort by frequency
        sorted_indices = np.argsort(freqs)
        terms = [terms[i] for i in sorted_indices]
        freqs = [freqs[i] for i in sorted_indices]
        
        # Create horizontal bar chart
        bars = ax.barh(terms, freqs)
        
        # Add value labels
        for bar in bars:
            width = bar.get_width()
            ax.text(width, bar.get_y() + bar.get_height()/2,
                   f'{int(width)}',
                   ha='left', va='center')
        
        ax.set_title(f'{category} Terms')
        ax.set_xlabel('Frequency')
    
    plt.tight_layout()
    return fig

def create_context_network(results):
    """Create a network visualization showing relationships between barrier categories."""
    # Create figure
    fig = plt.figure(figsize=(12, 8))
    
    # Calculate node positions (circular layout)
    n_categories = len(results)
    angles = np.linspace(0, 2*np.pi, n_categories, endpoint=False)
    radius = 0.4
    pos = {cat: (radius*np.cos(angle), radius*np.sin(angle)) 
           for cat, angle in zip(results.keys(), angles)}
    
    # Calculate edge weights based on shared contexts
    edges = []
    for cat1 in results:
        contexts1 = set(results[cat1]['context_groups'].keys())
        for cat2 in results:
            if cat1 < cat2:  # Avoid duplicates
                contexts2 = set(results[cat2]['context_groups'].keys())
                shared = len(contexts1.intersection(contexts2))
                if shared > 0:
                    edges.append((cat1, cat2, shared))
    
    # Draw edges
    max_weight = max(w for _, _, w in edges) if edges else 1
    for cat1, cat2, weight in edges:
        x1, y1 = pos[cat1]
        x2, y2 = pos[cat2]
        width = weight / max_weight * 3  # Scale line width
        alpha = 0.3 + 0.7 * (weight / max_weight)  # Scale transparency
        plt.plot([x1, x2], [y1, y2], 
                color='gray', 
                alpha=alpha,
                linewidth=width)
    
    # Draw nodes
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFA07A']
    for (cat, (x, y)), color in zip(pos.items(), colors):
        plt.scatter(x, y, s=1000, c=[color], alpha=0.6)
        plt.annotate(cat, (x, y), 
                    xytext=(0, 0), 
                    textcoords='offset points',
                    ha='center', va='center',
                    fontsize=11,
                    fontweight='bold')
    
    plt.title('Relationships Between Barrier Categories')
    plt.axis('equal')
    plt.axis('off')
    
    # Add legend for edge weights
    legend_elements = [
        plt.Line2D([0], [0], color='gray', alpha=0.3, linewidth=1, label='Weak'),
        plt.Line2D([0], [0], color='gray', alpha=0.7, linewidth=2, label='Medium'),
        plt.Line2D([0], [0], color='gray', alpha=1.0, linewidth=3, label='Strong')
    ]
    plt.legend(handles=legend_elements, 
              title='Connection Strength',
              loc='center left',
              bbox_to_anchor=(1, 0.5))
    
    return fig

# Load and process the barrier analysis results
def load_barrier_results():
    """Simulate loading barrier analysis results."""
    return {
        'Educational': {
            'total_mentions': 88,
            'unique_contexts': 23,
            'frequency': 20.0,
            'term_frequencies': {
                'school': 38,
                'money': 22,
                'years': 11,
                'financial': 8,
                'college': 6,
                'degree': 4,
                'debt': 3,
                'classes': 3,
                'education': 3,
                'expensive': 2
            },
            'context_groups': {
                'money stress': ['financial stress while in school'],
                'time education': ['balancing education with work'],
                'support education': ['need support during education']
            }
        },
        'Work-Life Balance': {
            'total_mentions': 158,
            'unique_contexts': 23,
            'frequency': 35.8,
            'term_frequencies': {
                'time': 47,
                'personal': 35,
                'family': 34,
                'life': 25,
                'stress': 6,
                'relationship': 4,
                'pressure': 4,
                'children': 3,
                'workload': 2,
                'balance': 2
            },
            'context_groups': {
                'time stress': ['managing time and stress'],
                'family support': ['family support needs'],
                'money time': ['working while studying']
            }
        },
        'Emotional/Mental': {
            'total_mentions': 140,
            'unique_contexts': 22,
            'frequency': 31.7,
            'term_frequencies': {
                'mental': 54,
                'health': 49,
                'support': 47,
                'emotional': 9,
                'trauma': 8,
                'stress': 6,
                'difficult': 6,
                'anxiety': 2,
                'manage': 2,
                'heavy': 2
            },
            'context_groups': {
                'stress support': ['need emotional support'],
                'mental time': ['mental health and time management'],
                'emotional support': ['emotional support systems']
            }
        },
        'Professional/Career': {
            'total_mentions': 94,
            'unique_contexts': 18,
            'frequency': 21.3,
            'term_frequencies': {
                'career': 36,
                'experience': 20,
                'future': 14,
                'pay': 10,
                'salary': 8,
                'income': 6,
                'level': 6,
                'growth': 2,
                'opportunity': 2,
                'demand': 1
            },
            'context_groups': {
                'career support': ['career development support'],
                'money career': ['financial aspects of career'],
                'time career': ['career progression timeline']
            }
        },
        'Social/Support': {
            'total_mentions': 173,
            'unique_contexts': 14,
            'frequency': 39.2,
            'term_frequencies': {
                'help': 117,
                'support': 47,
                'connection': 6,
                'relationship': 4,
                'resources': 4,
                'guidance': 4,
                'respect': 2,
                'colleagues': 1,
                'stigma': 1
            },
            'context_groups': {
                'support stress': ['support systems for stress'],
                'emotional support': ['emotional support needs'],
                'family support': ['family and social support']
            }
        }
    }

if __name__ == "__main__":
    # Load results
    results = load_barrier_results()
    
    # Create visualizations
    print("Creating barrier overview visualization...")
    overview_fig = create_barrier_overview(results)
    overview_fig.savefig('barrier_overview.png', dpi=300, bbox_inches='tight')
    
    print("Creating term frequency visualization...")
    terms_fig = create_term_frequency_plots(results)
    terms_fig.savefig('barrier_terms.png', dpi=300, bbox_inches='tight')
    
    print("Creating context network visualization...")
    network_fig = create_context_network(results)
    network_fig.savefig('barrier_network.png', dpi=300, bbox_inches='tight')
    
    print("Visualizations saved!") 