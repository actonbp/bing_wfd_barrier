import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
from pathlib import Path
import json

def create_interactive_barrier_overview(results):
    """Create interactive overview visualization of barrier categories."""
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
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("Frequency of Barrier Categories in Discussions",
                       "",  # Empty title for better layout
                       "Mentions vs Unique Contexts",
                       "Distribution of Barrier Types"),
        specs=[[{"colspan": 2}, None],
               [{"type": "bar"}, {"type": "pie"}]],
        vertical_spacing=0.12,
        horizontal_spacing=0.1
    )
    
    # 1. Bar chart of frequencies
    fig.add_trace(
        go.Bar(
            x=categories,
            y=frequencies,
            text=[f'{freq:.1f}%' for freq in frequencies],
            textposition='auto',
            name='Frequency',
            hovertemplate='Category: %{x}<br>Frequency: %{y:.1f}%<extra></extra>'
        ),
        row=1, col=1
    )
    
    # 2. Comparison of mentions vs contexts
    fig.add_trace(
        go.Bar(
            x=categories,
            y=mentions,
            name='Total Mentions',
            hovertemplate='Category: %{x}<br>Mentions: %{y}<extra></extra>'
        ),
        row=2, col=1
    )
    
    fig.add_trace(
        go.Bar(
            x=categories,
            y=contexts,
            name='Unique Contexts',
            hovertemplate='Category: %{x}<br>Contexts: %{y}<extra></extra>'
        ),
        row=2, col=1
    )
    
    # 3. Pie chart of relative frequencies
    fig.add_trace(
        go.Pie(
            labels=categories,
            values=frequencies,
            textinfo='label+percent',
            hovertemplate='Category: %{label}<br>Frequency: %{value:.1f}%<extra></extra>'
        ),
        row=2, col=2
    )
    
    # Update layout
    fig.update_layout(
        height=800,
        showlegend=True,
        title_text="Barrier Analysis Overview",
        title_x=0.5,
        barmode='group'
    )
    
    # Update axes
    fig.update_xaxes(tickangle=45)
    fig.update_yaxes(title_text="Percentage of Discussions", row=1, col=1)
    fig.update_yaxes(title_text="Count", row=2, col=1)
    
    return fig

def create_interactive_term_frequencies(results):
    """Create interactive visualization of term frequencies."""
    # Prepare data for all categories
    all_data = []
    for category, data in results.items():
        for term, freq in data['term_frequencies'].items():
            all_data.append({
                'Category': category,
                'Term': term,
                'Frequency': freq
            })
    
    df = pd.DataFrame(all_data)
    
    # Create figure
    fig = px.bar(
        df,
        x='Frequency',
        y='Term',
        color='Category',
        facet_col='Category',
        facet_col_wrap=2,
        height=1000,
        title='Term Frequencies by Barrier Category',
        labels={'Frequency': 'Number of Mentions', 'Term': ''},
        hover_data={'Category': False}  # Hide category in hover as it's redundant
    )
    
    # Update layout
    fig.update_layout(
        showlegend=False,
        title_x=0.5,
    )
    
    # Update yaxis for each subplot to show all terms
    for annotation in fig.layout.annotations:
        annotation.text = annotation.text.split("=")[1]  # Remove "Category=" prefix
    
    return fig

def create_interactive_network(results):
    """Create interactive network visualization."""
    # Calculate node positions (circular layout)
    n_categories = len(results)
    angles = np.linspace(0, 2*np.pi, n_categories, endpoint=False)
    radius = 1
    
    # Create node positions
    node_x = radius * np.cos(angles)
    node_y = radius * np.sin(angles)
    
    # Create edges
    edge_x = []
    edge_y = []
    edge_weights = []
    edge_texts = []
    categories = list(results.keys())
    
    for i, cat1 in enumerate(categories):
        contexts1 = set(results[cat1]['context_groups'].keys())
        for j, cat2 in enumerate(categories[i+1:], i+1):
            contexts2 = set(results[cat2]['context_groups'].keys())
            shared = len(contexts1.intersection(contexts2))
            if shared > 0:
                edge_x.extend([node_x[i], node_x[j], None])
                edge_y.extend([node_y[i], node_y[j], None])
                edge_weights.append(shared)
                edge_texts.append(f'{cat1} ↔ {cat2}<br>Shared contexts: {shared}')
    
    # Create figure
    fig = go.Figure()
    
    # Add edges
    max_weight = max(edge_weights) if edge_weights else 1
    edge_weights_normalized = [w/max_weight * 5 for w in edge_weights]
    
    fig.add_trace(go.Scatter(
        x=edge_x,
        y=edge_y,
        mode='lines',
        line=dict(width=edge_weights_normalized, color='gray'),
        hoverinfo='text',
        text=edge_texts,
        name='Connections'
    ))
    
    # Add nodes
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFA07A']
    
    fig.add_trace(go.Scatter(
        x=node_x,
        y=node_y,
        mode='markers+text',
        marker=dict(
            size=40,
            color=colors[:n_categories],
            line=dict(width=2, color='white')
        ),
        text=categories,
        textposition='middle center',
        hoverinfo='text',
        hovertext=[f'{cat}<br>Total mentions: {results[cat]["total_mentions"]}' 
                  for cat in categories],
        name='Categories'
    ))
    
    # Update layout
    fig.update_layout(
        title='Barrier Category Relationships',
        showlegend=False,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        plot_bgcolor='white',
        height=600,
        width=800
    )
    
    return fig

def load_barrier_results():
    """Load barrier analysis results."""
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
    
    # Create interactive visualizations
    print("Creating interactive barrier overview...")
    overview_fig = create_interactive_barrier_overview(results)
    overview_fig.write_html('barrier_overview_interactive.html')
    
    print("Creating interactive term frequencies visualization...")
    terms_fig = create_interactive_term_frequencies(results)
    terms_fig.write_html('barrier_terms_interactive.html')
    
    print("Creating interactive network visualization...")
    network_fig = create_interactive_network(results)
    network_fig.write_html('barrier_network_interactive.html')
    
    print("Interactive visualizations saved!") 