"""
Module for analyzing barriers in focus group discussions
"""

from collections import Counter
import matplotlib.pyplot as plt
import re

class BarrierAnalyzer:
    def __init__(self):
        """Initialize barrier categories and analysis tools."""
        self.barrier_categories = {
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

    def clean_text(self, text: str) -> str:
        """Clean text for analysis."""
        if not isinstance(text, str):
            return ""
        # Convert to lowercase
        text = text.lower()
        # Remove special characters but keep sentence structure
        text = re.sub(r'[^a-z\s.,!?]', '', text)
        # Remove extra whitespace
        text = ' '.join(text.split())
        return text

    def tokenize(self, text: str) -> set:
        """Tokenize text into words."""
        return set(re.findall(r'\b\w+\b', text.lower()))

    def analyze(self, texts: list) -> dict:
        """Analyze barriers in texts."""
        results = {}
        
        for category, terms in self.barrier_categories.items():
            mentions = []
            contexts = []
            term_freq = Counter()
            
            for text in texts:
                try:
                    # Clean and tokenize
                    cleaned = self.clean_text(text)
                    words = self.tokenize(cleaned)
                    
                    # Find matching terms
                    matches = words.intersection(terms)
                    if matches:
                        mentions.append(text)
                        term_freq.update(matches)
                        
                        # Get context
                        for sentence in cleaned.split('.'):
                            if any(term in sentence for term in matches):
                                contexts.append(sentence.strip())
                except:
                    continue
            
            results[category] = {
                'count': len(mentions),
                'frequency': len(mentions) / len(texts) * 100,
                'contexts': list(set(contexts))[:5],
                'common_terms': dict(term_freq.most_common(5))
            }
        
        return results

    def visualize(self, results: dict) -> plt.Figure:
        """Create visualization of barrier analysis."""
        # Prepare data
        categories = list(results.keys())
        frequencies = [r['frequency'] for r in results.values()]
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8))
        
        # Bar chart
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFA07A']
        bars = ax1.bar(categories, frequencies, color=colors)
        
        # Customize bar chart
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
        
        # Summary text
        ax2.axis('off')
        summary = "Barrier Analysis Summary:\n\n"
        for category, data in results.items():
            summary += f"\n{category.title()}:\n"
            summary += f"- Found in {data['count']} discussions\n"
            if data['common_terms']:
                summary += f"- Common terms: {', '.join(data['common_terms'])}\n"
            if data['contexts']:
                summary += f"- Example: '{data['contexts'][0][:100]}...'\n"
        
        ax2.text(0, 1.0, summary,
                va='top', ha='left',
                fontsize=10,
                fontfamily='monospace',
                transform=ax2.transAxes)
        
        plt.tight_layout()
        return fig 