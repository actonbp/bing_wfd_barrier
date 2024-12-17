import sys
import pandas as pd
from collections import Counter
import re
from pathlib import Path

def load_texts():
    """Load focus group discussion texts."""
    processed_dir = Path('data/processed')
    all_responses = []
    for file in processed_dir.glob('*_processed.csv'):
        df = pd.read_csv(file)
        # Filter out moderator (Speaker 1)
        student_responses = df[df['Speaker'] != 'Speaker 1']
        all_responses.extend(student_responses['cleaned_text'].tolist())
    return all_responses

def clean_text(text):
    """Clean text for analysis."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-z\s.,!?]', '', text)
    return ' '.join(text.split())

def find_barrier_mentions(text, terms):
    """Find mentions of barrier terms in context."""
    text = clean_text(text)
    mentions = []
    
    # Split into sentences for better context
    sentences = text.split('.')
    for sentence in sentences:
        sentence = sentence.strip()
        # Look for barrier terms
        found_terms = [term for term in terms if term in sentence]
        if found_terms:
            mentions.append({
                'terms': found_terms,
                'context': sentence
            })
    return mentions

def analyze_barriers(texts):
    """Detailed analysis of barriers mentioned in discussions."""
    # Define barrier categories with expanded terms
    barrier_categories = {
        'Educational': {
            # Direct costs
            'cost', 'tuition', 'expensive', 'debt', 'loans', 'afford', 'money', 'financial',
            # Requirements
            'degree', 'certification', 'requirements', 'credits', 'classes', 'training',
            'education', 'school', 'university', 'college', 'program', 'coursework',
            # Time investment
            'years', 'semester', 'duration', 'timeline'
        },
        'Work-Life Balance': {
            # Time management
            'schedule', 'balance', 'time', 'hours', 'workload', 'overtime',
            # Stress factors
            'stress', 'burnout', 'demanding', 'pressure', 'exhausting', 'overwhelming',
            # Personal life
            'family', 'personal', 'life', 'children', 'relationship', 'commitments'
        },
        'Emotional/Mental': {
            # Emotional impact
            'emotional', 'draining', 'heavy', 'trauma', 'vicarious', 'burden',
            'intense', 'overwhelming', 'difficult', 'challenging',
            # Mental health
            'stress', 'anxiety', 'depression', 'mental', 'health',
            # Coping
            'cope', 'handle', 'manage', 'support', 'self-care'
        },
        'Professional/Career': {
            # Compensation
            'salary', 'pay', 'income', 'compensation', 'benefits', 'insurance',
            # Career growth
            'advancement', 'growth', 'opportunity', 'career', 'promotion', 'future',
            # Job market
            'market', 'demand', 'competition', 'experience', 'entry', 'level',
            # Professional challenges
            'caseload', 'paperwork', 'documentation', 'liability'
        },
        'Social/Support': {
            # Social perception
            'stigma', 'perception', 'reputation', 'respect', 'status', 'recognition',
            # Support systems
            'support', 'mentor', 'guidance', 'help', 'resources', 'network',
            # Community
            'community', 'isolation', 'connection', 'relationship', 'colleagues'
        }
    }
    
    results = {}
    
    # Analyze each category
    for category, terms in barrier_categories.items():
        all_mentions = []
        term_freq = Counter()
        
        # Process each text
        for text in texts:
            mentions = find_barrier_mentions(text, terms)
            if mentions:
                all_mentions.extend(mentions)
                for mention in mentions:
                    term_freq.update(mention['terms'])
        
        # Group similar contexts
        context_groups = {}
        for mention in all_mentions:
            context = mention['context']
            terms_key = ' '.join(sorted(mention['terms']))
            if terms_key not in context_groups:
                context_groups[terms_key] = []
            context_groups[terms_key].append(context)
        
        results[category] = {
            'total_mentions': len(all_mentions),
            'unique_contexts': len(context_groups),
            'frequency': (len(set(m['context'] for m in all_mentions)) / len(texts)) * 100,
            'term_frequencies': dict(term_freq.most_common()),
            'context_groups': context_groups
        }
    
    return results

# Load and analyze texts
print("\nAnalyzing barriers in focus group discussions...")
texts = load_texts()
results = analyze_barriers(texts)

# Print detailed analysis
print('\nDetailed Barrier Analysis:\n')
for category, data in results.items():
    print(f'\n{category} BARRIERS:')
    print(f'Total mentions: {data["total_mentions"]}')
    print(f'Unique contexts: {data["unique_contexts"]}')
    print(f'Frequency: {data["frequency"]:.1f}% of discussions')
    
    if data['term_frequencies']:
        print('\nMost frequently mentioned barrier terms:')
        for term, freq in data['term_frequencies'].items():
            print(f'- "{term}": {freq} times')
    
    if data['context_groups']:
        print('\nExample contexts by term groups:')
        for terms, contexts in data['context_groups'].items():
            print(f'\nTerms: {terms}')
            # Show up to 3 example contexts for each term group
            for context in contexts[:3]:
                print(f'- "{context}"')
    
    print('\n' + '='*80) 