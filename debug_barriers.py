import sys
from collections import Counter
import re
sys.path.append('src')
from analysis.data_loader import load_focus_group_data

def clean_text(text):
    """Clean text for analysis."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-z\s.,!?]', '', text)
    return ' '.join(text.split())

def analyze_barriers(texts):
    """Analyze barriers in texts."""
    # Define barrier categories with terms
    barrier_categories = {
        'Educational': {
            'cost', 'tuition', 'expensive', 'debt', 'loans', 'afford',
            'degree', 'certification', 'requirements', 'credits', 'classes',
            'training', 'education', 'school', 'university', 'college'
        },
        'Work-Life Balance': {
            'stress', 'burnout', 'workload', 'hours', 'schedule', 'balance',
            'demanding', 'pressure', 'exhausting', 'overwhelming', 'difficult',
            'challenging', 'time', 'family', 'personal'
        },
        'Emotional/Mental': {
            'emotional', 'draining', 'heavy', 'trauma', 'vicarious', 'burden',
            'tough', 'hard', 'intense', 'overwhelming', 'challenging',
            'difficult', 'stress', 'anxiety', 'depression'
        },
        'Professional/Career': {
            'salary', 'pay', 'money', 'income', 'compensation', 'benefits',
            'advancement', 'growth', 'opportunity', 'career', 'job', 'market',
            'demand', 'competition', 'experience'
        },
        'Social/Support': {
            'stigma', 'perception', 'reputation', 'respect', 'status',
            'recognition', 'support', 'isolation', 'community', 'family',
            'friends', 'social', 'network'
        }
    }
    
    results = {}
    
    # Process each category
    for category, terms in barrier_categories.items():
        mentions = []
        contexts = []
        term_freq = Counter()
        
        # Look for barrier terms in each text
        for text in texts:
            text = clean_text(text)
            
            # Check each sentence for barrier terms
            for sentence in text.split('.'):
                sentence = sentence.strip()
                if any(term in sentence for term in terms):
                    # Count which terms were found
                    found_terms = [term for term in terms if term in sentence]
                    term_freq.update(found_terms)
                    contexts.append(sentence)
                    mentions.append(text)
        
        results[category] = {
            'count': len(set(mentions)),  # Unique mentions
            'frequency': (len(set(mentions)) / len(texts)) * 100,
            'terms': dict(term_freq.most_common()),
            'example_contexts': list(set(contexts))[:5]  # Top 5 unique examples
        }
    
    return results

# Load and analyze texts
print("\nAnalyzing barriers in focus group discussions...")
texts = load_focus_group_data()
results = analyze_barriers(texts)

# Print detailed results
print('\nBarrier Analysis Results:\n')
for category, data in results.items():
    print(f'\n{category} BARRIERS:')
    print(f'Found in {data["count"]} discussions ({data["frequency"]:.1f}%)')
    
    if data['terms']:
        print('\nBarrier terms mentioned:')
        for term, freq in data['terms'].items():
            print(f'- "{term}": {freq} times')
    
    if data['example_contexts']:
        print('\nExample contexts:')
        for i, context in enumerate(data['example_contexts'][:3], 1):
            if len(context) > 20:  # Only show meaningful contexts
                print(f'\n{i}. "{context}"')
    
    print('\n' + '='*80) 