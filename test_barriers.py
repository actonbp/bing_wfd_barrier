import sys
sys.path.append('src')
from analysis.barrier_analyzer import BarrierAnalyzer
from analysis.data_loader import load_focus_group_data

# Initialize analyzer and load data
analyzer = BarrierAnalyzer()
texts = load_focus_group_data()

# Test barrier analysis
print("\nAnalyzing barriers in focus group discussions...")
barrier_results = analyzer.analyze(texts)

# Print detailed results
print('\nDetailed Barrier Analysis Results:\n')
for category, results in barrier_results.items():
    print(f'\n{category.upper()} BARRIERS:')
    print(f'Frequency: {results["frequency"]:.1f}% of discussions')
    print(f'Total mentions: {results["count"]} times')
    
    if results['common_terms']:
        print('\nMost common barrier terms in this category:')
        for term, freq in results['common_terms'].items():
            print(f'- {term}: mentioned {freq} times')
    
    if results['contexts']:
        print('\nExample contexts:')
        for i, context in enumerate(results['contexts'][:3], 1):
            print(f'\n{i}. "{context.strip()}"')
    
    print('\n' + '='*80) 