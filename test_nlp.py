from src.analysis.advanced_nlp import AdvancedTextAnalyzer
import matplotlib.pyplot as plt

# Initialize analyzer
analyzer = AdvancedTextAnalyzer()

# Test text
test_texts = [
    'I want to help people overcome addiction and substance use disorders',
    'The counseling career path requires education and certification',
    'Working with patients can be challenging but rewarding',
    'Mental health support is really important in this field'
]

# Test word cloud
fig = analyzer.create_enhanced_wordcloud(test_texts, 'Test Word Cloud')
plt.savefig('test_wordcloud.png')
print('Created test_wordcloud.png')

# Test collocations
bigrams = analyzer.find_collocations(test_texts)
print('\nTop word pairs:')
for pair in bigrams[:5]:
    print(f'  {pair[0]} {pair[1]}')

# Test semantic clustering with fewer clusters
clusters, texts = analyzer.cluster_responses(test_texts, n_clusters=2)
print('\nCluster assignments:', clusters)

print('\nTest completed successfully!') 