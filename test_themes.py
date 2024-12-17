import sys
sys.path.append('src')
from analysis.advanced_nlp_updated import AdvancedTextAnalyzer
from analysis.data_loader import load_focus_group_data

# Initialize analyzer and load data
analyzer = AdvancedTextAnalyzer()
texts = load_focus_group_data()

# Discover themes
print("\nDiscovering themes from focus group discussions...")
themes = analyzer.discover_themes(texts)

# Print themes with details
print('\nDiscovered Themes:\n')
for theme_name, theme_info in themes.items():
    print(f'\n{theme_name}:')
    print(f'Size: {theme_info["size"]} responses')
    print(f'Keywords: {", ".join(theme_info["keywords"])}')
    print(f'Representative quote: {theme_info["representative_text"][:200]}...\n')
    print('-' * 80) 