# Binghamton University Workforce Development Barriers Analysis

This repository contains analysis tools and code for investigating barriers to entry in substance use disorder (SUD) counseling workforce development. The project focuses on understanding and addressing the significant shortage of SUD counselors in the United States.

## Project Overview

According to recent SAMHSA reports, while 48 million Americans meet criteria for substance use disorders, 85% do not receive needed care, partly due to workforce shortages. This project aims to:

1. Identify socio-economic barriers/challenges and potential mitigating factors impacting SUD counselor workforce development
2. Analyze how these barriers vary based on demographic factors

## Repository Structure

```
.
├── data/                    # Data directory (not tracked in git)
│   ├── focus-groups/       # Focus group transcripts
│   ├── raw/               # Raw data files
│   └── processed/         # Processed data files
├── src/                    # Source code
│   ├── preprocessing/     # Data preprocessing scripts
│   ├── analysis/         # Analysis scripts
│   └── visualization/    # Visualization scripts
└── results/               # Analysis results and figures

```

## Data Security

All sensitive data is stored locally and not tracked in git. The `.gitignore` file is configured to exclude data directories while maintaining the structure through `.gitkeep` files.

## Setup

### Option 1: Automatic Setup (Recommended)

1. Clone the repository:
```bash
git clone https://github.com/yourusername/bing_wfd_barrier.git
cd bing_wfd_barrier
```

2. Run the setup script:
```bash
./setup.sh
```

3. Activate the virtual environment:
```bash
source venv/bin/activate
```

### Option 2: Manual Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/bing_wfd_barrier.git
cd bing_wfd_barrier
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create necessary directories:
```bash
mkdir -p data/focus-groups data/raw data/processed results/visualizations
```

## Analysis Pipeline

The analysis pipeline consists of Python scripts for:
- Text preprocessing of focus group transcripts
- Embedding model analysis for theme generation
- Temporal pattern analysis
- Topic modeling
- Qualitative analysis support tools

### Running Analysis

The analysis can be run using the Python scripts in the `src` directory:

```bash
# Process focus group transcripts
python src/preprocessing/process_transcripts.py

# Run topic modeling analysis
python src/analysis/topic_modeling.py

# Generate visualizations
python src/visualization/generate_plots.py
```

### Analysis Outputs

The analysis generates several outputs in the `results` directory:
- Descriptive statistics of focus group discussions
- Identified themes with representative quotes
- Temporal pattern visualizations
- Speaker engagement analysis
- Barrier and solution distribution plots
- Theme similarity networks

## Contributing

Please contact the project maintainers for information about contributing to this project.

## License

[Add appropriate license information]

## Contact

[Add contact information]
