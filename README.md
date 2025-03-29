# Greenwashing Detection in Corporate Social Media Communication

This project implements a comprehensive system for detecting potential greenwashing in corporate social media communication using advanced NLP techniques and machine learning models.

## Features

- **Sentiment Analysis**: Combines multiple models (RoBERTa, DistilBERT, VADER, TextBlob) for robust sentiment detection
- **Emotion Classification**: Uses GoEmotions model fine-tuned for sustainability context
- **Contradiction Detection**: Implements Universal Sentence Encoder and SBERT for semantic similarity analysis
- **Interactive Dashboard**: Real-time visualization and analysis of social media data
- **Synthetic Data Generation**: Tools for generating realistic test data

## Project Structure

```
.
├── README.md
├── requirements.txt
├── sentiment_analyzer.py    # Sentiment analysis implementation
├── emotion_classifier.py    # Emotion classification implementation
├── contradiction_detector.py # Contradiction detection implementation
├── data_generator.py       # Synthetic data generation
├── dashboard.py           # Interactive dashboard implementation
└── synthetic_social_media_data.csv  # Generated sample data
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd greenwashing-detection
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Generate synthetic data (optional):
```python
from data_generator import DataGenerator
generator = DataGenerator()
df = generator.generate_synthetic_data(100)
generator.save_to_csv(df)
```

2. Run the dashboard:
```bash
python dashboard.py
```

3. Open your web browser and navigate to `http://localhost:8050`

## Data Format

The system expects CSV data with the following columns:
- `post_id`: Unique identifier for each post
- `company`: Company name
- `platform`: Social media platform
- `timestamp`: Post timestamp
- `content`: Post content
- `engagement_metrics`: JSON string containing platform-specific metrics
- `comment_id`: Optional comment identifier
- `sentiment_score`: Pre-calculated sentiment score
- `contradiction_score`: Pre-calculated contradiction score

## Analysis Components

### Sentiment Analysis
- Uses ensemble of multiple models for robust sentiment detection
- Normalizes scores across models for consistent analysis
- Provides weighted average sentiment score

### Emotion Classification
- Identifies emotions in sustainability-related content
- Detects potential greenwashing indicators through emotional patterns
- Categorizes emotions into sustainability-specific contexts

### Contradiction Detection
- Analyzes semantic similarity between claims
- Detects inconsistencies in sustainability messaging
- Calculates overall contradiction scores

## Dashboard Features

- Interactive data upload
- Company and platform filtering
- Date range selection
- Real-time visualization updates
- Detailed analysis metrics
- Greenwashing risk assessment

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details. 