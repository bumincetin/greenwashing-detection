# Greenwashing Detection in Corporate Social Media Communication

This project implements a comprehensive system for detecting potential greenwashing in corporate social media communication using advanced NLP techniques and machine learning models.

## Features

- **Sentiment Analysis**: Combines multiple models (RoBERTa, DistilBERT, VADER, TextBlob) for robust sentiment detection
- **Emotion Classification**: Uses GoEmotions model fine-tuned for sustainability context
- **Contradiction Detection**: Implements specialized NLI models and attention-based pooling for accurate greenwashing claim detection
- **Interactive Dashboard**: Real-time visualization and analysis of social media data
- **Synthetic Data Generation**: Tools for generating realistic test data
- **Methodology Guide**: Detailed explanation of algorithms and interpretation guidelines

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
git clone https://github.com/bumincetin/greenwashing-detection.git
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

## Contradiction Detection

Our enhanced contradiction detection system uses:

- **RoBERTa MNLI Model**: Fine-tuned on the MultiNLI dataset for superior contradiction detection
- **MPNet Embeddings**: State-of-the-art sentence embeddings for semantic similarity analysis
- **Attention-based Pooling**: Weighted token representation for more accurate text embeddings
- **Dynamic Thresholding**: Adaptive threshold determination based on similarity distributions
- **Multi-factor Analysis**: Combined context weight, negation detection, and temporal consistency
- **Fallback Model Support**: Graceful degradation with fallback models for improved reliability

## Dashboard Components

### 1. Post Analysis Section
- **Sentiment Trend**: Track sentiment changes over time
- **Emotion Distribution**: Analyze emotional content distribution
- **Greenwashing Risk Analysis**: Multi-factor risk assessment

### 2. Comment Analysis Section
- **Comment Sentiment Distribution**: Public response analysis
- **Comment Engagement Trend**: Track engagement patterns
- **Comment Statistics**: Detailed metrics and counts

### 3. Detailed Analysis Section
- **Key Metrics**: Average scores and totals
- **Top Emotions**: Most frequent emotional expressions
- **Greenwashing Indicators**: Risk factor detection

### 4. Methodology Guide
- **Interactive Documentation**: Collapsible methodology explanations
- **Interpretation Guidelines**: How to read each visualization
- **Technical Details**: Formulas and calculations explained

## Data Format

The system expects CSV data with the following columns:
- `post_id`: Unique identifier for each post
- `company`: Company name
- `platform`: Social media platform
- `timestamp`: Post timestamp
- `content`: Post content
- `engagement_metrics`: JSON string containing platform-specific metrics
- `comments`: List of comment texts
- `comment_sentiments`: List of comment sentiment labels
- `sentiment_score`: Pre-calculated sentiment score
- `contradiction_score`: Pre-calculated contradiction score

## Algorithm Details

### 1. Sentiment Analysis (Ensemble Approach)
- **VADER (30%)**: Social media optimized
- **TextBlob (20%)**: Pattern-based analysis
- **RoBERTa (30%)**: Deep learning model
- **DistilBERT (20%)**: Efficient transformer

### 2. Emotion Classification
- Uses GoEmotions model with sustainability focus
- Detects 12 primary emotions
- Identifies greenwashing indicators:
  - Excessive optimism (> 0.8)
  - Lack of authenticity (> 0.9)
  - Overconfidence (> 0.8)

### 3. Contradiction Detection Algorithm
- **Similarity Calculation**: Cosine similarity between claim embeddings using MPNet
- **Negation Detection**: MNLI-based contradiction detection for identifying negated claims
- **Context Weighting**: Domain-specific importance factors (carbon neutrality, renewable energy, waste reduction)
- **Temporal Consistency**: Comparison with historical claims to detect inconsistencies
- **Weighted Scoring**: Combined factor scoring with relative importance weights 
- **Calibration**: Temperature-based softmax calibration for improved probability estimates

### 4. Risk Analysis
- **Risk Score**: `(|sentiment| + contradiction + comment_risk) / 3`
- **Comment Risk**: `(negative + 0.5 * skeptical) / total_comments`
- **Visualization**:
  - Point size: 5-25 (normalized risk score)
  - Color: Green (low risk) to Red (high risk)
  - Position: Sentiment vs Contradiction scores

## Visualization Features

### 1. Interactive Elements
- Company and platform filters
- Date range selection
- Hover information
- Zoom and pan capabilities
- Show/Hide methodology guide

### 2. Real-time Updates
- Automatic recalculation
- Dynamic filtering
- Responsive layout
- Error handling and empty state displays

### 3. Enhanced Visualizations
- Custom color schemes
- Clear labeling
- Informative tooltips
- Trend lines and rolling averages
- Risk zone indicators

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details. 