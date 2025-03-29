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

## Detailed Algorithm Explanation

### 1. Sentiment Analysis
The sentiment analysis component uses an ensemble of multiple models to provide robust sentiment detection:

- **VADER (Valence Aware Dictionary and sEntiment Reasoner)**
  - Rule-based model specifically designed for social media text
  - Handles emojis, slang, and common expressions
  - Provides compound score between -1 and 1

- **TextBlob**
  - Uses pattern-based analysis
  - Provides polarity (-1 to 1) and subjectivity (0 to 1) scores
  - Good for formal text analysis

- **RoBERTa and DistilBERT**
  - Transformer-based models fine-tuned for sentiment analysis
  - RoBERTa: More accurate but computationally expensive
  - DistilBERT: Lighter version with good performance

The final sentiment score is calculated as a weighted ensemble:
```python
final_score = 0.3 * vader_score + 0.2 * textblob_score + 0.3 * roberta_score + 0.2 * distilbert_score
```

### 2. Emotion Classification
The emotion classification system uses a specialized model for sustainability context:

- **Climate-Specific Emotion Detection**
  - Uses `j-hartmann/emotion-english-distilroberta-base` model
  - Classifies emotions into 12 categories:
    - joy, sadness, anger, love, fear, surprise
    - neutral, disgust, shame, guilt, pride, optimism

- **Greenwashing Indicators**
  - Excessive optimism (score > 0.8)
  - Lack of authenticity (joy score > 0.9)
  - Overconfidence (pride score > 0.8)

### 3. Contradiction Detection
The contradiction detection system uses multiple approaches:

- **Semantic Similarity Analysis**
  - Uses BERT embeddings from `sentence-transformers/bert-base-nli-mean-tokens`
  - Calculates cosine similarity between claims
  - Identifies similar sustainability claims

- **Stance Detection**
  - Uses `cardiffnlp/twitter-roberta-base-stance-climate` model
  - Classifies stance as support, oppose, or neutral
  - Specifically trained for climate-related content

- **Contradiction Scoring**
  ```python
  contradiction_score = (similarity_score + stance_inconsistency) / 2
  ```

## Visualization Components

### 1. Sentiment Analysis Visualizations
- **Sentiment Trend**
  - Line plot showing sentiment scores over time
  - Helps identify patterns and changes in sentiment
  - Interactive tooltips for detailed information

- **Sentiment Distribution**
  - Histogram showing the distribution of sentiment scores
  - Helps identify overall sentiment bias
  - Includes statistical summary

### 2. Emotion Classification Visualizations
- **Emotion Distribution**
  - Pie chart showing distribution of emotions
  - Color-coded for easy interpretation
  - Interactive legend and tooltips

- **Emotion Trend**
  - Line plot showing emotion changes over time
  - Helps identify emotional patterns
  - Multiple emotion tracking

### 3. Contradiction Analysis Visualizations
- **Contradiction Score Distribution**
  - Box plot showing distribution of contradiction scores
  - Identifies outliers and patterns
  - Statistical summary included

- **Greenwashing Risk Analysis**
  - Scatter plot with:
    - X-axis: Sentiment Score
    - Y-axis: Contradiction Score
    - Point size: Risk Score
    - Color: Risk Level (Red to Green)
  - Risk Score Calculation:
    ```python
    risk_score = (abs(sentiment) + contradiction) / 2
    normalized_size = (risk_score - min_risk) / (max_risk - min_risk) * 20 + 5
    ```

### 4. Detailed Analysis Section
- **Key Metrics**
  - Average sentiment score
  - Average contradiction score
  - Total posts analyzed

- **Top Emotions**
  - Most frequent emotions
  - Occurrence counts
  - Percentage distribution

## Dashboard Features

### 1. Data Upload
- CSV file upload support
- Automatic data validation
- Real-time processing

### 2. Filtering Options
- Company selection
- Platform selection
- Date range selection

### 3. Interactive Elements
- Hover tooltips
- Zoom capabilities
- Pan and zoom controls
- Download plot options

### 4. Real-time Updates
- Automatic refresh on filter changes
- Dynamic data loading
- Responsive layout

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details. 