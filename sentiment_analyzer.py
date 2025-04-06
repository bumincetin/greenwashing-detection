import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
import numpy as np

class SentimentAnalyzer:
    def __init__(self):
        # Initialize VADER
        self.vader = SentimentIntensityAnalyzer()
        
        # Initialize RoBERTa for sentiment
        self.roberta_tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
        self.roberta_model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
        
        # Initialize DistilBERT for sentiment
        self.distilbert_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
        self.distilbert_model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
        
    def analyze_vader(self, text):
        """Analyze sentiment using VADER"""
        scores = self.vader.polarity_scores(text)
        return {
            'compound': scores['compound'],
            'pos': scores['pos'],
            'neg': scores['neg'],
            'neu': scores['neu']
        }
    
    def analyze_textblob(self, text):
        """Analyze sentiment using TextBlob"""
        analysis = TextBlob(text)
        return {
            'polarity': analysis.sentiment.polarity,
            'subjectivity': analysis.sentiment.subjectivity
        }
    
    def analyze_roberta(self, text):
        """Analyze sentiment using RoBERTa"""
        inputs = self.roberta_tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        outputs = self.roberta_model(**inputs)
        scores = torch.nn.functional.softmax(outputs.logits, dim=1)
        return {
            'positive': scores[0][2].item(),
            'negative': scores[0][0].item(),
            'neutral': scores[0][1].item()
        }
    
    def analyze_distilbert(self, text):
        """Analyze sentiment using DistilBERT"""
        inputs = self.distilbert_tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        outputs = self.distilbert_model(**inputs)
        scores = torch.nn.functional.softmax(outputs.logits, dim=1)
        return {
            'positive': scores[0][1].item(),
            'negative': scores[0][0].item()
        }
    
    def get_ensemble_sentiment(self, text):
        """Get ensemble sentiment score combining all models"""
        vader_scores = self.analyze_vader(text)
        textblob_scores = self.analyze_textblob(text)
        roberta_scores = self.analyze_roberta(text)
        distilbert_scores = self.analyze_distilbert(text)
        
        # Normalize scores to [-1, 1] range
        normalized_scores = {
            'vader': vader_scores['compound'],
            'textblob': textblob_scores['polarity'],
            'roberta': (roberta_scores['positive'] - roberta_scores['negative']),
            'distilbert': (distilbert_scores['positive'] - distilbert_scores['negative'])
        }
        
        # Calculate ensemble score (weighted average)
        weights = {'vader': 0.3, 'textblob': 0.2, 'roberta': 0.3, 'distilbert': 0.2}
        ensemble_score = sum(score * weights[model] for model, score in normalized_scores.items())
        
        return {
            'ensemble_score': ensemble_score,
            'individual_scores': normalized_scores
        }

    def get_sentiment_category(self, text: str) -> str:
        """Classify text into 'positive', 'negative', or 'skeptical' based on ensemble score."""
        if not text or not isinstance(text, str):
            return 'neutral' # Or handle empty/invalid input as needed
            
        ensemble_result = self.get_ensemble_sentiment(text)
        score = ensemble_result['ensemble_score']
        
        # Define thresholds for categorization
        if score > 0.1:  # Adjust threshold as needed
            return 'positive'
        elif score < -0.1: # Adjust threshold as needed
            return 'negative'
        else:
            return 'skeptical' # Treat near-neutral scores as skeptical/uncertain 