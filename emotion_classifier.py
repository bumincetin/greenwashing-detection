import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np

class EmotionClassifier:
    def __init__(self):
        # Initialize GoEmotions model
        self.tokenizer = AutoTokenizer.from_pretrained("bhadresh-savani/bert-base-go-emotion")
        self.model = AutoModelForSequenceClassification.from_pretrained("bhadresh-savani/bert-base-go-emotion")
        
        # Define emotion categories
        self.emotion_categories = [
            'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring',
            'confusion', 'curiosity', 'desire', 'disappointment', 'disapproval',
            'disgust', 'embarrassment', 'excitement', 'fear', 'gratitude', 'grief',
            'joy', 'love', 'nervousness', 'optimism', 'pride', 'realization',
            'relief', 'remorse', 'sadness', 'satisfaction', 'shame', 'surprise',
            'neutral'
        ]
        
    def classify_emotions(self, text):
        """Classify emotions in the text"""
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        outputs = self.model(**inputs)
        scores = torch.nn.functional.sigmoid(outputs.logits)
        
        # Get top emotions (threshold > 0.5)
        emotion_scores = {}
        for idx, score in enumerate(scores[0]):
            if score.item() > 0.5:
                emotion_scores[self.emotion_categories[idx]] = score.item()
        
        # Sort emotions by score
        sorted_emotions = dict(sorted(emotion_scores.items(), key=lambda x: x[1], reverse=True))
        
        return {
            'emotions': sorted_emotions,
            'primary_emotion': next(iter(sorted_emotions)) if sorted_emotions else 'neutral'
        }
    
    def get_sustainability_emotions(self, text):
        """Analyze emotions specifically related to sustainability"""
        sustainability_keywords = [
            'environment', 'climate', 'sustainable', 'green', 'renewable',
            'carbon', 'emissions', 'recycling', 'eco-friendly', 'biodiversity'
        ]
        
        # Check if text contains sustainability keywords
        has_sustainability_context = any(keyword in text.lower() for keyword in sustainability_keywords)
        
        if not has_sustainability_context:
            return {
                'has_sustainability_context': False,
                'message': 'No sustainability-related content detected'
            }
        
        # Get emotion classification
        emotion_results = self.classify_emotions(text)
        
        # Analyze potential greenwashing indicators
        greenwashing_indicators = {
            'excessive_optimism': emotion_results['emotions'].get('optimism', 0) > 0.8,
            'lack_of_authenticity': emotion_results['emotions'].get('joy', 0) > 0.9,
            'overconfidence': emotion_results['emotions'].get('pride', 0) > 0.8
        }
        
        return {
            'has_sustainability_context': True,
            'emotions': emotion_results['emotions'],
            'primary_emotion': emotion_results['primary_emotion'],
            'greenwashing_indicators': greenwashing_indicators
        } 