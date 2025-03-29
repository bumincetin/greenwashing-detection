import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModel
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class ContradictionDetector:
    def __init__(self):
        # Initialize climate stance detection model
        self.stance_tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-stance-climate")
        self.stance_model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-stance-climate")
        
        # Initialize BERT for text embeddings
        self.embedding_tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/bert-base-nli-mean-tokens")
        self.embedding_model = AutoModel.from_pretrained("sentence-transformers/bert-base-nli-mean-tokens")
        
        # Define sustainability-related claims and their contradictions
        self.sustainability_claims = {
            'carbon_neutrality': [
                "We are carbon neutral",
                "We have achieved net-zero emissions",
                "Our operations are carbon-free"
            ],
            'renewable_energy': [
                "We use 100% renewable energy",
                "Our facilities are powered by clean energy",
                "We have transitioned to renewable sources"
            ],
            'waste_reduction': [
                "We have zero waste",
                "All our waste is recycled",
                "We are waste-free"
            ]
        }
    
    def get_embeddings(self, text):
        """Get embeddings using BERT"""
        inputs = self.embedding_tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            outputs = self.embedding_model(**inputs)
            # Use the [CLS] token embedding
            embeddings = outputs.last_hidden_state[0][0].numpy()
        return embeddings
    
    def calculate_semantic_similarity(self, text1, text2):
        """Calculate semantic similarity between two texts"""
        emb1 = self.get_embeddings(text1)
        emb2 = self.get_embeddings(text2)
        return cosine_similarity([emb1], [emb2])[0][0]
    
    def detect_contradictions(self, text):
        """Detect contradictions in sustainability claims"""
        contradictions = []
        
        for claim_type, claim_examples in self.sustainability_claims.items():
            for claim in claim_examples:
                similarity = self.calculate_semantic_similarity(text, claim)
                if similarity > 0.7:  # High similarity threshold
                    # Check for contradictory statements in the same text
                    contradictions.append({
                        'claim_type': claim_type,
                        'original_claim': claim,
                        'similarity': similarity
                    })
        
        return contradictions
    
    def detect_stance(self, text):
        """Detect stance on sustainability claims"""
        inputs = self.stance_tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        outputs = self.stance_model(**inputs)
        scores = torch.nn.functional.softmax(outputs.logits, dim=1)
        
        # Map climate stance model outputs to stance categories
        return {
            'support': scores[0][1].item(),
            'oppose': scores[0][0].item(),
            'neutral': scores[0][2].item() if scores.shape[1] > 2 else 0.0
        }
    
    def analyze_sustainability_claims(self, text):
        """Analyze sustainability claims for potential greenwashing"""
        contradictions = self.detect_contradictions(text)
        stance = self.detect_stance(text)
        
        # Calculate overall contradiction score
        contradiction_score = 0
        if contradictions:
            contradiction_score = sum(c['similarity'] for c in contradictions) / len(contradictions)
        
        # Analyze potential greenwashing indicators
        greenwashing_indicators = {
            'high_contradiction_score': contradiction_score > 0.8,
            'inconsistent_stance': abs(stance['support'] - stance['oppose']) < 0.2,
            'multiple_claims': len(contradictions) > 2
        }
        
        return {
            'contradictions': contradictions,
            'contradiction_score': contradiction_score,
            'stance': stance,
            'greenwashing_indicators': greenwashing_indicators
        } 