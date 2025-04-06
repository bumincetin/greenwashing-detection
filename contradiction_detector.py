import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModel
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Any, Optional
import logging
from dataclasses import dataclass
from collections import defaultdict

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ContradictionResult:
    claim_type: str
    original_claim: str
    similarity: float
    context_weight: float
    negation_score: float
    temporal_consistency: float
    stakeholder_feedback: float

class ContradictionDetector:
    def __init__(self):
        try:
            # Initialize climate stance detection model
            self.stance_tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-stance-climate")
            self.stance_model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-stance-climate")
            
            # Initialize BERT for text embeddings with mean pooling
            self.embedding_tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/bert-base-nli-mean-tokens")
            self.embedding_model = AutoModel.from_pretrained("sentence-transformers/bert-base-nli-mean-tokens")
            
            # Initialize negation detection model - Using bert-base-uncased as replacement for DeBERTa
            negation_model_name = "bert-base-uncased"
            try:
                logger.info(f"Loading negation model: {negation_model_name}")
                # Load a standard BERT model for sequence classification (assuming 2 labels: no negation, negation)
                # Note: This model isn't pre-trained for negation. Fine-tuning on a negation dataset is recommended for accuracy.
                self.negation_tokenizer = AutoTokenizer.from_pretrained(negation_model_name)
                self.negation_model = AutoModelForSequenceClassification.from_pretrained(negation_model_name, num_labels=2) # Assuming 2 labels
                logger.info("Negation model loaded successfully.")
            except Exception as e:
                logger.error(f"Error loading negation model '{negation_model_name}': {str(e)}")
                # Fallback or raise error if critical
                self.negation_tokenizer = None
                self.negation_model = None
                raise RuntimeError(f"Failed to load necessary negation model: {negation_model_name}") from e
            
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
            
            # Initialize dynamic threshold parameters
            self.similarity_threshold = 0.7
            self.threshold_history = []
            self.min_threshold = 0.5
            self.max_threshold = 0.9
            
            # Initialize context weights
            self.context_weights = {
                'carbon_neutrality': 1.0,
                'renewable_energy': 0.9,
                'waste_reduction': 0.8
            }
            
            logger.info("ContradictionDetector initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing ContradictionDetector: {str(e)}")
            raise
    
    def get_embeddings(self, text: str, batch_size: int = 32) -> np.ndarray:
        """Get embeddings using BERT with mean pooling and batch processing"""
        try:
            # Split text into chunks if too long
            max_length = 512
            chunks = [text[i:i + max_length] for i in range(0, len(text), max_length)]
            all_embeddings = []
            
            for chunk in chunks:
                inputs = self.embedding_tokenizer(chunk, return_tensors="pt", truncation=True, max_length=max_length)
                with torch.no_grad():
                    outputs = self.embedding_model(**inputs)
                    # Mean pooling over all token embeddings
                    embeddings = outputs.last_hidden_state.mean(dim=1)[0].numpy()
                    all_embeddings.append(embeddings)
            
            # Average embeddings from all chunks
            return np.mean(all_embeddings, axis=0)
        except Exception as e:
            logger.error(f"Error getting embeddings: {str(e)}")
            return np.zeros(768)  # Return zero vector as fallback
    
    def calculate_semantic_similarity(self, text1: str, text2: str) -> float:
        """Calculate semantic similarity between two texts with error handling"""
        try:
            emb1 = self.get_embeddings(text1)
            emb2 = self.get_embeddings(text2)
            return cosine_similarity([emb1], [emb2])[0][0]
        except Exception as e:
            logger.error(f"Error calculating semantic similarity: {str(e)}")
            return 0.0
    
    def detect_negation(self, text: str) -> float:
        """Detect negation in text using the loaded sequence classification model (e.g., BERT)."""
        if not self.negation_model or not self.negation_tokenizer:
            logger.warning("Negation model/tokenizer not loaded. Skipping negation detection.")
            return 0.0 # Default to 0 if model isn't available
            
        try:
            inputs = self.negation_tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=True)
            with torch.no_grad():
                outputs = self.negation_model(**inputs)
                scores = torch.nn.functional.softmax(outputs.logits, dim=1)
            # Assuming label 1 corresponds to 'negation' for this placeholder model
            # This assumption needs validation if a specific fine-tuned model is used.
            negation_probability = scores[0][1].item()
            return negation_probability
        except Exception as e:
            logger.error(f"Error detecting negation with {self.negation_model.config.model_type}: {str(e)}")
            return 0.0 # Return 0 probability on error
    
    def update_threshold(self, similarity_scores: List[float]):
        """Dynamically update similarity threshold based on distribution"""
        if not similarity_scores:
            return
        
        # Calculate new threshold based on distribution
        mean_score = np.mean(similarity_scores)
        std_score = np.std(similarity_scores)
        new_threshold = mean_score + 0.5 * std_score
        
        # Ensure threshold stays within bounds
        new_threshold = max(self.min_threshold, min(self.max_threshold, new_threshold))
        
        # Update threshold history
        self.threshold_history.append(new_threshold)
        if len(self.threshold_history) > 100:  # Keep last 100 updates
            self.threshold_history.pop(0)
        
        # Update current threshold
        self.similarity_threshold = new_threshold
    
    def detect_contradictions(self, text: str, historical_claims: Optional[List[str]] = None) -> List[ContradictionResult]:
        """Detect contradictions in sustainability claims with context awareness"""
        contradictions = []
        similarity_scores = []
        
        try:
            # Process each claim type
            for claim_type, claim_examples in self.sustainability_claims.items():
                context_weight = self.context_weights.get(claim_type, 1.0)
                
                for claim in claim_examples:
                    similarity = self.calculate_semantic_similarity(text, claim)
                    similarity_scores.append(similarity)
                    
                    if similarity > self.similarity_threshold:
                        # Calculate additional scores
                        negation_score = self.detect_negation(text)
                        
                        # Calculate temporal consistency if historical claims provided
                        temporal_consistency = 1.0
                        if historical_claims:
                            historical_similarities = [
                                self.calculate_semantic_similarity(text, hc)
                                for hc in historical_claims
                            ]
                            temporal_consistency = 1.0 - np.mean(historical_similarities)
                        
                        # Calculate stakeholder feedback (placeholder for future implementation)
                        stakeholder_feedback = 0.5  # Default neutral value
                        
                        contradictions.append(ContradictionResult(
                            claim_type=claim_type,
                            original_claim=claim,
                            similarity=similarity,
                            context_weight=context_weight,
                            negation_score=negation_score,
                            temporal_consistency=temporal_consistency,
                            stakeholder_feedback=stakeholder_feedback
                        ))
            
            # Update threshold based on similarity distribution
            self.update_threshold(similarity_scores)
            
            return contradictions
        except Exception as e:
            logger.error(f"Error detecting contradictions: {str(e)}")
            return []
    
    def detect_stance(self, text: str) -> Dict[str, float]:
        """Detect stance on sustainability claims with calibration"""
        try:
            inputs = self.stance_tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            outputs = self.stance_model(**inputs)
            scores = torch.nn.functional.softmax(outputs.logits, dim=1)
            
            # Apply calibration to improve probability estimates
            calibrated_scores = self._calibrate_scores(scores[0])
            
            return {
                'support': calibrated_scores[1].item(),
                'oppose': calibrated_scores[0].item(),
                'neutral': calibrated_scores[2].item() if scores.shape[1] > 2 else 0.0
            }
        except Exception as e:
            logger.error(f"Error detecting stance: {str(e)}")
            return {'support': 0.0, 'oppose': 0.0, 'neutral': 1.0}
    
    def _calibrate_scores(self, scores: torch.Tensor) -> torch.Tensor:
        """Apply temperature scaling calibration to improve probability estimates"""
        temperature = 1.5  # Adjust this value based on validation
        return torch.nn.functional.softmax(scores / temperature, dim=0)
    
    def analyze_sustainability_claims(self, text: str, historical_claims: Optional[List[str]] = None) -> Dict[str, Any]:
        """Analyze sustainability claims for potential greenwashing with enhanced metrics"""
        try:
            contradictions = self.detect_contradictions(text, historical_claims)
            stance = self.detect_stance(text)
            
            # Calculate weighted contradiction score
            contradiction_score = 0
            if contradictions:
                # Weight each contradiction by its context and additional factors
                weighted_scores = []
                for c in contradictions:
                    weight = (
                        0.3 * c.context_weight +  # Context weight
                        0.3 * c.negation_score +  # Negation detection
                        0.2 * c.temporal_consistency +  # Temporal consistency
                        0.2 * c.stakeholder_feedback  # Stakeholder feedback
                    )
                    weighted_scores.append(c.similarity * weight)
                
                contradiction_score = np.mean(weighted_scores)
            
            # Analyze potential greenwashing indicators with enhanced metrics
            greenwashing_indicators = {
                'high_contradiction_score': contradiction_score > 0.8,
                'inconsistent_stance': abs(stance['support'] - stance['oppose']) < 0.2,
                'multiple_claims': len(contradictions) > 2,
                'high_negation_rate': any(c.negation_score > 0.7 for c in contradictions),
                'temporal_inconsistency': any(c.temporal_consistency > 0.7 for c in contradictions)
            }
            
            return {
                'contradictions': contradictions,
                'contradiction_score': contradiction_score,
                'stance': stance,
                'greenwashing_indicators': greenwashing_indicators,
                'threshold_used': self.similarity_threshold
            }
        except Exception as e:
            logger.error(f"Error analyzing sustainability claims: {str(e)}")
            return {
                'contradictions': [],
                'contradiction_score': 0.0,
                'stance': {'support': 0.0, 'oppose': 0.0, 'neutral': 1.0},
                'greenwashing_indicators': {},
                'threshold_used': self.similarity_threshold
            } 