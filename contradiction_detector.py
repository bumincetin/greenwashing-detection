import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModel
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Any, Optional, Tuple
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
    def __init__(self, use_fallback_models: bool = False):
        try:
            # Initialize climate stance detection model
            self.stance_tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-stance-climate")
            self.stance_model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-stance-climate")
            
            # Initialize sentence transformer model for better text embeddings
            self.embedding_tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-mpnet-base-v2")
            self.embedding_model = AutoModel.from_pretrained("sentence-transformers/all-mpnet-base-v2")
            
            # Initialize negation detection model using RoBERTa fine-tuned on MultiNLI
            # This model can detect contradictions which serves as a proxy for negation
            negation_model_name = "roberta-large-mnli"
            try:
                logger.info(f"Loading negation model: {negation_model_name}")
                self.negation_tokenizer = AutoTokenizer.from_pretrained(negation_model_name)
                self.negation_model = AutoModelForSequenceClassification.from_pretrained(negation_model_name)
                logger.info("Negation model loaded successfully.")
            except Exception as e:
                logger.error(f"Error loading negation model '{negation_model_name}': {str(e)}")
                if use_fallback_models:
                    # Fallback to a simpler model if primary fails
                    logger.info("Attempting to load fallback negation model")
                    self.negation_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
                    self.negation_model = AutoModelForSequenceClassification.from_pretrained(
                        "distilbert-base-uncased", num_labels=2
                    )
                    logger.warning("Using fallback negation model which is not optimized for negation detection")
                else:
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
            # Similarity threshold for detecting potential contradictions
            self.similarity_threshold = 0.7  # Higher values increase precision but reduce recall
            self.threshold_history = []
            self.min_threshold = 0.5  # Minimum acceptable threshold
            self.max_threshold = 0.9  # Maximum acceptable threshold
            
            # Initialize context weights - higher weights indicate more emphasis on that claim type
            self.context_weights = {
                'carbon_neutrality': 1.0,  # Full weight for carbon neutrality claims
                'renewable_energy': 0.9,   # Slightly lower weight for renewable energy claims
                'waste_reduction': 0.8     # Lower weight for waste reduction claims
            }
            
            # Calibration parameters for stance detection
            self.temperature = 1.5  # Temperature for softmax calibration
            
            logger.info("ContradictionDetector initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing ContradictionDetector: {str(e)}")
            raise
    
    def get_embeddings(self, text: str, batch_size: int = 32) -> np.ndarray:
        """
        Get embeddings using a sentence transformer model with mean pooling and batch processing.
        
        Args:
            text: Input text to embed
            batch_size: Batch size for processing
            
        Returns:
            numpy array of embeddings
        """
        try:
            # Split text into chunks if too long
            max_length = self.embedding_tokenizer.model_max_length
            chunks = [text[i:i + max_length] for i in range(0, len(text), max_length)]
            all_embeddings = []
            
            for i in range(0, len(chunks), batch_size):
                batch = chunks[i:i + batch_size]
                inputs = self.embedding_tokenizer(batch, return_tensors="pt", truncation=True, 
                                                  max_length=max_length, padding=True)
                
                with torch.no_grad():
                    outputs = self.embedding_model(**inputs)
                    # Mean pooling over all token embeddings and all sequences in batch
                    attention_mask = inputs['attention_mask']
                    token_embeddings = outputs.last_hidden_state
                    
                    # Apply attention mask for accurate mean pooling
                    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
                    embeddings = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
                        input_mask_expanded.sum(1), min=1e-9
                    )
                    
                    for emb in embeddings:
                        all_embeddings.append(emb.numpy())
            
            # Average embeddings from all chunks
            return np.mean(all_embeddings, axis=0)
        except Exception as e:
            logger.error(f"Error getting embeddings: {str(e)}")
            # Return zero vector of appropriate dimension as fallback
            model_dim = 768  # Default for many models, should match the actual model dimension
            return np.zeros(model_dim)
    
    def calculate_semantic_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate semantic similarity between two texts using cosine similarity.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Similarity score between 0 and 1
        """
        try:
            emb1 = self.get_embeddings(text1)
            emb2 = self.get_embeddings(text2)
            return cosine_similarity([emb1], [emb2])[0][0]
        except Exception as e:
            logger.error(f"Error calculating semantic similarity: {str(e)}")
            return 0.0
    
    def detect_negation(self, text: str) -> float:
        """
        Detect negation in text using a natural language inference model.
        
        The RoBERTa MNLI model can identify contradiction relationships between statements,
        which serves as a proxy for negation detection.
        
        Args:
            text: Text to analyze for negation
            
        Returns:
            Probability of negation between 0 and 1
        """
        if not self.negation_model or not self.negation_tokenizer:
            logger.warning("Negation model/tokenizer not loaded. Skipping negation detection.")
            return 0.0 # Default to 0 if model isn't available
            
        try:
            # For MNLI models, we need both a premise and hypothesis
            # We can frame negation detection as contradiction between statement and its opposite
            # Create a simple positive premise
            premise = "This is positive and affirming."
            
            # Check if input contradicts the positive premise
            inputs = self.negation_tokenizer(premise, text, return_tensors="pt", 
                                            truncation=True, max_length=512, padding=True)
            
            with torch.no_grad():
                outputs = self.negation_model(**inputs)
                scores = torch.nn.functional.softmax(outputs.logits, dim=1)
            
            # For MNLI models: [contradiction, neutral, entailment]
            # We're interested in contradiction score (index 0)
            contradiction_score = scores[0][0].item()
            return contradiction_score
        except Exception as e:
            logger.error(f"Error detecting negation: {str(e)}")
            return 0.0 # Return 0 probability on error
    
    def update_threshold(self, similarity_scores: List[float]):
        """
        Dynamically update similarity threshold based on the distribution of observed scores.
        
        Args:
            similarity_scores: List of similarity scores from recent comparisons
        """
        if not similarity_scores:
            return
        
        # Calculate new threshold based on distribution
        mean_score = np.mean(similarity_scores)
        std_score = np.std(similarity_scores)
        # Set threshold at mean + 0.5 std deviations to catch outliers
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
        """
        Detect contradictions in sustainability claims with context awareness.
        
        Args:
            text: Text to analyze for contradictions
            historical_claims: Optional list of previous claims for temporal consistency
            
        Returns:
            List of contradiction results
        """
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
        """
        Detect stance on sustainability claims with calibrated probability estimates.
        
        Args:
            text: Text to analyze for stance
            
        Returns:
            Dictionary with support, oppose, and neutral stance probabilities
        """
        try:
            inputs = self.stance_tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            with torch.no_grad():
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
        """
        Apply temperature scaling calibration to improve probability estimates.
        
        Args:
            scores: Raw logit scores from the model
            
        Returns:
            Calibrated probability distribution
        """
        # Temperature scaling: T > 1 makes distribution more uniform, T < 1 makes it more peaked
        return torch.nn.functional.softmax(scores / self.temperature, dim=0)
    
    def analyze_sustainability_claims(self, text: str, historical_claims: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Analyze sustainability claims for potential greenwashing with enhanced metrics.
        
        Args:
            text: Text containing sustainability claims to analyze
            historical_claims: Optional list of previous claims for temporal consistency
            
        Returns:
            Dictionary with analysis results, including contradictions, scores, and indicators
        """
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
                        0.3 * c.context_weight +     # Context weight: domain-specific importance
                        0.3 * c.negation_score +     # Negation detection: presence of negating terms
                        0.2 * c.temporal_consistency + # Temporal consistency: changes over time
                        0.2 * c.stakeholder_feedback   # Stakeholder feedback: external validation
                    )
                    weighted_scores.append(c.similarity * weight)
                
                contradiction_score = np.mean(weighted_scores)
            
            # Analyze potential greenwashing indicators with enhanced metrics
            # These thresholds are based on research and may require adjustment
            greenwashing_indicators = {
                'high_contradiction_score': contradiction_score > 0.8,  # High internal contradiction
                'inconsistent_stance': abs(stance['support'] - stance['oppose']) < 0.2,  # Ambiguous stance
                'multiple_claims': len(contradictions) > 2,  # Multiple contradictory claims
                'high_negation_rate': any(c.negation_score > 0.7 for c in contradictions),  # Strong negations
                'temporal_inconsistency': any(c.temporal_consistency > 0.7 for c in contradictions)  # Changes over time
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