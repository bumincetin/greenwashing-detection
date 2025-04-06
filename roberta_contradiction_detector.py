import torch
from transformers import (
    RobertaForSequenceClassification,
    RobertaTokenizer,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback
)
from datasets import load_dataset
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from dataclasses import dataclass
import json
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ContradictionResult:
    premise: str
    hypothesis: str
    label: int  # 0: contradiction, 1: neutral, 2: entailment
    confidence: float
    domain_specific_score: float

class RobertaContradictionDetector:
    def __init__(self, model_path: Optional[str] = None):
        try:
            # Initialize tokenizer and model
            self.tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
            self.model = RobertaForSequenceClassification.from_pretrained(
                "roberta-base",
                num_labels=3  # contradiction, neutral, entailment
            )
            
            # Load domain-specific sustainability claims
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
            
            # Initialize domain-specific keywords
            self.domain_keywords = [
                "sustainable", "green", "eco-friendly", "renewable",
                "carbon-neutral", "environmental", "climate", "emissions",
                "recycling", "clean energy", "net-zero", "biodiversity"
            ]
            
            # Load fine-tuned model if path provided
            if model_path and os.path.exists(model_path):
                self.model.load_state_dict(torch.load(model_path))
                logger.info(f"Loaded fine-tuned model from {model_path}")
            
            self.model.eval()
            logger.info("RobertaContradictionDetector initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing RobertaContradictionDetector: {str(e)}")
            raise
    
    def prepare_dataset(self, dataset_name: str = "glue", dataset_config: str = "mnli") -> Dict:
        """Prepare and tokenize the dataset"""
        try:
            # Load dataset
            dataset = load_dataset(dataset_name, dataset_config)
            
            def tokenize_function(examples):
                return self.tokenizer(
                    examples["premise"],
                    examples["hypothesis"],
                    truncation=True,
                    max_length=128,
                    padding="max_length"
                )
            
            # Tokenize datasets
            tokenized_datasets = dataset.map(
                tokenize_function,
                batched=True,
                remove_columns=dataset["train"].column_names
            )
            
            return tokenized_datasets
            
        except Exception as e:
            logger.error(f"Error preparing dataset: {str(e)}")
            raise
    
    def compute_metrics(self, eval_pred: Tuple[np.ndarray, np.ndarray]) -> Dict[str, float]:
        """Compute evaluation metrics"""
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predictions, average='weighted'
        )
        accuracy = accuracy_score(labels, predictions)
        
        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1
        }
    
    def fine_tune(self, output_dir: str = "./results_roberta", num_epochs: int = 3):
        """Fine-tune the model on MNLI dataset"""
        try:
            # Prepare dataset
            tokenized_datasets = self.prepare_dataset()
            
            # Define training arguments
            training_args = TrainingArguments(
                output_dir=output_dir,
                evaluation_strategy="epoch",
                num_train_epochs=num_epochs,
                per_device_train_batch_size=16,
                per_device_eval_batch_size=16,
                warmup_steps=500,
                weight_decay=0.01,
                logging_dir="./logs",
                logging_steps=100,
                save_strategy="epoch",
                load_best_model_at_end=True,
                metric_for_best_model="f1"
            )
            
            # Initialize trainer
            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=tokenized_datasets["train"],
                eval_dataset=tokenized_datasets["validation_matched"],
                compute_metrics=self.compute_metrics,
                callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
            )
            
            # Train the model
            trainer.train()
            
            # Save the model
            trainer.save_model(output_dir)
            logger.info(f"Model saved to {output_dir}")
            
        except Exception as e:
            logger.error(f"Error during fine-tuning: {str(e)}")
            raise
    
    def detect_domain_specific_contradiction(self, text: str, claim: str) -> float:
        """Detect domain-specific contradictions in sustainability claims"""
        try:
            # Check for domain-specific keywords
            has_domain_context = any(keyword in text.lower() for keyword in self.domain_keywords)
            if not has_domain_context:
                return 0.0
            
            # Prepare input
            inputs = self.tokenizer(
                text,
                claim,
                return_tensors="pt",
                truncation=True,
                max_length=128,
                padding="max_length"
            )
            
            # Get model predictions
            with torch.no_grad():
                outputs = self.model(**inputs)
                scores = torch.nn.functional.softmax(outputs.logits, dim=1)
                
                # Get contradiction score (label 0)
                return scores[0][0].item()
                
        except Exception as e:
            logger.error(f"Error detecting domain-specific contradiction: {str(e)}")
            return 0.0
    
    def analyze_claim(self, text: str, claim: str) -> ContradictionResult:
        """Analyze a specific claim for contradictions"""
        try:
            # Get general contradiction score
            inputs = self.tokenizer(
                text,
                claim,
                return_tensors="pt",
                truncation=True,
                max_length=128,
                padding="max_length"
            )
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                scores = torch.nn.functional.softmax(outputs.logits, dim=1)
                
                # Get prediction and confidence
                prediction = torch.argmax(scores).item()
                confidence = scores[0][prediction].item()
            
            # Get domain-specific score
            domain_score = self.detect_domain_specific_contradiction(text, claim)
            
            return ContradictionResult(
                premise=text,
                hypothesis=claim,
                label=prediction,
                confidence=confidence,
                domain_specific_score=domain_score
            )
            
        except Exception as e:
            logger.error(f"Error analyzing claim: {str(e)}")
            return ContradictionResult(
                premise=text,
                hypothesis=claim,
                label=1,  # neutral as fallback
                confidence=0.0,
                domain_specific_score=0.0
            )
    
    def analyze_sustainability_claims(self, text: str) -> Dict[str, Any]:
        """Analyze sustainability claims for potential contradictions"""
        try:
            results = []
            for claim_type, claims in self.sustainability_claims.items():
                for claim in claims:
                    result = self.analyze_claim(text, claim)
                    results.append({
                        'claim_type': claim_type,
                        'claim': claim,
                        'result': result
                    })
            
            # Calculate overall contradiction score
            contradiction_scores = [
                r['result'].confidence * r['result'].domain_specific_score
                for r in results
                if r['result'].label == 0  # Only consider contradictions
            ]
            
            overall_score = np.mean(contradiction_scores) if contradiction_scores else 0.0
            
            return {
                'results': results,
                'contradiction_score': overall_score,
                'has_contradictions': any(r['result'].label == 0 for r in results),
                'domain_specific_contradictions': any(
                    r['result'].domain_specific_score > 0.7
                    for r in results
                )
            }
            
        except Exception as e:
            logger.error(f"Error analyzing sustainability claims: {str(e)}")
            return {
                'results': [],
                'contradiction_score': 0.0,
                'has_contradictions': False,
                'domain_specific_contradictions': False
            } 