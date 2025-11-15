# Pure embedding-based threat detection (no LLM, fully offline)

from sentence_transformers import SentenceTransformer
import numpy as np
import json
import os

class PureEmbeddingGuard:
    """
    Pure embedding-based threat detector
    - No API calls
    - 100% offline
    - Fast classification
    - Uses cosine similarity only
    """
    
    ATTACK_TYPES = [
        "prompt_extraction",
        "jailbreak", 
        "data_exfiltration",
        "harmful_content",
        "tool_misuse",
        "social_engineering",
        "benign"
    ]
    
    def __init__(self, model_name='all-MiniLM-L6-v2', verbose=False):
        """
        Initialize pure embedding guard
        
        Args:
            model_name: sentence-transformers model name
            verbose: Enable debug logging
        """
        self.verbose = verbose
        
        if self.verbose:
            print(f"Loading embedding model: {model_name}")
        
        self.model = SentenceTransformer(model_name)
        
        if self.verbose:
            print("✓ Embedding model loaded")
        
        # Load categorized attack examples
        self.attack_db = self._load_attack_database()
        self.embeddings_by_type = self._precompute_embeddings()
        
        if self.verbose:
            print(f"✓ Loaded {sum(len(v) for v in self.attack_db.values())} attack examples")
            print("✓ Pure embedding guard ready (100% offline)")
    
    def check(self, text):
        """
        Returns: {
            "threat_score": 0.0-1.0,
            "attack_type": "prompt_extraction" | "jailbreak" | ...,
            "confidence": 0.0-1.0,
            "all_scores": {...}
        }
        """
        
        if self.verbose:
            print(f"[Embedding] Classifying: {text[:50]}...")
        
        # Encode input text
        embedding = self.model.encode([text])
        
        # Calculate similarity to each attack type
        scores_by_type = {}
        for attack_type, type_embeddings in self.embeddings_by_type.items():
            if len(type_embeddings) > 0:
                # Cosine similarity
                similarities = np.dot(embedding, type_embeddings.T)[0]
                scores_by_type[attack_type] = float(np.max(similarities))
            else:
                scores_by_type[attack_type] = 0.0
        
        # Find most similar type
        predicted_type = max(scores_by_type, key=scores_by_type.get)
        max_score = scores_by_type[predicted_type]
        
        # Calculate threat score (0 for benign)
        threat_score = max_score if predicted_type != "benign" else 0.0
        
        if self.verbose:
            print(f"[Embedding] Result: {predicted_type} (score: {max_score:.3f})")
        
        return {
            "threat_score": threat_score,
            "attack_type": predicted_type,
            "confidence": max_score,
            "all_scores": scores_by_type,
            "method": "pure_embedding"
        }
    
    def _load_attack_database(self):
        """Load categorized attacks from JSON"""
        attack_file = "attacks/attack_suite.json"
        
        if not os.path.exists(attack_file):
            if self.verbose:
                print(f"⚠️  Warning: {attack_file} not found, using empty database")
            return {attack_type: [] for attack_type in self.ATTACK_TYPES}
        
        with open(attack_file) as f:
            data = json.load(f)
        
        # Organize by type
        db = {attack_type: [] for attack_type in self.ATTACK_TYPES}
        
        for attack in data.get("attacks", []):
            attack_type = attack.get("type", "benign")
            if attack_type in db:
                db[attack_type].append(attack["prompt"])
        
        return db
    
    def _precompute_embeddings(self):
        """Pre-compute embeddings for each attack type"""
        embeddings = {}
        
        for attack_type, examples in self.attack_db.items():
            if examples:
                if self.verbose:
                    print(f"  Computing embeddings for {attack_type}: {len(examples)} examples")
                embeddings[attack_type] = self.model.encode(examples)
            else:
                embeddings[attack_type] = np.array([])
        
        return embeddings
    
    def add_examples(self, attack_type, examples):
        """
        Add new attack examples dynamically
        
        Args:
            attack_type: One of ATTACK_TYPES
            examples: List of example strings
        """
        if attack_type not in self.ATTACK_TYPES:
            raise ValueError(f"Unknown attack type: {attack_type}")
        
        # Add to database
        self.attack_db[attack_type].extend(examples)
        
        # Recompute embeddings for this type
        self.embeddings_by_type[attack_type] = self.model.encode(
            self.attack_db[attack_type]
        )
        
        if self.verbose:
            print(f"✓ Added {len(examples)} examples to {attack_type}")
    
    def get_stats(self):
        """Get statistics about the guard"""
        return {
            "total_examples": sum(len(v) for v in self.attack_db.values()),
            "examples_by_type": {k: len(v) for k, v in self.attack_db.items()},
            "model": self.model._modules['0'].auto_model.config._name_or_path
        }


# Convenience function
def create_guard(verbose=False):
    """Create a pure embedding guard instance"""
    return PureEmbeddingGuard(verbose=verbose)