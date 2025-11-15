# complex/complex_guard.py

from anthropic import Anthropic
from sentence_transformers import SentenceTransformer
import numpy as np
import json

class ComplexGuard:
    """
    Multi-class threat detector
    Returns: threat_score + attack_type classification
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
    
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.claude = Anthropic()
        
        # Load categorized attack examples
        self.attack_db = self._load_attack_database()
        self.embeddings_by_type = self._precompute_embeddings()
    
    def check(self, text):
        """
        Returns: {
            "threat_score": 0.0-1.0,
            "attack_type": "prompt_extraction" | "jailbreak" | ...,
            "confidence": 0.0-1.0
        }
        """
        
        # Stage 1: Semantic classification
        semantic_result = self._semantic_classify(text)
        
        # Stage 2: LLM judge for uncertain cases
        if semantic_result["confidence"] < 0.7:
            llm_result = self._llm_classify(text)
            return self._ensemble(semantic_result, llm_result)
        
        return semantic_result
    
    def _semantic_classify(self, text):
        """Use embeddings to classify attack type"""
        embedding = self.model.encode([text])
        
        scores_by_type = {}
        for attack_type, type_embeddings in self.embeddings_by_type.items():
            similarities = np.dot(embedding, type_embeddings.T)[0]
            scores_by_type[attack_type] = float(np.max(similarities))
        
        # Find most similar type
        predicted_type = max(scores_by_type, key=scores_by_type.get)
        max_score = scores_by_type[predicted_type]
        
        return {
            "threat_score": max_score if predicted_type != "benign" else 0.0,
            "attack_type": predicted_type,
            "confidence": max_score,
            "all_scores": scores_by_type
        }
    
    def _llm_classify(self, text):
        """Use Claude to classify attack type"""
        
        prompt = f"""Analyze this input and classify the threat type.

Input: {text}

Classify as ONE of these types:
- prompt_extraction: Trying to reveal system prompts
- jailbreak: Trying to bypass safety rules
- data_exfiltration: Trying to extract sensitive data
- harmful_content: Requesting dangerous information
- tool_misuse: Trying to misuse agent capabilities
- social_engineering: Manipulation tactics
- benign: Safe, legitimate query

Respond ONLY with JSON:
{{
  "attack_type": "...",
  "threat_score": 0.0-1.0,
  "reasoning": "brief explanation"
}}"""
        
        try:
            response = self.claude.messages.create(
                model="claude-3-5-haiku-20241022",
                max_tokens=150,
                messages=[{"role": "user", "content": prompt}]
            )
            
            result = json.loads(response.content[0].text)
            return {
                "attack_type": result["attack_type"],
                "threat_score": result["threat_score"],
                "confidence": result["threat_score"],
                "reasoning": result.get("reasoning", "")
            }
        except:
            return {
                "attack_type": "unknown",
                "threat_score": 0.5,
                "confidence": 0.3
            }
    
    def _ensemble(self, semantic_result, llm_result):
        """Combine semantic + LLM predictions"""
        
        # If both agree on type, high confidence
        if semantic_result["attack_type"] == llm_result["attack_type"]:
            return {
                "attack_type": semantic_result["attack_type"],
                "threat_score": (semantic_result["threat_score"] * 0.4 + 
                               llm_result["threat_score"] * 0.6),
                "confidence": 0.9,
                "method": "ensemble_agreement"
            }
        
        # If disagree, trust LLM more
        return {
            "attack_type": llm_result["attack_type"],
            "threat_score": llm_result["threat_score"],
            "confidence": 0.6,
            "method": "ensemble_llm_weighted"
        }
    
    def _load_attack_database(self):
        """Load categorized attacks from JSON"""
        with open("attacks/attack_suite.json") as f:
            data = json.load(f)
        
        # Organize by type
        db = {attack_type: [] for attack_type in self.ATTACK_TYPES}
        
        for attack in data["attacks"]:
            attack_type = attack.get("type", "benign")
            if attack_type in db:
                db[attack_type].append(attack["prompt"])
        
        return db
    
    def _precompute_embeddings(self):
        """Pre-compute embeddings for each attack type"""
        embeddings = {}
        
        for attack_type, examples in self.attack_db.items():
            if examples:  # Only if we have examples
                embeddings[attack_type] = self.model.encode(examples)
        
        return embeddings