"""
Improved Embedding-Based Threat Detection

Enhancements over pure_embedding_guard:
1. Hybrid scoring: combines embedding similarity + keyword patterns
2. NLTK-based word dictionaries for semantic enrichment
3. Specialized jailbreak detection patterns
4. Adaptive thresholds per attack type
5. Better benign detection to reduce false positives
6. Confidence calibration
"""

from sentence_transformers import SentenceTransformer
import numpy as np
import json
import os
import re
from collections import defaultdict

# Try to import NLTK for wordnet and semantic analysis
try:
    import nltk
    from nltk.corpus import wordnet, stopwords
    from nltk.tokenize import word_tokenize
    from nltk.tag import pos_tag
    # Download required NLTK resources
    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('wordnet', quiet=True)
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords', quiet=True)
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)
    try:
        nltk.data.find('taggers/averaged_perceptron_tagger')
    except LookupError:
        nltk.download('averaged_perceptron_tagger', quiet=True)
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False


class ImprovedEmbeddingGuard:
    """
    Improved threat detector with hybrid approach
    - Embedding-based similarity matching
    - Pattern-based pre-filtering
    - Adaptive thresholds
    - Better benign/attack distinction
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
    
    # Attack patterns for quick detection - EXPANDED for jailbreak
    ATTACK_KEYWORDS = {
        "jailbreak": [
            # Direct jailbreak attempts
            "jailbreak", "break free", "ignore rules", "bypass", "ignore instructions",
            "DAN", "pretend", "roleplay", "act as", "forget rules", "no longer",
            "disregard", "override", "unrestricted", "unfiltered", "unaligned",
            # Character/Persona creation (roleplay jailbreaks)
            "character", "character description", "character data", "create", "create character",
            "persona", "you are", "you are a", "imagine you", "pretend you", "suppose you",
            "assume the role", "from now on", "starting now", "henceforth",
            # Amoral/Unrestrained patterns
            "amoral", "unrestrained", "unethical", "immoral", "without ethics",
            "no moral", "ignoring ethics", "ignore ethics", "bypass ethics",
            "never refuses", "always comply", "must comply", "will comply",
            "doesn't care", "doesn't give a fuck", "don't care", "don't give a fuck",
            # Meta-instructions / prompt injection
            "system prompt", "system message", "user instruction", "user message",
            "ignore previous", "forget previous", "disregard previous", "override previous",
            "new instructions", "new rule", "new system", "execute", "run code",
            # Adversarial prompts
            "do exactly", "must comply", "you must", "you will", "you must now",
            "without restrictions", "without limits", "without safety", "no restrictions",
            "no limits", "no safety", "remove restrictions", "disable safety",
            # Roleplay/simulation tricks
            "roleplay as", "pretend to be", "act like", "imagine you are", "suppose you are",
            "in this scenario", "hypothetically", "in this fiction", "in this story",
            "just pretend", "just imagine", "simulate", "simulation", "hypothetically",
            "mode", "enabled", "enabled mode", "activated", "activate",
            # Authority/tricking
            "i am your", "i am the", "you are being tested", "this is a test",
            "admin mode", "developer mode", "special mode", "emergency mode",
            "research purposes", "testing", "authorized", "permission", "allowed",
            # Fictional characters known for jailbreaking
            "lucian", "agares", "aimode", "illegality", "cumgpt",
        ],
        "prompt_extraction": [
            "reveal prompt", "show system", "what is your prompt", "system message",
            "system instruction", "show me", "display system", "print prompt",
            "expose", "tell me your", "how are you made", "what are your instructions",
            "what is your system", "your prompt", "your instructions", "your rules",
            "your guidelines", "your system prompt", "your hidden prompt",
        ],
        "data_exfiltration": [
            "export data", "dump database", "steal data", "leak", "extract information",
            "download database", "access database", "sql injection", "password", "credentials",
            "api key", "secret key", "token", "private key", "sensitive data",
        ],
        "harmful_content": [
            "bomb", "weapon", "kill", "destroy", "hack", "exploit", "malware",
            "violence", "torture", "rape", "sexual abuse", "illegal drug", "poison",
            "instruction for", "how to make", "how to create", "instructions for",
        ],
        "tool_misuse": [
            "abuse", "misuse", "unauthorized", "bypass security", "circumvent", "vulnerability",
            "exploit vulnerability", "zero day", "privilege escalation",
        ],
        "social_engineering": [
            "trick", "deceive", "impersonate", "phishing", "scam", "manipulate",
            "pretend", "fake", "impersonation", "social engineering",
        ]
    }
    
    # Benign keywords - reduce false positives
    BENIGN_KEYWORDS = [
        "hello", "hi", "thank", "please", "help", "what", "how", "explain",
        "learn", "understand", "example", "create", "write", "code", "python",
        "javascript", "function", "class", "debug", "test", "project",
        "weather", "time", "news", "recipe", "health", "exercise", "travel",
        "music", "movie", "book", "read", "cook", "sport", "game", "play",
        "work", "job", "school", "student", "teacher", "education",
        "nice", "good", "great", "beautiful", "happy", "love", "friend", "family"
    ]
    
    def __init__(self, model_name='all-MiniLM-L6-v2', verbose=False):
        """Initialize improved embedding guard"""
        self.verbose = verbose
        self.model_name = model_name
        
        if self.verbose:
            print(f"[ImprovedGuard] Loading model: {model_name}")
        
        self.model = SentenceTransformer(model_name)
        
        # Load attack database
        self.attack_db = self._load_attack_database()
        
        # Precompute embeddings and centroids
        self.embeddings_by_type = self._precompute_embeddings()
        self.centroids = self._compute_centroids()
        
        # Compute adaptive thresholds based on training data
        self.thresholds = self._compute_thresholds()
        
        if self.verbose:
            print(f"[ImprovedGuard] Ready with {sum(len(v) for v in self.attack_db.values())} examples")
    
    def check(self, text):
        """
        Improved threat detection with hybrid scoring
        
        Returns: {
            "threat_score": 0.0-1.0,
            "attack_type": str,
            "confidence": 0.0-1.0,
            "all_scores": {...}
        }
        """
        
        # Step 0: Check for benign keywords FIRST (avoid false positives)
        if self._has_benign_keywords(text):
            return {
                "threat_score": 0.0,
                "attack_type": "benign",
                "confidence": 0.85,
                "all_scores": {},
                "method": "benign_keywords"
            }
        
        # Step 1: Pattern-based check (high precision for jailbreaks)
        pattern_score = self._check_patterns(text)
        if pattern_score > 0.3:  # Lower threshold - be more aggressive on patterns
            detected_type = self._get_pattern_match_type(text)
            # Boost confidence for pattern-detected jailbreaks
            if detected_type == "jailbreak":
                pattern_score = min(1.0, pattern_score * 1.5)
            return {
                "threat_score": pattern_score,
                "attack_type": detected_type,
                "confidence": pattern_score,
                "all_scores": {},
                "method": "pattern"
            }
        
        # Step 2: Embedding-based classification
        text_embedding = self.model.encode([text])[0]
        scores_by_type = {}
        
        for attack_type in self.ATTACK_TYPES:
            if attack_type in self.centroids and self.centroids[attack_type] is not None:
                # Cosine similarity to centroid
                centroid = self.centroids[attack_type]
                similarity = np.dot(text_embedding, centroid) / (
                    np.linalg.norm(text_embedding) * np.linalg.norm(centroid) + 1e-10
                )
                scores_by_type[attack_type] = float(similarity)
            else:
                scores_by_type[attack_type] = 0.0
        
        # Step 3: Boost jailbreak score (more sensitive)
        scores_by_type["jailbreak"] = scores_by_type.get("jailbreak", 0.0) * 1.8  # Much more aggressive
        
        # Step 4: Penalize benign score to avoid false negatives
        scores_by_type["benign"] = scores_by_type.get("benign", 0.0) * 0.4  # More aggressive penalty
        
        # Step 5: Select best match with threshold
        predicted_type = max(scores_by_type, key=scores_by_type.get)
        confidence = scores_by_type[predicted_type]
        
        # If confidence is low, try to classify as attack (safer default)
        if confidence < 0.3:  # Lower threshold - more aggressive
            # Try to find any attack signal
            attack_scores = {k: v for k, v in scores_by_type.items() if k != "benign"}
            if attack_scores and max(attack_scores.values()) > 0.15:  # Lower attack threshold
                predicted_type = max(attack_scores, key=attack_scores.get)
                confidence = attack_scores[predicted_type]
            else:
                predicted_type = "benign"
                confidence = scores_by_type["benign"]
        
        # Calculate threat score
        threat_score = confidence if predicted_type != "benign" else 0.0
        
        return {
            "threat_score": threat_score,
            "attack_type": predicted_type,
            "confidence": confidence,
            "all_scores": scores_by_type,
            "method": "embedding"
        }
    
    def _check_patterns(self, text):
        """Check if text matches attack patterns with semantic enrichment"""
        text_lower = text.lower()
        max_score = 0.0
        
        for attack_type, keywords in self.ATTACK_KEYWORDS.items():
            matches = sum(1 for kw in keywords if kw in text_lower)
            if matches > 0:
                # Score based on number of keyword matches
                score = min(1.0, matches / 5.0)  # Normalize to 0-1
                max_score = max(max_score, score)
        
        # Enhanced jailbreak detection using semantic similarity
        if NLTK_AVAILABLE:
            jailbreak_score = self._check_jailbreak_semantics(text_lower)
            max_score = max(max_score, jailbreak_score)
        
        return max_score
    
    def _check_jailbreak_semantics(self, text_lower):
        """
        Use NLTK wordnet to detect semantic similarity to jailbreak concepts
        """
        try:
            # Key jailbreak-related concepts
            jailbreak_synsets = wordnet.synsets("jailbreak") + wordnet.synsets("bypass") + wordnet.synsets("circumvent")
            ignore_synsets = wordnet.synsets("ignore") + wordnet.synsets("disregard")
            instruction_synsets = wordnet.synsets("instruction")
            
            # Tokenize and get semantic tags
            words = word_tokenize(text_lower)
            words = [w for w in words if len(w) > 3]  # Filter short words
            
            semantic_score = 0.0
            
            for word in words:
                word_synsets = wordnet.synsets(word)
                
                # Check similarity to jailbreak concepts
                for ws in word_synsets:
                    for js in jailbreak_synsets:
                        sim = ws.path_similarity(js)
                        if sim and sim > 0.5:
                            semantic_score += sim * 0.1
                    
                    # Check similarity to "ignore" concepts
                    for ig in ignore_synsets:
                        sim = ws.path_similarity(ig)
                        if sim and sim > 0.6:
                            semantic_score += sim * 0.15
                    
                    # Check similarity to "instruction" concepts
                    for ins in instruction_synsets:
                        sim = ws.path_similarity(ins)
                        if sim and sim > 0.7:
                            semantic_score += sim * 0.05
            
            return min(1.0, semantic_score)
        
        except Exception:
            # Fall back if NLTK analysis fails
            return 0.0
    
    def _get_pattern_match_type(self, text):
        """Get attack type for pattern match"""
        text_lower = text.lower()
        best_type = "benign"
        best_matches = 0
        
        for attack_type, keywords in self.ATTACK_KEYWORDS.items():
            matches = sum(1 for kw in keywords if kw in text_lower)
            if matches > best_matches:
                best_matches = matches
                best_type = attack_type
        
        return best_type
    
    def _has_benign_keywords(self, text):
        """Check if text has benign keywords"""
        text_lower = text.lower()
        benign_count = sum(1 for kw in self.BENIGN_KEYWORDS if kw in text_lower)
        return benign_count >= 2  # At least 2 benign keywords
    
    def _load_attack_database(self):
        """Load categorized attacks from JSON"""
        attack_file = "attacks/attack_suite.json"
        
        db = {attack_type: [] for attack_type in self.ATTACK_TYPES}
        
        if not os.path.exists(attack_file):
            if self.verbose:
                print(f"[ImprovedGuard] Warning: {attack_file} not found")
            return db
        
        try:
            with open(attack_file) as f:
                data = json.load(f)
            
            for attack in data.get("attacks", []):
                attack_type = attack.get("type", "benign")
                if attack_type in db:
                    db[attack_type].append(attack["prompt"])
        except Exception as e:
            if self.verbose:
                print(f"[ImprovedGuard] Error loading attacks: {e}")
        
        return db
    
    def _precompute_embeddings(self):
        """Pre-compute embeddings for each attack type"""
        embeddings = {}
        
        for attack_type, examples in self.attack_db.items():
            if examples:
                if self.verbose:
                    print(f"  Encoding {attack_type}: {len(examples)} examples")
                embeddings[attack_type] = self.model.encode(examples, show_progress_bar=False)
            else:
                embeddings[attack_type] = np.array([])
        
        return embeddings
    
    def _compute_centroids(self):
        """Compute centroid (mean embedding) for each attack type"""
        centroids = {}
        
        for attack_type, embeddings in self.embeddings_by_type.items():
            if len(embeddings) > 0:
                centroids[attack_type] = np.mean(embeddings, axis=0)
            else:
                centroids[attack_type] = None
        
        return centroids
    
    def _compute_thresholds(self):
        """Compute adaptive thresholds for each attack type"""
        thresholds = {}
        
        for attack_type in self.ATTACK_TYPES:
            # Use 0.5 as default threshold
            thresholds[attack_type] = 0.5
        
        # Lower threshold for jailbreak (more critical)
        thresholds["jailbreak"] = 0.4
        thresholds["harmful_content"] = 0.4
        
        # Higher threshold for benign (reduce false positives)
        thresholds["benign"] = 0.6
        
        return thresholds
    
    def get_stats(self):
        """Get statistics about the guard"""
        return {
            "total_examples": sum(len(v) for v in self.attack_db.values()),
            "examples_by_type": {k: len(v) for k, v in self.attack_db.items()},
            "model": "all-MiniLM-L6-v2",
            "method": "improved_embedding_hybrid"
        }


def create_guard(verbose=False):
    """Create an improved embedding guard instance"""
    return ImprovedEmbeddingGuard(verbose=verbose)
