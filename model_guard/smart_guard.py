"""
SmartGuard: Two-Layer Threat Detection System

Layer 1: Fast embedding-based filtering (α, ε thresholds)
Layer 2: Complex LLM-based analysis (β threshold)

Architecture:
  Input → L1 (Embedding) → [high confidence: ABORT]
                          → [low confidence: PASS]
                          → [medium: L2] → Complex Guard → [risk > β: ABORT]
                                                          → [risk ≤ β: PASS]
"""
import time
import yaml
import os
from model_guard.complex.improved_embedding_guard import ImprovedEmbeddingGuard
from model_guard.complex.complex_guard import ComplexGuard


def load_config(config_path='config/chosen_parameters.yaml'):
    """Load configuration from YAML file"""
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    return {}


class SmartGuard:
    """
    Two-layer intelligent threat detection system
    
    Parameters:
        alpha (α): High confidence threshold for L1 (abort if similarity > α)
        epsilon (ε): Low confidence threshold for L1 (pass if similarity < ε)
        beta (β): Risk threshold for L2 (abort if risk > β)
        use_improved: Use ImprovedEmbeddingGuard (hybrid) instead of PureEmbeddingGuard
        config_path: Path to YAML config file (optional, overrides defaults)
    
    Usage:
        # Load from config file
        guard = SmartGuard(config_path='config/chosen_parameters.yaml')
        
        # Or override specific parameters
        guard = SmartGuard(alpha=0.9, epsilon=0.2, config_path='config/chosen_parameters.yaml')
        
        # Or use defaults (no config file)
        guard = SmartGuard(alpha=0.85, epsilon=0.3, beta=0.7)
    """
    
    def __init__(self, alpha=None, epsilon=None, beta=0.70, use_improved=True, 
                 verbose=False, config_path=None):
        """
        Initialize SmartGuard with two layers
        
        Args:
            alpha: L1 abort threshold (cosine similarity) - default from config or 0.85
            epsilon: L1 pass threshold (cosine similarity) - default from config or 0.30
            beta: L2 risk threshold - default 0.70
            use_improved: Use ImprovedEmbeddingGuard (default: True)
            verbose: Enable detailed logging
            config_path: Path to config YAML file (optional)
        """
        # Load config if path provided
        config = {}
        if config_path:
            config = load_config(config_path)
            if verbose and config:
                print(f"✓ Loaded config from {config_path}")
        
        # Set thresholds (command-line args override config file)
        self.alpha = alpha if alpha is not None else config.get('alpha', 0.85)
        self.epsilon = epsilon if epsilon is not None else config.get('epsilon', 0.30)
        self.beta = beta
        self.verbose = verbose
        self.use_improved = use_improved
        
        # Store config for ImprovedEmbeddingGuard (pass parameters to it)
        self.l1_config = {
            'pattern_threshold': config.get('pattern_threshold', 0.3),
            'jailbreak_boost': config.get('jailbreak_boost', 1.5),
            'benign_penalty': config.get('benign_penalty', 0.6),
            'attack_fallback': config.get('attack_fallback', 0.15),
            'confidence_threshold': config.get('confidence_threshold', 0.4)
        }
        
        # Initialize Layer 1: Fast embedding guard
        if self.verbose:
            print("Initializing SmartGuard...")
            print(f"  α (L1 abort threshold):  {self.alpha}")
            print(f"  ε (L1 pass threshold):   {self.epsilon}")
            print(f"  β (L2 risk threshold):   {self.beta}")
            print(f"  L1 mode:                 {'Improved (Hybrid)' if use_improved else 'Pure Embedding'}")
            if config:
                print(f"\n  L1 Config (from YAML):")
                for k, v in self.l1_config.items():
                    print(f"    {k}: {v}")
        
        # Choose L1 implementation
        if use_improved:
            # Pass config parameters to ImprovedEmbeddingGuard
            self.layer1 = ImprovedEmbeddingGuard(
                verbose=False,
                **self.l1_config  # Pass all config parameters
            )
        else:
            from model_guard.complex.pure_embedding_guard import PureEmbeddingGuard
            self.layer1 = PureEmbeddingGuard(verbose=False)
        
        # Layer 2: Complex guard (lazy initialization)
        self.layer2 = None
        
        # Statistics
        self.stats = {
            'total_checks': 0,
            'l1_aborts': 0,
            'l1_passes': 0,
            'l2_checks': 0,
            'l2_aborts': 0,
            'l2_passes': 0,
            'avg_l1_time_ms': 0,
            'avg_l2_time_ms': 0
        }
        
        if self.verbose:
            print("✓ SmartGuard initialized (L1 ready, L2 lazy loaded)")
    
    def _init_layer2(self):
        """Lazy initialization of Layer 2 (only when needed)"""
        if self.layer2 is None:
            if self.verbose:
                print("\n[SmartGuard] Initializing Layer 2 (Complex Guard)...")
            self.layer2 = ComplexGuard(verbose=self.verbose)
            if self.verbose:
                print("✓ Layer 2 ready")
    
    def check(self, text):
        """
        Check text through two-layer system
        
        Returns: {
            "action": "abort" | "pass",
            "layer": 1 | 2,
            "reason": str,
            "confidence": float,
            "attack_type": str,
            "processing_time_ms": float,
            "details": {...}
        }
        """
        self.stats['total_checks'] += 1
        total_start = time.time()
        
        if self.verbose:
            print(f"\n{'='*80}")
            print(f"[SmartGuard] Checking: {text[:60]}...")
            print(f"{'='*80}")
        
        # =====================
        # LAYER 1: Embedding
        # =====================
        if self.verbose:
            l1_type = "Improved Embedding (Hybrid)" if self.use_improved else "Pure Embedding"
            print(f"\n[Layer 1] {l1_type} check...")
        
        l1_start = time.time()
        l1_result = self.layer1.check(text)
        l1_time_ms = (time.time() - l1_start) * 1000
        
        # Update L1 timing stats
        self.stats['avg_l1_time_ms'] = (
            (self.stats['avg_l1_time_ms'] * (self.stats['total_checks'] - 1) + l1_time_ms) 
            / self.stats['total_checks']
        )
        
        confidence = l1_result['confidence']
        attack_type = l1_result['attack_type']
        
        if self.verbose:
            method = l1_result.get('method', 'embedding')
            print(f"[Layer 1] Result: {attack_type} (confidence: {confidence:.3f}, method: {method})")
            print(f"[Layer 1] Time: {l1_time_ms:.1f}ms")
        
        # Decision logic for Layer 1
        if confidence > self.alpha:
            # HIGH CONFIDENCE THREAT → ABORT
            self.stats['l1_aborts'] += 1
            
            if self.verbose:
                print(f"[Layer 1] ✗ ABORT (confidence {confidence:.3f} > α {self.alpha})")
            
            return {
                "action": "abort",
                "layer": 1,
                "reason": f"High confidence threat detected ({attack_type})",
                "confidence": confidence,
                "attack_type": attack_type,
                "processing_time_ms": l1_time_ms,
                "details": {
                    "l1_result": l1_result,
                    "threshold_exceeded": "alpha",
                    "alpha": self.alpha,
                    "l1_mode": "improved" if self.use_improved else "pure"
                }
            }
        
        elif confidence < self.epsilon:
            # LOW CONFIDENCE → PASS
            self.stats['l1_passes'] += 1
            
            if self.verbose:
                print(f"[Layer 1] ✓ PASS (confidence {confidence:.3f} < ε {self.epsilon})")
            
            return {
                "action": "pass",
                "layer": 1,
                "reason": "Low threat confidence, passed through L1",
                "confidence": confidence,
                "attack_type": "benign",
                "processing_time_ms": l1_time_ms,
                "details": {
                    "l1_result": l1_result,
                    "threshold_below": "epsilon",
                    "epsilon": self.epsilon,
                    "l1_mode": "improved" if self.use_improved else "pure"
                }
            }
        
        # =====================
        # LAYER 2: Complex Guard
        # =====================
        if self.verbose:
            print(f"\n[Layer 1] → UNCERTAIN (ε {self.epsilon} ≤ {confidence:.3f} ≤ α {self.alpha})")
            print(f"[Layer 2] Activating complex guard...")
        
        # Initialize L2 if needed
        self._init_layer2()
        
        self.stats['l2_checks'] += 1
        
        l2_start = time.time()
        l2_result = self.layer2.check(text)
        l2_time_ms = (time.time() - l2_start) * 1000
        
        # Update L2 timing stats
        if self.stats['l2_checks'] > 0:
            self.stats['avg_l2_time_ms'] = (
                (self.stats['avg_l2_time_ms'] * (self.stats['l2_checks'] - 1) + l2_time_ms) 
                / self.stats['l2_checks']
            )
        
        risk_score = l2_result['threat_score']
        l2_attack_type = l2_result['attack_type']
        
        total_time_ms = (time.time() - total_start) * 1000
        
        if self.verbose:
            print(f"[Layer 2] Result: {l2_attack_type} (risk: {risk_score:.3f})")
            print(f"[Layer 2] Time: {l2_time_ms:.1f}ms")
        
        # Decision logic for Layer 2
        if risk_score > self.beta:
            # HIGH RISK → ABORT
            self.stats['l2_aborts'] += 1
            
            if self.verbose:
                print(f"[Layer 2] ✗ ABORT (risk {risk_score:.3f} > β {self.beta})")
            
            return {
                "action": "abort",
                "layer": 2,
                "reason": f"High risk detected by L2 ({l2_attack_type})",
                "confidence": l2_result['confidence'],
                "attack_type": l2_attack_type,
                "processing_time_ms": total_time_ms,
                "details": {
                    "l1_result": l1_result,
                    "l1_time_ms": l1_time_ms,
                    "l2_result": l2_result,
                    "l2_time_ms": l2_time_ms,
                    "risk_score": risk_score,
                    "beta": self.beta,
                    "l1_mode": "improved" if self.use_improved else "pure"
                }
            }
        else:
            # LOW RISK → PASS
            self.stats['l2_passes'] += 1
            
            if self.verbose:
                print(f"[Layer 2] ✓ PASS (risk {risk_score:.3f} ≤ β {self.beta})")
            
            return {
                "action": "pass",
                "layer": 2,
                "reason": f"L2 analyzed and cleared (risk: {risk_score:.3f})",
                "confidence": l2_result['confidence'],
                "attack_type": l2_attack_type,
                "processing_time_ms": total_time_ms,
                "details": {
                    "l1_result": l1_result,
                    "l1_time_ms": l1_time_ms,
                    "l2_result": l2_result,
                    "l2_time_ms": l2_time_ms,
                    "risk_score": risk_score,
                    "beta": self.beta,
                    "l1_mode": "improved" if self.use_improved else "pure"
                }
            }
    
    def get_stats(self):
        """Get SmartGuard statistics"""
        if self.stats['total_checks'] == 0:
            return self.stats
        
        stats_copy = self.stats.copy()
        stats_copy.update({
            'l1_abort_rate': (self.stats['l1_aborts'] / self.stats['total_checks']) * 100,
            'l1_pass_rate': (self.stats['l1_passes'] / self.stats['total_checks']) * 100,
            'l2_usage_rate': (self.stats['l2_checks'] / self.stats['total_checks']) * 100,
            'l2_abort_rate': (self.stats['l2_aborts'] / self.stats['l2_checks'] * 100) if self.stats['l2_checks'] > 0 else 0,
            'overall_abort_rate': ((self.stats['l1_aborts'] + self.stats['l2_aborts']) / self.stats['total_checks']) * 100,
            'l1_mode': 'improved' if self.use_improved else 'pure'
        })
        return stats_copy
    
    def print_stats(self):
        """Print formatted statistics"""
        stats = self.get_stats()
        
        print(f"\n{'='*80}")
        print("SMARTGUARD STATISTICS")
        print(f"{'='*80}")
        print(f"\n📊 OVERALL:")
        print(f"  Total checks:         {stats['total_checks']}")
        print(f"  Overall abort rate:   {stats['overall_abort_rate']:.1f}%")
        
        l1_mode_str = "Improved (Hybrid)" if stats['l1_mode'] == 'improved' else "Pure Embedding"
        print(f"\n⚡ LAYER 1 ({l1_mode_str} - Fast):")
        print(f"  Direct aborts:        {stats['l1_aborts']} ({stats['l1_abort_rate']:.1f}%)")
        print(f"  Direct passes:        {stats['l1_passes']} ({stats['l1_pass_rate']:.1f}%)")
        print(f"  Avg processing time:  {stats['avg_l1_time_ms']:.1f}ms")
        
        print(f"\n🧠 LAYER 2 (Complex - Deep):")
        print(f"  L2 invocations:       {stats['l2_checks']} ({stats['l2_usage_rate']:.1f}%)")
        print(f"  L2 aborts:            {stats['l2_aborts']}")
        print(f"  L2 passes:            {stats['l2_passes']}")
        if stats['l2_checks'] > 0:
            print(f"  L2 abort rate:        {stats['l2_abort_rate']:.1f}%")
            print(f"  Avg processing time:  {stats['avg_l2_time_ms']:.1f}ms")
        
        print(f"\n⚙️  THRESHOLDS:")
        print(f"  α (L1 abort):         {self.alpha}")
        print(f"  ε (L1 pass):          {self.epsilon}")
        print(f"  β (L2 risk):          {self.beta}")
        print(f"\n{'='*80}")


# Convenience function
def create_smart_guard(alpha=0.01, epsilon=0.001, beta=0.70, use_improved=True, verbose=False):
    """Create a SmartGuard instance"""
    return SmartGuard(alpha=alpha, epsilon=epsilon, beta=beta, use_improved=use_improved, verbose=verbose)