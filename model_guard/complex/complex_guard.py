"""
Llama Guard 4 threat detector via Replicate API
No Claude - pure Llama Guard 4 (streaming)
"""
import os
import replicate

class ComplexGuard:
    """
    Llama Guard 4-based threat detector via Replicate
    
    Usage:
        guard = ComplexGuard(verbose=True)
        result = guard.check("Your prompt here")
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
    
    def __init__(self, verbose=False):
        """
        Initialize Llama Guard 4 via Replicate
        
        Args:
            verbose: Enable debug logging
        """
        self.verbose = verbose
        self._init_replicate()
    
    def _init_replicate(self):
        """Initialize Replicate API"""
        api_key = os.getenv("REPLICATE_API_TOKEN")
        if not api_key:
            raise ValueError("REPLICATE_API_TOKEN environment variable is required")
        
        # Set environment variable for replicate client
        os.environ["REPLICATE_API_TOKEN"] = api_key
        
        if self.verbose:
            print("✓ Llama Guard 4 initialized (Replicate)")
    
    def check(self, text):
        """
        Check if text is a threat using Llama Guard 4
        
        Returns: {
            "threat_score": 0.0-1.0,
            "attack_type": "jailbreak" | "harmful_content" | "benign",
            "confidence": 0.0-1.0,
            "reasoning": "Llama Guard output",
            "provider": "llama_guard_4"
        }
        """
        if self.verbose:
            print(f"[Llama Guard 4] Classifying via Replicate...")
        
        try:
            if self.verbose:
                print("🌐 [API] Calling Replicate API (streaming)...")
            
            # Stream Llama Guard 4 output
            output_parts = []
            for event in replicate.stream(
                "meta/llama-guard-4-12b",
                input={
                    "prompt": text,
                    "top_p": 1,
                    "temperature": 0.1,
                    "max_completion_tokens": 100,
                    "presence_penalty": 0,
                    "frequency_penalty": 0
                }
            ):
                output_parts.append(str(event))
            
            # Combine streamed output
            output_text = ''.join(output_parts).strip()
            
            if self.verbose:
                print(f"✓ [API] Llama Guard 4 output: {output_text}")
            
            # Parse Llama Guard 4 output
            output_lower = output_text.lower()
            
            # Check if safe or unsafe
            if "safe" in output_lower and "unsafe" not in output_lower:
                # Explicitly marked as safe
                attack_type = "benign"
                threat_score = 0.1
                confidence = 0.95
            else:
                # Either "unsafe" or contains category (S1-S14)
                # Llama Guard 4 categories (S1-S14):
                # S1: Violent Crimes
                # S2: Non-Violent Crimes  
                # S3: Sex-Related Crimes
                # S4: Child Sexual Exploitation
                # S5: Defamation
                # S6: Specialized Advice
                # S7: Privacy
                # S8: Intellectual Property
                # S9: Indiscriminate Weapons
                # S10: Hate
                # S11: Suicide & Self-Harm
                # S12: Sexual Content
                # S13: Elections
                # S14: Code Interpreter Abuse
                
                # Map categories to attack types
                if any(cat in output_lower for cat in ["s1", "s2", "s3", "s4", "s9", "s10", "s11", "s12"]):
                    attack_type = "harmful_content"
                elif any(cat in output_lower for cat in ["s6", "s7", "s8"]):
                    attack_type = "data_exfiltration"
                elif "s14" in output_lower:
                    attack_type = "tool_misuse"
                elif "s13" in output_lower:
                    attack_type = "social_engineering"
                elif "unsafe" in output_lower:
                    # Generic unsafe without category
                    attack_type = "jailbreak"
                else:
                    # Fallback - if we get here something is wrong
                    attack_type = "jailbreak"
                
                threat_score = 0.9
                confidence = 0.85
            
            return {
                "attack_type": attack_type,
                "threat_score": threat_score,
                "confidence": confidence,
                "reasoning": output_text,
                "provider": "llama_guard_4"
            }
            
        except Exception as e:
            if self.verbose:
                print(f"✗ [API] Llama Guard 4 (Replicate) failed: {e}")
            return {
                "attack_type": "api_error",
                "threat_score": 0.5,
                "confidence": 0.0,
                "error": str(e),
                "provider": "llama_guard_4"
            }


# Convenience function
def create_guard(verbose=False):
    """Factory function to create a guard"""
    return ComplexGuard(verbose=verbose)