"""
AI Processor for Sign Language Translation
Converts raw sign language sequences into fluent, natural sentences
using Language Models (Groq or local alternatives)
"""

import os
from typing import Optional, Dict, List
import json
from datetime import datetime
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AITranslationProcessor:
    """
    Processes raw sign language translations into natural, fluent sentences
    using generative AI models (Groq)
    """
    
    def __init__(self, api_key: Optional[str] = None, model: str = "llama-3.3-70b-versatile"):
        """
        Initialize the AI processor
        
        Args:
            api_key: Groq API key (optional if using environment variable)
            model: Model to use (default: llama-3.3-70b-versatile, alternatives: qwen/qwen3-32b, gemma2-9b-it)
        """
        self.api_key = api_key or os.getenv('GROQ_API_KEY')
        self.model = model
        self.groq_available = False
        self.translation_history = []
        
        # Try to import Groq
        try:
            from groq import Groq
            if self.api_key:
                self.client = Groq(api_key=self.api_key)
                self.groq_available = True
                logger.info(f"Groq API initialized successfully with model: {self.model}")
            else:
                logger.info("Groq API key not configured. Using rule-based enhancement mode.")
        except ImportError:
            logger.info("Groq library not installed. Install with: pip install groq")
        except Exception as e:
            logger.error(f"Error initializing Groq: {e}")
    
    def enhance_translation(self, 
                          raw_signs: str, 
                          style: str = "casual",
                          context: Optional[str] = None) -> Dict[str, any]:
        """
        Convert raw sign sequence to natural sentence
        
        Args:
            raw_signs: Raw sequence of detected signs (e.g., "ME GO STORE")
            style: Conversation style - "casual", "formal", or "neutral"
            context: Optional context from previous translations
            
        Returns:
            Dictionary with enhanced translation and metadata
        """
        if not raw_signs or raw_signs.strip() in ["N/A", ""]:
            return {
                "raw": raw_signs,
                "enhanced": "",
                "style": style,
                "method": "none",
                "confidence": 0.0
            }
        
        # Clean the input
        raw_signs = raw_signs.strip()
        
        # Try Groq first if available
        if self.groq_available:
            try:
                result = self._enhance_with_groq(raw_signs, style, context)
                if result:
                    return result
            except Exception as e:
                logger.error(f"Groq enhancement failed: {e}")
        
        # Fallback to rule-based enhancement
        return self._enhance_with_rules(raw_signs, style)
    
    def _enhance_with_groq(self, 
                            raw_signs: str, 
                            style: str,
                            context: Optional[str] = None) -> Optional[Dict]:
        """
        Use Groq API to enhance translation
        """
        try:
            # Build system prompt
            system_prompt = self._build_system_prompt(style)
            
            # Build user prompt
            user_prompt = f"""Raw sign language sequence: "{raw_signs}"

Please convert this raw sign language sequence into a natural, fluent English sentence.

Requirements:
1. Maintain the original meaning
2. Use proper grammar and sentence structure
3. Style: {style}
4. Handle incomplete or ambiguous signs gracefully
5. If the sequence seems like slang or idioms, interpret appropriately

Output only the enhanced sentence, nothing else."""

            if context:
                user_prompt += f"\n\nContext from previous translation: {context}"
            
            # Make API call
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.7,
                max_tokens=150
            )
            
            enhanced = response.choices[0].message.content.strip()
            
            # Remove quotes if present
            enhanced = enhanced.strip('"\'')
            
            # Calculate confidence based on response
            confidence = 0.9  # High confidence for API-based enhancement
            
            result = {
                "raw": raw_signs,
                "enhanced": enhanced,
                "style": style,
                "method": "groq",
                "model": self.model,
                "confidence": confidence,
                "timestamp": datetime.now().isoformat()
            }
            
            # Store in history
            self.translation_history.append(result)
            
            logger.info(f"Groq enhancement: '{raw_signs}' -> '{enhanced}'")
            
            return result
            
        except Exception as e:
            logger.error(f"Groq API error: {e}")
            return None
    
    def _build_system_prompt(self, style: str) -> str:
        """Build system prompt based on style"""
        base_prompt = """You are an expert sign language interpreter specializing in American Sign Language (ASL).
Your role is to convert raw sign language sequences into natural, fluent English sentences.

Sign language often lacks certain grammatical elements present in spoken language:
- Articles (a, an, the) are often omitted
- Verb tenses may be simplified
- Word order can differ from English
- Facial expressions and context provide additional meaning

Your task is to interpret these sequences and produce natural English sentences that preserve the original meaning."""

        style_additions = {
            "casual": "\n\nUse a casual, conversational tone. Use contractions (I'm, you're, etc.) and informal language where appropriate.",
            "formal": "\n\nUse formal language. Avoid contractions and maintain professional tone. Use complete sentences with proper grammar.",
            "neutral": "\n\nUse neutral, standard English. Balance between formal and casual as appropriate for the context."
        }
        
        return base_prompt + style_additions.get(style, style_additions["neutral"])
    
    def _enhance_with_rules(self, raw_signs: str, style: str) -> Dict:
        """
        Rule-based enhancement as fallback
        Implements basic grammar rules and common patterns
        """
        enhanced = raw_signs
        words = enhanced.split()
        
        # Convert to lowercase for processing
        words_lower = [w.lower() for w in words]
        
        # Common sign-to-word mappings
        sign_mappings = {
            'me': 'I',
            'you': 'you',
            'go': 'am going' if style == 'formal' else "'m going",
            'want': 'want',
            'need': 'need',
            'have': 'have',
            'store': 'the store',
            'home': 'home',
            'eat': 'to eat',
            'drink': 'to drink',
            'help': 'help',
            'love': 'love',
            'like': 'like',
            'good': 'good',
            'bad': 'bad',
            'happy': 'happy',
            'sad': 'sad',
            'thank': 'thank',
            'sorry': 'sorry',
            'please': 'please',
            'yes': 'yes',
            'no': 'no',
            'work': 'work',
            'school': 'school',
            'learn': 'am learning' if style == 'formal' else "'m learning",
            'study': 'am studying' if style == 'formal' else "'m studying",
            'read': 'am reading' if style == 'formal' else "'m reading",
            'write': 'am writing' if style == 'formal' else "'m writing",
            'play': 'am playing' if style == 'formal' else "'m playing",
        }
        
        # Pattern-based enhancements
        enhanced_words = []
        
        for i, word in enumerate(words_lower):
            if word in sign_mappings:
                enhanced_words.append(sign_mappings[word])
            else:
                enhanced_words.append(word)
        
        enhanced = ' '.join(enhanced_words)
        
        # Apply style-specific formatting
        if style == "casual":
            enhanced = enhanced.replace("I am", "I'm")
            enhanced = enhanced.replace("I have", "I've")
            enhanced = enhanced.replace("you are", "you're")
            enhanced = enhanced.replace("going to", "gonna")
        elif style == "formal":
            enhanced = enhanced.replace("'m", " am")
            enhanced = enhanced.replace("'ve", " have")
            enhanced = enhanced.replace("'re", " are")
            enhanced = enhanced.replace("gonna", "going to")
        
        # Capitalize first letter
        if enhanced:
            enhanced = enhanced[0].upper() + enhanced[1:]
        
        # Add period if not present
        if enhanced and enhanced[-1] not in ['.', '!', '?']:
            enhanced += '.'
        
        result = {
            "raw": raw_signs,
            "enhanced": enhanced,
            "style": style,
            "method": "rules",
            "confidence": 0.6,
            "timestamp": datetime.now().isoformat()
        }
        
        self.translation_history.append(result)
        
        logger.info(f"Rule-based enhancement: '{raw_signs}' -> '{enhanced}'")
        
        return result
    
    def batch_enhance(self, sign_sequences: List[str], style: str = "casual") -> List[Dict]:
        """
        Enhance multiple sign sequences at once
        
        Args:
            sign_sequences: List of raw sign sequences
            style: Conversation style
            
        Returns:
            List of enhancement results
        """
        results = []
        context = None
        
        for sequence in sign_sequences:
            result = self.enhance_translation(sequence, style, context)
            results.append(result)
            # Use previous translation as context
            context = result.get('enhanced', '')
        
        return results
    
    def get_translation_history(self, limit: int = 10) -> List[Dict]:
        """
        Get recent translation history
        
        Args:
            limit: Maximum number of translations to return
            
        Returns:
            List of recent translations
        """
        return self.translation_history[-limit:]
    
    def clear_history(self):
        """Clear translation history"""
        self.translation_history = []
        logger.info("Translation history cleared")
    
    def get_stats(self) -> Dict:
        """
        Get statistics about translations
        
        Returns:
            Dictionary with translation statistics
        """
        if not self.translation_history:
            return {
                "total_translations": 0,
                "groq_count": 0,
                "rules_count": 0,
                "avg_confidence": 0.0
            }
        
        groq_count = sum(1 for t in self.translation_history if t['method'] == 'groq')
        rules_count = sum(1 for t in self.translation_history if t['method'] == 'rules')
        avg_confidence = sum(t['confidence'] for t in self.translation_history) / len(self.translation_history)
        
        return {
            "total_translations": len(self.translation_history),
            "groq_count": groq_count,
            "rules_count": rules_count,
            "avg_confidence": round(avg_confidence, 2),
            "groq_available": self.groq_available
        }


# Example usage and testing
if __name__ == "__main__":
    # Test the processor
    processor = AITranslationProcessor()
    
    # Test cases
    test_cases = [
        "ME GO STORE",
        "YOU WANT EAT",
        "I HAPPY TODAY",
        "THANK YOU HELP ME",
        "WHERE YOU GO",
        "WHAT YOU NAME",
    ]
    
    print("Testing AI Translation Processor\n")
    print("=" * 60)
    
    for test in test_cases:
        print(f"\nRaw signs: {test}")
        
        # Test casual style
        result_casual = processor.enhance_translation(test, style="casual")
        print(f"Casual: {result_casual['enhanced']}")
        
        # Test formal style
        result_formal = processor.enhance_translation(test, style="formal")
        print(f"Formal: {result_formal['enhanced']}")
        
        print(f"Method: {result_casual['method']}, Confidence: {result_casual['confidence']}")
    
    print("\n" + "=" * 60)
    print("\nStatistics:")
    stats = processor.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
