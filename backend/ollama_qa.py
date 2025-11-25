"""
Ollama Q&A Processor for SignSpeak
Handles question-answering using local Ollama models with Groq fallback
"""

import requests
import json
import logging
import os
from typing import Dict, List, Optional
from datetime import datetime

class OllamaQAProcessor:
    """
    Handles Q&A using local Ollama models.
    Allows users to ask questions in sign language and get intelligent answers.
    """
    
    def __init__(self, base_url: str = "http://localhost:11434", model: str = "llama2", groq_api_key: Optional[str] = None):
        """
        Initialize Ollama Q&A processor with Groq fallback
        
        Args:
            base_url: Ollama server URL (default: http://localhost:11434)
            model: Model to use (default: llama2, alternatives: mistral, phi, gemma)
            groq_api_key: Groq API key for fallback (optional, reads from env if not provided)
        """
        self.base_url = base_url
        self.model = model
        self.ollama_available = False
        self.groq_available = False
        self.conversation_history: List[Dict] = []
        self.max_history = 10
        
        # Initialize Groq client for fallback
        self.groq_api_key = groq_api_key or os.getenv('GROQ_API_KEY')
        if self.groq_api_key:
            try:
                from groq import Groq
                self.groq_client = Groq(api_key=self.groq_api_key)
                self.groq_available = True
                logging.info("✅ Groq fallback enabled")
            except Exception as e:
                logging.warning(f"⚠️  Groq fallback not available: {e}")
        
        # Check if Ollama is available
        self._check_ollama_availability()
    
    def _is_statement(self, text: str) -> bool:
        """
        Check if the input is a statement rather than a question
        
        Args:
            text: Input text to check
            
        Returns:
            True if it's a statement, False if it's a question
        """
        # Remove extra spaces
        text = text.strip()
        
        # Check if it ends with question mark
        if text.endswith('?'):
            return False
        
        # Check if it starts with question words
        question_words = ['what', 'why', 'how', 'when', 'where', 'who', 'which', 'whose', 'whom', 
                         'can', 'could', 'would', 'should', 'is', 'are', 'do', 'does', 'did']
        first_word = text.lower().split()[0] if text else ""
        
        if first_word in question_words:
            return False
        
        # If none of the above, it's likely a statement
        return True
    
    def _clean_thinking_tags(self, text: str) -> str:
        """
        Remove <think>...</think> tags and their content from response
        This is specific to some models like deepseek-r1 that show reasoning
        
        Args:
            text: Response text that may contain thinking tags
            
        Returns:
            Cleaned text without thinking process
        """
        import re
        
        # Remove everything between <think> and </think> including the tags
        cleaned = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
        
        # Clean up extra whitespace
        cleaned = re.sub(r'\n\s*\n', '\n\n', cleaned)
        cleaned = cleaned.strip()
        
        return cleaned if cleaned else text  # Return original if cleaning resulted in empty string
    
    def _enhance_statement_to_question(self, statement: str, context: Optional[str]) -> str:
        """
        Convert a statement into an implicit question when we have context
        
        Args:
            statement: The statement from user
            context: Document context (if available)
            
        Returns:
            Enhanced prompt that encourages information sharing
        """
        if not context:
            return statement
        
        statement_lower = statement.lower().strip()
        
        # Detect specific intents and create appropriate questions
        if any(word in statement_lower for word in ['quiz', 'mcq', 'multiple choice', 'test', 'exam']):
            return "Based on the provided document, create 5 multiple-choice questions (MCQs) to test understanding of the content. Each question should have 4 options (A, B, C, D) and indicate the correct answer."
        
        elif any(word in statement_lower for word in ['summary', 'summarize', 'overview']):
            return "Please provide a concise summary of the key points from the document."
        
        elif any(word in statement_lower for word in ['explain', 'tell me about', 'what is']):
            return f"Based on the document, please explain: {statement}"
        
        elif any(word in statement_lower for word in ['list', 'enumerate', 'what are']):
            return f"From the document, please list and explain: {statement}"
        
        else:
            # Generic enhancement for other statements
            return f"{statement}\n\nBased on the above context and the provided document, please share relevant information about this topic."
    
    def _check_ollama_availability(self) -> bool:
        """Check if Ollama server is running and model is available"""
        try:
            # Check if server is running
            response = requests.get(f"{self.base_url}/api/tags", timeout=2)
            if response.status_code == 200:
                models = response.json().get('models', [])
                model_names = [m.get('name', '') for m in models]
                
                # Check if our model is available (exact match or base name match)
                model_base = self.model.split(':')[0]
                if self.model in model_names or model_base in [m.split(':')[0] for m in model_names]:
                    self.ollama_available = True
                    logging.info(f"✅ Ollama available with model: {self.model}")
                    return True
                else:
                    logging.warning(f"⚠️  Ollama running but model '{self.model}' not found. Available: {model_names}")
                    logging.warning(f"   Install with: ollama pull {self.model}")
            return False
        except requests.exceptions.RequestException as e:
            logging.info(f"ℹ️  Ollama not available: {str(e)}")
            logging.info(f"   Start Ollama with: ollama serve")
            return False
    
    def ask_question(self, question: str, context: Optional[str] = None, use_history: bool = True) -> Dict:
        """
        Ask a question and get an answer from Ollama with Groq fallback
        
        Args:
            question: The question to ask
            context: Optional context to provide with the question
            use_history: Whether to use conversation history
            
        Returns:
            Dict with answer, status, and metadata
        """
        logging.info(f"🔵 ask_question called: question='{question}', context_available={bool(context)}, ollama_available={self.ollama_available}, groq_available={self.groq_available}")
        
        # Check if input is a statement and we have context
        if context and self._is_statement(question):
            logging.info(f"📝 Detected statement with context, enhancing to encourage information sharing")
            enhanced_question = self._enhance_statement_to_question(question, context)
            logging.info(f"✨ Enhanced to: {enhanced_question[:100]}...")
        else:
            enhanced_question = question
        
        # Try Ollama first with 30 second timeout
        if self.ollama_available:
            logging.info("🟡 Trying Ollama...")
            try:
                # Build prompt with context and history
                prompt = self._build_prompt(enhanced_question, context, use_history)
                
                # Call Ollama API with 30 second timeout
                response = requests.post(
                    f"{self.base_url}/api/generate",
                    json={
                        "model": self.model,
                        "prompt": prompt,
                        "stream": False,
                        "options": {
                            "temperature": 0.7,
                            "top_p": 0.9,
                            "max_tokens": 500
                        }
                    },
                    timeout=30  # 30 second timeout
                )
                
                if response.status_code == 200:
                    result = response.json()
                    answer = result.get('response', '').strip()
                    
                    # Remove <think> tags and content if present (deepseek-r1 sometimes includes these)
                    answer = self._clean_thinking_tags(answer)
                    
                    # Store ORIGINAL question in history (not enhanced)
                    self._add_to_history(question, answer)
                    
                    return {
                        'answer': answer,
                        'status': 'success',
                        'method': 'ollama',
                        'model': self.model,
                        'timestamp': datetime.now().isoformat()
                    }
            except requests.exceptions.Timeout:
                logging.warning(f"⏱️  Ollama timed out after 30 seconds, falling back to Groq")
                # Fall through to Groq fallback
            except Exception as e:
                logging.error(f"Ollama error: {e}, falling back to Groq")
                # Fall through to Groq fallback
        
        # Groq fallback
        if self.groq_available:
            logging.info("🟢 Trying Groq fallback...")
            try:
                return self._ask_with_groq(enhanced_question, context, use_history, original_question=question)
            except Exception as e:
                logging.error(f"Groq fallback failed: {e}")
        
        # Final fallback - rule-based
        logging.info("🔴 Using rule-based fallback...")
        return self._fallback_answer(question)
    
    def _build_prompt(self, question: str, context: Optional[str], use_history: bool) -> str:
        """Build the prompt for Ollama including context and history"""
        prompt_parts = []
        
        # System instruction with better handling for statements
        if context:
            # When we have document context, be more proactive
            prompt_parts.append(
                "You are a helpful assistant answering questions from a sign language user. "
                "DO NOT show your thinking process (<think> tags). "
                "Provide ONLY the final answer in plain English. "
                "If the user makes a statement or mentions a topic, provide relevant information about that topic from the provided context. "
                "If it's a question, answer it directly. "
                "Provide clear, concise, and helpful answers based on the context."
            )
        else:
            # Without context, be more conversational
            prompt_parts.append(
                "You are a helpful assistant answering questions from a sign language user. "
                "DO NOT show your thinking process (<think> tags). "
                "Provide ONLY the final answer in plain English. "
                "Provide clear, concise, and helpful answers. Keep responses under 100 words unless more detail is needed."
            )
        
        # Add conversation history if enabled
        if use_history and self.conversation_history:
            prompt_parts.append("\nPrevious conversation:")
            for entry in self.conversation_history[-3:]:  # Last 3 exchanges
                prompt_parts.append(f"Q: {entry['question']}")
                prompt_parts.append(f"A: {entry['answer']}")
        
        # Add context if provided
        if context:
            prompt_parts.append(f"\nDocument Content:\n{context}")
            prompt_parts.append("\nIMPORTANT: Use the above document to answer the question.")
        
        # Add the current question
        prompt_parts.append(f"\nQuestion: {question}")
        prompt_parts.append("\nAnswer (plain English, no <think> tags):")
        
        return "\n".join(prompt_parts)
    
    def _ask_with_groq(self, question: str, context: Optional[str], use_history: bool, original_question: Optional[str] = None) -> Dict:
        """
        Ask a question using Groq as fallback
        
        Args:
            question: The question to ask (possibly enhanced)
            context: Optional context to provide with the question
            use_history: Whether to use conversation history
            original_question: Original question before enhancement (for history)
            
        Returns:
            Dict with answer, status, and metadata
        """
        try:
            # Build messages for Groq
            messages = []
            
            # System message with better context handling
            if context:
                # When we have document context, be more proactive
                messages.append({
                    "role": "system",
                    "content": "You are a helpful assistant answering questions from a sign language user. "
                              "If the user makes a statement or mentions a topic, provide relevant information about that topic from the provided context. "
                              "If it's a question, answer it directly. "
                              "Provide clear, concise, and helpful answers based on the context. "
                              "Keep responses under 150 words unless more detail is needed."
                })
            else:
                # Without context, be more conversational
                messages.append({
                    "role": "system",
                    "content": "You are a helpful assistant answering questions from a sign language user. "
                              "Provide clear, concise, and helpful answers. Keep responses under 100 words unless more detail is needed."
                })
            
            # Add conversation history if enabled
            if use_history and self.conversation_history:
                for entry in self.conversation_history[-3:]:  # Last 3 exchanges
                    messages.append({
                        "role": "user",
                        "content": entry['question']
                    })
                    messages.append({
                        "role": "assistant",
                        "content": entry['answer']
                    })
            
            # Add context if provided
            if context:
                messages.append({
                    "role": "user",
                    "content": f"Context: {context}"
                })
            
            # Add the current question
            messages.append({
                "role": "user",
                "content": question
            })
            
            # Call Groq API with timeout
            completion = self.groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=messages,
                temperature=0.7,
                max_tokens=500,
                timeout=30  # 30 second timeout for Groq as well
            )
            
            answer = completion.choices[0].message.content.strip()
            
            # Store ORIGINAL question in history (not enhanced)
            self._add_to_history(original_question or question, answer)
            
            return {
                'answer': answer,
                'status': 'success',
                'method': 'groq',
                'model': 'llama-3.3-70b-versatile',
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logging.error(f"Groq API error: {e}")
            raise  # Re-raise to trigger final fallback
    
    def _add_to_history(self, question: str, answer: str):
        """Add Q&A pair to conversation history"""
        self.conversation_history.append({
            'question': question,
            'answer': answer,
            'timestamp': datetime.now().isoformat()
        })
        
        # Keep only recent history
        if len(self.conversation_history) > self.max_history:
            self.conversation_history = self.conversation_history[-self.max_history:]
    
    def _fallback_answer(self, question: str) -> Dict:
        """Provide a fallback answer when Ollama fails"""
        # Simple rule-based responses for common questions
        question_lower = question.lower()
        
        fallback_responses = {
            'hello': "Hello! I'm here to help answer your questions.",
            'hi': "Hi there! How can I assist you today?",
            'what': "I'm having trouble connecting to the AI model. Please make sure Ollama is running.",
            'how': "I'd love to help, but I'm having trouble accessing the AI model right now.",
            'who': "I'm a sign language Q&A assistant powered by Ollama AI.",
            'where': "I'm running locally on your computer using Ollama.",
            'when': "I'm available anytime to answer your questions!",
            'why': "Good question! I need the AI model running to give you a detailed answer."
        }
        
        for keyword, response in fallback_responses.items():
            if keyword in question_lower:
                return {
                    'answer': response,
                    'status': 'success',
                    'method': 'fallback',
                    'timestamp': datetime.now().isoformat()
                }
        
        return {
            'answer': "I'm having trouble processing your question. Please make sure Ollama is running with 'ollama serve'.",
            'status': 'error',
            'method': 'fallback',
            'timestamp': datetime.now().isoformat()
        }
    
    def clear_history(self):
        """Clear conversation history"""
        self.conversation_history = []
        logging.info("Conversation history cleared")
    
    def get_history(self, limit: int = 10) -> List[Dict]:
        """Get conversation history"""
        return self.conversation_history[-limit:]
    
    def get_stats(self) -> Dict:
        """Get processor statistics"""
        return {
            'ollama_available': self.ollama_available,
            'model': self.model,
            'base_url': self.base_url,
            'conversation_count': len(self.conversation_history),
            'max_history': self.max_history
        }
    
    def list_available_models(self) -> List[str]:
        """List all available Ollama models"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=2)
            if response.status_code == 200:
                models = response.json().get('models', [])
                return [m.get('name', '') for m in models]
            return []
        except:
            return []
    
    def switch_model(self, model: str) -> bool:
        """Switch to a different Ollama model"""
        self.model = model
        return self._check_ollama_availability()
