#!/usr/bin/env python3
"""
Test script for Ollama Q&A integration in SignSpeak
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

from ollama_qa import OllamaQAProcessor

def test_ollama_qa():
    print("\n" + "="*60)
    print("🤖 Testing Ollama Q&A Integration")
    print("="*60)
    
    # Initialize processor
    print("\n1. Initializing Ollama Q&A Processor...")
    qa = OllamaQAProcessor(model="llama2")
    
    if qa.ollama_available:
        print(f"   ✅ Ollama is available with model: {qa.model}")
    else:
        print(f"   ⚠️  Ollama not available. Please:")
        print(f"      1. Start Ollama: ollama serve")
        print(f"      2. Install model: ollama pull llama2")
        print(f"\n   Testing fallback mode...")
    
    # Test basic question
    print("\n2. Testing Basic Question...")
    question1 = "What is sign language?"
    print(f"   Question: {question1}")
    result1 = qa.ask_question(question1, use_history=False)
    
    if result1['status'] == 'success':
        print(f"   ✅ Answer received!")
        print(f"   Method: {result1.get('method')}")
        print(f"   Answer: {result1['answer'][:100]}...")
    else:
        print(f"   ⚠️  Status: {result1['status']}")
        print(f"   Message: {result1['answer']}")
    
    # Test with conversation history
    if qa.ollama_available:
        print("\n3. Testing Conversation History...")
        question2 = "How do I learn it?"
        print(f"   Follow-up: {question2}")
        result2 = qa.ask_question(question2, use_history=True)
        
        if result2['status'] == 'success':
            print(f"   ✅ Context-aware answer received!")
            print(f"   Answer: {result2['answer'][:100]}...")
        else:
            print(f"   ⚠️  {result2['answer']}")
    
    # Test statistics
    print("\n4. Checking Statistics...")
    stats = qa.get_stats()
    print(f"   Ollama Available: {stats['ollama_available']}")
    print(f"   Model: {stats['model']}")
    print(f"   Conversation Count: {stats['conversation_count']}")
    
    # Test model listing
    print("\n5. Listing Available Models...")
    models = qa.list_available_models()
    if models:
        print(f"   Available models: {', '.join(models[:5])}")
    else:
        print(f"   No models found or Ollama not running")
    
    # Get conversation history
    if qa.conversation_history:
        print("\n6. Conversation History...")
        history = qa.get_history(limit=2)
        for i, item in enumerate(history, 1):
            print(f"   Q{i}: {item['question']}")
            print(f"   A{i}: {item['answer'][:80]}...")
    
    print("\n" + "="*60)
    print("✅ Ollama Q&A Test Complete!")
    print("="*60 + "\n")
    
    if qa.ollama_available:
        print("🎉 Your Q&A system is ready to use!")
        print("   Start SignSpeak with: python backend/app.py")
    else:
        print("💡 To enable full Q&A functionality:")
        print("   1. Install Ollama: https://ollama.com/download")
        print("   2. Start server: ollama serve")
        print("   3. Pull model: ollama pull llama2")
        print("   4. Run this test again!")
    
    print()

if __name__ == "__main__":
    try:
        test_ollama_qa()
    except KeyboardInterrupt:
        print("\n\n⚠️  Test interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n❌ Error during test: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
