#!/usr/bin/env python3
"""
Test script to verify RAG flow with PDF upload and question asking
"""

import requests
import json
import time
import os

BASE_URL = "http://127.0.0.1:5001"

def test_rag_flow():
    """Test the complete RAG flow"""
    
    print("="*60)
    print("🧪 Testing RAG Flow")
    print("="*60)
    
    # Step 1: Check initial status
    print("\n1️⃣ Checking initial status...")
    response = requests.get(f"{BASE_URL}/get_status")
    status = response.json()
    print(f"   Current sentence: {status.get('current_sentence', 'N/A')}")
    print(f"   Enhanced sentence: {status.get('enhanced_sentence', 'N/A')}")
    
    # Step 2: Simulate a sentence (in real app, this comes from sign detection)
    print("\n2️⃣ Setting up test sentence...")
    # We'll need to manually set this through the app interface
    # For testing, let's directly call the ask endpoint with a test question
    test_question = "What is machine learning?"
    print(f"   Test question: {test_question}")
    
    # Step 3: Check for uploaded documents
    print("\n3️⃣ Checking for uploaded documents...")
    response = requests.get(f"{BASE_URL}/list_documents")
    docs_data = response.json()
    print(f"   Uploaded documents: {docs_data.get('count', 0)}")
    
    has_documents = docs_data.get('count', 0) > 0
    
    if has_documents:
        print(f"   Documents found:")
        for doc in docs_data.get('documents', []):
            print(f"      - {doc['filename']} ({doc['word_count']} words)")
    else:
        print("   ℹ️  No documents uploaded. Upload a PDF to test RAG.")
        print("   📝 Testing without RAG (direct Ollama/Groq)...")
    
    # Step 4: Ask question (with or without RAG)
    endpoint = "/ask_with_document" if has_documents else "/ask_question"
    print(f"\n4️⃣ Asking question via {endpoint}...")
    
    payload = {
        "question": test_question,
        "use_history": True
    }
    
    print(f"   Payload: {json.dumps(payload, indent=2)}")
    
    try:
        response = requests.post(
            f"{BASE_URL}{endpoint}",
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=35  # 35 second timeout to allow for 30 second Ollama timeout + buffer
        )
        
        result = response.json()
        print(f"\n5️⃣ Response received:")
        print(f"   Status: {result.get('status', 'unknown')}")
        print(f"   Method: {result.get('method', 'unknown')}")
        print(f"   Model: {result.get('model', 'unknown')}")
        
        if has_documents:
            print(f"   Used RAG: {result.get('used_rag', False)}")
            print(f"   Context length: {result.get('context_length', 0)} chars")
        
        print(f"\n   Answer:")
        print(f"   {result.get('answer', 'No answer')}")
        
        # Step 5: Check Q&A history
        print("\n6️⃣ Checking Q&A history...")
        response = requests.get(f"{BASE_URL}/qa_history?limit=3")
        history = response.json()
        print(f"   History entries: {len(history)}")
        
        if history:
            print("   Recent Q&A:")
            for i, entry in enumerate(history[-3:], 1):
                print(f"      {i}. Q: {entry['question'][:50]}...")
                print(f"         A: {entry['answer'][:50]}...")
        
        print("\n" + "="*60)
        print("✅ RAG flow test complete!")
        print("="*60)
        
        if has_documents:
            print("\n💡 Test Results:")
            print(f"   - Documents are being used: {result.get('used_rag', False)}")
            print(f"   - Context retrieved: {result.get('context_length', 0) > 0}")
            print(f"   - Answer method: {result.get('method', 'unknown')}")
        else:
            print("\n💡 To test with documents:")
            print("   1. Open the web UI: http://127.0.0.1:5001")
            print("   2. Upload a PDF in the 'Upload Document' section")
            print("   3. Sign a question or type it")
            print("   4. Click 'Ask Question'")
            print("   5. Re-run this test")
        
    except requests.exceptions.Timeout:
        print("\n⏱️  Request timed out after 35 seconds")
        print("   This might indicate Ollama is taking too long")
        print("   Check if Ollama is running: ollama serve")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print(f"   Response status: {response.status_code if 'response' in locals() else 'N/A'}")
        if 'response' in locals():
            print(f"   Response: {response.text}")

def test_document_upload():
    """Test document upload functionality"""
    print("\n" + "="*60)
    print("📄 Testing Document Upload")
    print("="*60)
    
    # Check if a test PDF exists
    test_files = [
        "/Users/lovishgarg/SignSpeak/README.md",
        "/Users/lovishgarg/SignSpeak/AI_FEATURES.md",
    ]
    
    for test_file in test_files:
        if os.path.exists(test_file):
            print(f"\n📤 Uploading {os.path.basename(test_file)}...")
            
            with open(test_file, 'rb') as f:
                files = {'file': (os.path.basename(test_file), f, 'text/plain')}
                response = requests.post(f"{BASE_URL}/upload_document", files=files)
                
                if response.status_code == 200:
                    result = response.json()
                    print(f"   ✅ Upload successful!")
                    print(f"   Doc ID: {result.get('doc_id', 'N/A')}")
                    print(f"   Word count: {result.get('word_count', 0)}")
                    print(f"   Chunks: {result.get('chunk_count', 0)}")
                    return True
                else:
                    print(f"   ❌ Upload failed: {response.text}")
    
    print("\n   ℹ️  No test files found for upload")
    return False

if __name__ == "__main__":
    # First test with existing documents
    test_rag_flow()
    
    # Ask if user wants to test upload
    print("\n" + "="*60)
    choice = input("\n🔧 Do you want to test document upload? (y/n): ").lower()
    
    if choice == 'y':
        if test_document_upload():
            print("\n🔄 Re-running RAG test with uploaded document...")
            time.sleep(1)
            test_rag_flow()
