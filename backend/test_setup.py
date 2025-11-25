"""
Test script for AI Processor functionality
Run this to verify your setup is working correctly
"""

import sys
import os

def test_imports():
    """Test if all required packages are installed"""
    print("=" * 60)
    print("Testing imports...")
    print("=" * 60)
    
    try:
        import flask
        print("✅ Flask installed")
    except ImportError:
        print("❌ Flask not found. Run: pip install flask")
        return False
    
    try:
        import cv2
        print("✅ OpenCV installed")
    except ImportError:
        print("❌ OpenCV not found. Run: pip install opencv-python")
        return False
    
    try:
        import mediapipe
        print("✅ MediaPipe installed")
    except ImportError:
        print("❌ MediaPipe not found. Run: pip install mediapipe")
        return False
    
    try:
        from dotenv import load_dotenv
        print("✅ python-dotenv installed")
    except ImportError:
        print("❌ python-dotenv not found. Run: pip install python-dotenv")
        return False
    
    try:
        import openai
        print("✅ OpenAI library installed")
    except ImportError:
        print("⚠️  OpenAI library not found. Rule-based mode will be used.")
        print("   To use AI enhancement: pip install openai")
    
    return True

def test_ai_processor():
    """Test AI processor functionality"""
    print("\n" + "=" * 60)
    print("Testing AI Processor...")
    print("=" * 60)
    
    try:
        from ai_processor import AITranslationProcessor
        
        processor = AITranslationProcessor()
        
        # Test case
        test_input = "ME GO STORE"
        print(f"\nTest Input: '{test_input}'")
        
        # Test casual style
        result_casual = processor.enhance_translation(test_input, style="casual")
        print(f"\n✅ Casual Style: '{result_casual['enhanced']}'")
        print(f"   Method: {result_casual['method']}")
        print(f"   Confidence: {result_casual['confidence']}")
        
        # Test formal style
        result_formal = processor.enhance_translation(test_input, style="formal")
        print(f"\n✅ Formal Style: '{result_formal['enhanced']}'")
        
        # Check if OpenAI is available
        if processor.openai_available:
            print("\n🎉 OpenAI API is active!")
        else:
            print("\n⚠️  Using rule-based enhancement (OpenAI API not configured)")
            print("   To enable AI: Set OPENAI_API_KEY in .env file")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing AI processor: {e}")
        return False

def test_environment():
    """Check environment configuration"""
    print("\n" + "=" * 60)
    print("Checking Environment...")
    print("=" * 60)
    
    from dotenv import load_dotenv
    load_dotenv()
    
    api_key = os.getenv('OPENAI_API_KEY')
    
    if api_key and api_key != 'your_openai_api_key_here':
        print("✅ OpenAI API key configured")
        print(f"   Key: {api_key[:8]}...{api_key[-4:]}")
    elif os.path.exists('.env'):
        print("⚠️  .env file exists but API key not set or using placeholder")
        print("   Edit .env and add your real API key")
    else:
        print("⚠️  No .env file found")
        print("   Copy .env.example to .env and add your API key")
        print("   Or run without API key for rule-based mode")
    
    return True

def run_comprehensive_tests():
    """Run comprehensive test suite"""
    print("\n" + "=" * 60)
    print("Running Comprehensive Tests...")
    print("=" * 60)
    
    from ai_processor import AITranslationProcessor
    
    processor = AITranslationProcessor()
    
    test_cases = [
        ("ME GO STORE", "Going to store"),
        ("YOU WANT EAT", "Asking about eating"),
        ("THANK YOU HELP ME", "Expressing gratitude"),
        ("I HAPPY TODAY", "Expressing happiness"),
        ("WHERE YOU GO", "Asking location"),
    ]
    
    print("\nRunning test cases...\n")
    
    for raw, description in test_cases:
        print(f"📝 {description}")
        print(f"   Raw: {raw}")
        
        result = processor.enhance_translation(raw, style="casual")
        print(f"   Enhanced: {result['enhanced']}")
        print(f"   Method: {result['method']}\n")
    
    # Show statistics
    stats = processor.get_stats()
    print("=" * 60)
    print("Statistics:")
    print("=" * 60)
    for key, value in stats.items():
        print(f"  {key}: {value}")

def main():
    """Main test runner"""
    print("\n" + "=" * 60)
    print("SignSpeak AI Enhancement - Setup Verification")
    print("=" * 60)
    
    # Test imports
    if not test_imports():
        print("\n❌ Import tests failed. Please install missing packages.")
        return
    
    # Test environment
    test_environment()
    
    # Test AI processor
    if test_ai_processor():
        print("\n✅ Basic tests passed!")
        
        # Ask if user wants comprehensive tests
        print("\n" + "=" * 60)
        response = input("\nRun comprehensive tests? (y/n): ").lower()
        if response == 'y':
            run_comprehensive_tests()
    else:
        print("\n❌ AI processor tests failed.")
        return
    
    print("\n" + "=" * 60)
    print("🎉 Setup verification complete!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Run the application: python app.py")
    print("2. Open browser: http://localhost:5000")
    print("3. Start signing and watch the AI magic! ✨")
    print("\nFor more info, see AI_FEATURES.md or QUICKSTART.md")
    print("=" * 60)

if __name__ == "__main__":
    main()
