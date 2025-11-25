"""
Demo Script - AI-Enhanced Sign Language Translation
This script demonstrates the AI enhancement feature without needing the full app
"""

from ai_processor import AITranslationProcessor
import time

def print_header(text):
    """Print a formatted header"""
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70)

def print_example(raw, result, style):
    """Print a formatted example"""
    print(f"\n📝 Raw Signs: {raw}")
    print(f"🤖 {style.title()} Style: {result['enhanced']}")
    print(f"   Method: {result['method']} | Confidence: {result['confidence']}")

def demo_basic_translations():
    """Demo basic translation features"""
    print_header("🎯 DEMO 1: Basic Translations")
    
    processor = AITranslationProcessor()
    
    examples = [
        "ME GO STORE",
        "YOU WANT EAT",
        "I HAPPY TODAY",
        "THANK YOU HELP ME",
    ]
    
    for example in examples:
        result = processor.enhance_translation(example, style="casual")
        print_example(example, result, "casual")
        time.sleep(0.5)

def demo_style_comparison():
    """Demo different translation styles"""
    print_header("🎨 DEMO 2: Style Comparison")
    
    processor = AITranslationProcessor()
    
    raw_sign = "ME GO STORE"
    print(f"\n🔤 Raw Sign Sequence: '{raw_sign}'")
    print("\nHow different styles interpret it:\n")
    
    styles = ["casual", "neutral", "formal"]
    
    for style in styles:
        result = processor.enhance_translation(raw_sign, style=style)
        icon = "🙂" if style == "casual" else "⚖️" if style == "neutral" else "👔"
        print(f"{icon} {style.upper():8} → {result['enhanced']}")
        time.sleep(0.3)

def demo_complex_sentences():
    """Demo complex sentence translation"""
    print_header("💬 DEMO 3: Complex Sentences")
    
    processor = AITranslationProcessor()
    
    examples = [
        ("ME LEARN SIGN LANGUAGE", "Learning ASL"),
        ("WHERE YOU GO TODAY", "Asking about plans"),
        ("I WANT HELP YOU", "Offering assistance"),
        ("SORRY ME LATE", "Apologizing"),
    ]
    
    for raw, description in examples:
        print(f"\n📖 Scenario: {description}")
        print(f"   Raw: {raw}")
        
        casual = processor.enhance_translation(raw, style="casual")
        formal = processor.enhance_translation(raw, style="formal")
        
        print(f"   🙂 Casual: {casual['enhanced']}")
        print(f"   👔 Formal: {formal['enhanced']}")
        time.sleep(0.5)

def demo_real_conversation():
    """Demo a real conversation flow"""
    print_header("💭 DEMO 4: Real Conversation")
    
    processor = AITranslationProcessor()
    
    conversation = [
        ("ME GO STORE", "Person A signs"),
        ("YOU WANT WHAT", "Person B asks"),
        ("ME NEED MILK BREAD", "Person A replies"),
        ("OK ME GO WITH YOU", "Person B responds"),
    ]
    
    print("\n🎭 Simulating a conversation between two people:\n")
    
    for raw, speaker in conversation:
        result = processor.enhance_translation(raw, style="casual")
        print(f"👤 {speaker}:")
        print(f"   Signs: {raw}")
        print(f"   Says: {result['enhanced']}\n")
        time.sleep(1)

def demo_statistics():
    """Demo statistics and tracking"""
    print_header("📊 DEMO 5: Statistics & Tracking")
    
    processor = AITranslationProcessor()
    
    # Perform several translations
    test_signs = [
        "ME GO STORE",
        "YOU WANT EAT", 
        "I HAPPY TODAY",
        "THANK YOU",
    ]
    
    print("\n🔄 Processing translations...\n")
    for sign in test_signs:
        processor.enhance_translation(sign, style="casual")
        print(f"✓ Processed: {sign}")
    
    # Show statistics
    stats = processor.get_stats()
    
    print("\n📈 Translation Statistics:")
    print(f"   Total Translations: {stats['total_translations']}")
    print(f"   OpenAI API Used: {stats['openai_count']}")
    print(f"   Rule-based Used: {stats['rules_count']}")
    print(f"   Average Confidence: {stats['avg_confidence']}")
    print(f"   API Status: {'🟢 Active' if stats['openai_available'] else '⚪ Rule-based'}")

def demo_edge_cases():
    """Demo handling of edge cases"""
    print_header("🔧 DEMO 6: Handling Edge Cases")
    
    processor = AITranslationProcessor()
    
    edge_cases = [
        ("", "Empty input"),
        ("N/A", "No detection"),
        ("ME ME ME GO", "Repetition"),
        ("WHAT WHERE WHEN", "Multiple questions"),
    ]
    
    for raw, description in edge_cases:
        print(f"\n🧪 Edge Case: {description}")
        print(f"   Input: '{raw}'")
        result = processor.enhance_translation(raw, style="casual")
        print(f"   Output: '{result['enhanced']}'")
        print(f"   Handled: ✅")
        time.sleep(0.5)

def demo_performance():
    """Demo performance and speed"""
    print_header("⚡ DEMO 7: Performance Test")
    
    processor = AITranslationProcessor()
    
    test_sign = "ME GO STORE"
    iterations = 5
    
    print(f"\n⏱️  Processing '{test_sign}' {iterations} times...\n")
    
    start_time = time.time()
    
    for i in range(iterations):
        result = processor.enhance_translation(test_sign, style="casual")
        print(f"   Translation {i+1}: {result['enhanced']} ✓")
    
    end_time = time.time()
    total_time = end_time - start_time
    avg_time = total_time / iterations
    
    print(f"\n📊 Performance Results:")
    print(f"   Total Time: {total_time:.3f} seconds")
    print(f"   Average Time: {avg_time:.3f} seconds per translation")
    print(f"   Rate: {iterations/total_time:.1f} translations/second")

def main():
    """Run all demos"""
    print("\n")
    print("╔═══════════════════════════════════════════════════════════════════╗")
    print("║                                                                   ║")
    print("║         🤖 AI-ENHANCED SIGN LANGUAGE TRANSLATION DEMO 🤖          ║")
    print("║                                                                   ║")
    print("║              Making Sign Language More Natural                    ║")
    print("║                                                                   ║")
    print("╚═══════════════════════════════════════════════════════════════════╝")
    
    demos = [
        ("1", "Basic Translations", demo_basic_translations),
        ("2", "Style Comparison", demo_style_comparison),
        ("3", "Complex Sentences", demo_complex_sentences),
        ("4", "Real Conversation", demo_real_conversation),
        ("5", "Statistics & Tracking", demo_statistics),
        ("6", "Edge Cases", demo_edge_cases),
        ("7", "Performance Test", demo_performance),
    ]
    
    print("\nAvailable Demos:")
    for num, name, _ in demos:
        print(f"  {num}. {name}")
    print("  A. Run All Demos")
    print("  Q. Quit")
    
    while True:
        choice = input("\n👉 Select demo (1-7, A, or Q): ").strip().upper()
        
        if choice == 'Q':
            print("\n👋 Thanks for watching the demo! Goodbye!")
            break
        elif choice == 'A':
            print("\n🚀 Running all demos...\n")
            for _, _, demo_func in demos:
                demo_func()
                time.sleep(1)
            print("\n✅ All demos completed!")
            break
        elif choice in ['1', '2', '3', '4', '5', '6', '7']:
            idx = int(choice) - 1
            demos[idx][2]()
        else:
            print("❌ Invalid choice. Please select 1-7, A, or Q.")
    
    print("\n" + "=" * 70)
    print("  📚 For more information, see AI_FEATURES.md")
    print("  🚀 To run the full application: python app.py")
    print("=" * 70 + "\n")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Demo interrupted by user. Goodbye!")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("Make sure ai_processor.py is in the same directory!")
