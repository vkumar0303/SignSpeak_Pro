from flask import Flask, render_template, Response, jsonify, request
import pickle
import cv2
import mediapipe as mp
import numpy as np
import time
import warnings
import os
import logging
import threading
from werkzeug.utils import secure_filename
from ai_processor import AITranslationProcessor
from ollama_qa import OllamaQAProcessor
from rag_processor import RAGProcessor
from dotenv import load_dotenv
warnings.filterwarnings("ignore", category=UserWarning)

# Configure logging
logging.basicConfig(level=logging.INFO)

# Load environment variables
load_dotenv()

# Configure Flask to use frontend templates directory
template_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'frontend', 'templates'))
app = Flask(__name__, template_folder=template_dir)

# Configure upload settings
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'uploads')
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'docx', 'doc'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Initialize AI Processor
ai_processor = AITranslationProcessor()

# Initialize Ollama Q&A Processor with Groq fallback
qa_processor = OllamaQAProcessor(
    model="deepseek-r1:1.5b",
    groq_api_key=os.getenv('GROQ_API_KEY')  # Use Groq as fallback
)

# Initialize RAG Processor
rag_processor = RAGProcessor(upload_folder=UPLOAD_FOLDER)

# Load the model
model_path = os.path.join(os.path.dirname(__file__), 'model.p')
model_dict = pickle.load(open(model_path, 'rb'))
model = model_dict['model']

# Mediapipe setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.5, max_num_hands=1)

# Global variables for processing
labels_dict = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J', 10: 'K', 11: 'L', 12: 'M',
    13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y',
    25: 'Z', 26: '0', 27: '1', 28: '2', 29: '3', 30: '4', 31: '5', 32: '6', 33: '7', 34: '8', 35: '9',
    36: ' ', 37: '.'
}

expected_features = 42
stabilization_buffer = []
stable_char = None
word_buffer = ""
sentence = ""
enhanced_sentence = ""
last_registered_time = time.time()
registration_delay = 1.5
is_paused = False
current_alphabet = "N/A"
expected_features = 42
translation_style = "casual"  # default style: casual, formal, or neutral
auto_enhance = True  # automatically enhance sentence with AI
last_enhanced_sentence = ""  # track last enhanced sentence to avoid duplicate API calls

# Function to generate camera frames
def generate_frames():
    global stabilization_buffer, stable_char, word_buffer, sentence, last_registered_time, is_paused, current_alphabet
    
    cap = None
    try:
        cap = cv2.VideoCapture(0)
        
        # Check if camera opened successfully
        if not cap.isOpened():
            logging.error("Failed to open camera")
            return
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 400)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 300)
        
        # Give camera time to warm up
        time.sleep(0.1)
        
        while True:
            try:
                success, frame = cap.read()
                if not success:
                    logging.warning("Failed to read frame from camera")
                    break
                    
                if not is_paused:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = hands.process(frame_rgb)

                    if results.multi_hand_landmarks:
                        for hand_landmarks in results.multi_hand_landmarks:
                            data_aux = []
                            x_ = []
                            y_ = []

                            for i in range(len(hand_landmarks.landmark)):
                                x = hand_landmarks.landmark[i].x
                                y = hand_landmarks.landmark[i].y
                                x_.append(x)
                                y_.append(y)

                            for i in range(len(hand_landmarks.landmark)):
                                x = hand_landmarks.landmark[i].x
                                y = hand_landmarks.landmark[i].y
                                data_aux.append(x - min(x_))
                                data_aux.append(y - min(y_))

                            # Ensure valid data
                            if len(data_aux) < expected_features:
                                data_aux.extend([0] * (expected_features - len(data_aux)))
                            elif len(data_aux) > expected_features:
                                data_aux = data_aux[:expected_features]

                            # Predict gesture
                            prediction = model.predict([np.asarray(data_aux)])
                            predicted_character = labels_dict[int(prediction[0])]

                            # Stabilization logic
                            stabilization_buffer.append(predicted_character)
                            if len(stabilization_buffer) > 30:  # Buffer size for 1 second
                                stabilization_buffer.pop(0)

                            if stabilization_buffer.count(predicted_character) > 25:  # Stabilization threshold
                                # Register the character only if enough time has passed since the last registration
                                current_time = time.time()
                                if current_time - last_registered_time > registration_delay:
                                    stable_char = predicted_character
                                    last_registered_time = current_time  # Update last registered time
                                    current_alphabet = stable_char

                                    # Handle word and sentence formation
                                    if stable_char == ' ':
                                        if word_buffer.strip():  # Add word to sentence if not empty
                                            sentence += word_buffer + " "
                                        word_buffer = ""
                                    elif stable_char == '.':
                                        if word_buffer.strip():  # Add word to sentence before adding period
                                            sentence += word_buffer + "."
                                        word_buffer = ""
                                    else:
                                        word_buffer += stable_char

                            # Draw landmarks and bounding box
                            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                                                    mp_drawing_styles.get_default_hand_landmarks_style(),
                                                    mp_drawing_styles.get_default_hand_connections_style())

                # Draw alphabet on the video feed
                cv2.putText(frame, f"Alphabet: {current_alphabet}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)  # Yellow color

                ret, buffer = cv2.imencode('.jpg', frame)
                if not ret:
                    logging.warning("Failed to encode frame")
                    continue
                    
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                       
            except GeneratorExit:
                # Client disconnected
                logging.info("Client disconnected from video feed")
                break
            except Exception as e:
                logging.error(f"Error processing frame: {e}")
                # Continue processing next frame
                continue
                
    except Exception as e:
        logging.error(f"Fatal error in generate_frames: {e}")
    finally:
        # Always release the camera
        if cap is not None:
            cap.release()
            logging.info("Camera released")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), 
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_status')
def get_status():
    global current_alphabet, word_buffer, sentence, enhanced_sentence, translation_style, auto_enhance, last_enhanced_sentence
    
    # Auto-enhance sentence if enabled and sentence has changed
    if auto_enhance and sentence and sentence.strip() and sentence != last_enhanced_sentence:
        try:
            result = ai_processor.enhance_translation(sentence, style=translation_style)
            enhanced_sentence = result.get('enhanced', sentence)
            last_enhanced_sentence = sentence
        except Exception as e:
            print(f"Error enhancing sentence: {e}")
            enhanced_sentence = sentence
    
    return jsonify({
        'current_alphabet': current_alphabet,
        'current_word': word_buffer if word_buffer else "N/A",
        'current_sentence': sentence if sentence else "N/A",
        'enhanced_sentence': enhanced_sentence if enhanced_sentence else "N/A",
        'translation_style': translation_style,
        'auto_enhance': auto_enhance,
        'is_paused': is_paused,
        'ai_available': ai_processor.groq_available
    })

@app.route('/reset_sentence')
def reset_sentence():
    global word_buffer, sentence, current_alphabet, enhanced_sentence, last_enhanced_sentence
    word_buffer = ""
    sentence = ""
    enhanced_sentence = ""
    last_enhanced_sentence = ""
    current_alphabet = "N/A"
    return jsonify({'status': 'success'})

@app.route('/toggle_pause')
def toggle_pause():
    global is_paused
    is_paused = not is_paused
    return jsonify({'status': 'success', 'is_paused': is_paused})

@app.route('/speak_sentence')
def speak_sentence():
    global sentence, enhanced_sentence
    # Prefer enhanced sentence if available, otherwise use raw sentence
    sentence_to_speak = enhanced_sentence if enhanced_sentence else sentence
    # The actual speaking will be handled by JavaScript in the frontend
    return jsonify({'status': 'success', 'sentence': sentence_to_speak})

@app.route('/set_style', methods=['POST'])
def set_style():
    """Set the translation style (casual, formal, neutral)"""
    global translation_style, enhanced_sentence, last_enhanced_sentence
    data = request.get_json()
    style = data.get('style', 'casual')
    
    if style in ['casual', 'formal', 'neutral']:
        translation_style = style
        # Re-enhance current sentence with new style
        if sentence:
            try:
                result = ai_processor.enhance_translation(sentence, style=translation_style)
                enhanced_sentence = result.get('enhanced', sentence)
                last_enhanced_sentence = sentence
            except Exception as e:
                print(f"Error re-enhancing with new style: {e}")
        
        return jsonify({'status': 'success', 'style': translation_style})
    else:
        return jsonify({'status': 'error', 'message': 'Invalid style'}), 400

@app.route('/toggle_auto_enhance', methods=['POST'])
def toggle_auto_enhance():
    """Toggle automatic AI enhancement"""
    global auto_enhance
    auto_enhance = not auto_enhance
    return jsonify({'status': 'success', 'auto_enhance': auto_enhance})

@app.route('/manual_enhance', methods=['POST'])
def manual_enhance():
    """Manually trigger AI enhancement for current sentence"""
    global sentence, enhanced_sentence, translation_style, last_enhanced_sentence
    
    if not sentence or sentence.strip() in ["N/A", ""]:
        return jsonify({'status': 'error', 'message': 'No sentence to enhance'}), 400
    
    try:
        result = ai_processor.enhance_translation(sentence, style=translation_style)
        enhanced_sentence = result.get('enhanced', sentence)
        last_enhanced_sentence = sentence
        
        return jsonify({
            'status': 'success',
            'raw': sentence,
            'enhanced': enhanced_sentence,
            'method': result.get('method'),
            'confidence': result.get('confidence')
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/get_ai_stats')
def get_ai_stats():
    """Get AI processor statistics"""
    stats = ai_processor.get_stats()
    return jsonify(stats)

@app.route('/get_translation_history')
def get_translation_history():
    """Get recent translation history"""
    limit = request.args.get('limit', 10, type=int)
    history = ai_processor.get_translation_history(limit)
    return jsonify(history)

@app.route('/ask_question', methods=['POST'])
def ask_question():
    """Ask a question using the enhanced sentence and get an answer from Ollama"""
    global enhanced_sentence, sentence
    
    data = request.get_json()
    question = data.get('question', '')
    use_history = data.get('use_history', True)
    
    # If no question provided, try to use the enhanced or raw sentence
    if not question:
        question = enhanced_sentence if enhanced_sentence else sentence
    
    if not question or question.strip() in ["N/A", ""]:
        return jsonify({
            'status': 'error',
            'message': 'No question to ask. Please sign a question first.'
        }), 400
    
    try:
        result = qa_processor.ask_question(question, use_history=use_history)
        return jsonify(result)
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Error processing question: {str(e)}'
        }), 500

@app.route('/qa_history')
def qa_history():
    """Get Q&A conversation history"""
    limit = request.args.get('limit', 10, type=int)
    history = qa_processor.get_history(limit)
    return jsonify(history)

@app.route('/clear_qa_history', methods=['POST'])
def clear_qa_history():
    """Clear Q&A conversation history"""
    qa_processor.clear_history()
    return jsonify({'status': 'success', 'message': 'Q&A history cleared'})

@app.route('/qa_stats')
def qa_stats():
    """Get Q&A processor statistics"""
    stats = qa_processor.get_stats()
    return jsonify(stats)

@app.route('/list_models')
def list_models():
    """List available Ollama models"""
    models = qa_processor.list_available_models()
    return jsonify({'models': models, 'current_model': qa_processor.model})

@app.route('/switch_model', methods=['POST'])
def switch_model():
    """Switch to a different Ollama model"""
    data = request.get_json()
    model = data.get('model', 'llama2')
    
    success = qa_processor.switch_model(model)
    
    if success:
        return jsonify({
            'status': 'success',
            'model': model,
            'message': f'Switched to model: {model}'
        })
    else:
        return jsonify({
            'status': 'error',
            'message': f'Failed to switch to model: {model}. Make sure it is installed.'
        }), 400

# ========================================
# RAG (Document Upload) Endpoints
# ========================================

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/upload_document', methods=['POST'])
def upload_document():
    """Upload a document for RAG-based Q&A"""
    try:
        # Check if file is in request
        if 'file' not in request.files:
            return jsonify({
                'status': 'error',
                'message': 'No file provided'
            }), 400
        
        file = request.files['file']
        
        # Check if file is selected
        if file.filename == '':
            return jsonify({
                'status': 'error',
                'message': 'No file selected'
            }), 400
        
        # Check file extension
        if not allowed_file(file.filename):
            return jsonify({
                'status': 'error',
                'message': f'File type not supported. Allowed: {", ".join(ALLOWED_EXTENSIONS)}'
            }), 400
        
        # Secure the filename
        filename = secure_filename(file.filename)
        
        # Save file temporarily
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        # Process document
        result = rag_processor.upload_document(file_path, filename)
        
        # Remove temporary file
        try:
            os.remove(file_path)
        except:
            pass
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Upload failed: {str(e)}'
        }), 500

@app.route('/ask_with_document', methods=['POST'])
def ask_with_document():
    """Ask a question with document context (RAG)"""
    global enhanced_sentence, sentence
    
    data = request.get_json()
    question = data.get('question', '')
    doc_id = data.get('doc_id', None)  # Optional: specific document to query
    use_history = data.get('use_history', True)
    
    # Log what we received
    logging.info(f"🔵 ask_with_document called")
    logging.info(f"  📤 Received question: '{question}'")
    logging.info(f"  📝 Current enhanced_sentence: '{enhanced_sentence}'")
    logging.info(f"  📝 Current sentence: '{sentence}'")
    
    # If no question provided, try to use the enhanced or raw sentence
    if not question:
        question = enhanced_sentence if enhanced_sentence else sentence
        logging.info(f"  🔄 Using fallback question: '{question}'")
    
    if not question or question.strip() in ["N/A", ""]:
        return jsonify({
            'status': 'error',
            'message': 'No question to ask. Please sign a question first.'
        }), 400
    
    try:
        # Retrieve relevant context from documents
        logging.info(f"  🔍 Retrieving context for question: '{question[:50]}...'")
        context = rag_processor.retrieve_context(question, doc_id=doc_id)
        logging.info(f"  📚 Context retrieved: {len(context)} characters")
        logging.info(f"  📚 Context retrieved: {len(context)} characters")
        
        if not context:
            logging.warning(f"  ⚠️  No context found for question: '{question}'")
            return jsonify({
                'status': 'warning',
                'message': 'No relevant context found in uploaded documents',
                'answer': 'Please upload a document first or ask a question without document context.',
                'method': 'none',
                'question_used': question
            }), 200
        
        # Ask question with document context
        result = qa_processor.ask_question(question, context=context, use_history=use_history)
        
        # Add indicator that this used RAG
        result['used_rag'] = True
        result['context_length'] = len(context)
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Error processing question: {str(e)}'
        }), 500

@app.route('/list_documents')
def list_documents():
    """Get list of uploaded documents"""
    documents = rag_processor.get_documents()
    return jsonify({
        'status': 'success',
        'documents': documents,
        'count': len(documents)
    })

@app.route('/document_info/<doc_id>')
def document_info(doc_id):
    """Get information about a specific document"""
    info = rag_processor.get_document_info(doc_id)
    if info:
        return jsonify({
            'status': 'success',
            'document': info
        })
    else:
        return jsonify({
            'status': 'error',
            'message': 'Document not found'
        }), 404

@app.route('/delete_document/<doc_id>', methods=['DELETE'])
def delete_document(doc_id):
    """Delete a document"""
    success = rag_processor.delete_document(doc_id)
    if success:
        return jsonify({
            'status': 'success',
            'message': 'Document deleted'
        })
    else:
        return jsonify({
            'status': 'error',
            'message': 'Document not found'
        }), 404

@app.route('/clear_documents', methods=['POST'])
def clear_documents():
    """Clear all uploaded documents"""
    rag_processor.clear_all_documents()
    return jsonify({
        'status': 'success',
        'message': 'All documents cleared'
    })

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🎉 SignSpeak AI-Enhanced Sign Language Translator")
    print("="*60)
    
    if ai_processor.groq_available:
        print(f"✅ AI Translation: Groq ({ai_processor.model}) - Enhanced translations")
    else:
        print("ℹ️  AI Translation: Rule-based (Works great without API key!)")
        print("   💡 To enable Groq: Add GROQ_API_KEY to .env file")
    
    if qa_processor.ollama_available:
        fallback_status = " + Groq fallback" if qa_processor.groq_available else ""
        print(f"✅ Q&A System: Ollama ({qa_processor.model}){fallback_status} - Ready to answer questions!")
    elif qa_processor.groq_available:
        print("✅ Q&A System: Groq fallback (Ollama not available)")
    else:
        print("ℹ️  Q&A System: Limited functionality")
        print(f"   💡 Start Ollama: ollama serve")
        print(f"   💡 Install model: ollama pull llama2")
        print(f"   💡 Or add GROQ_API_KEY to .env for Groq fallback")
    
    print("\n📍 Server starting at: http://127.0.0.1:5001")
    print("🔧 Press CTRL+C to quit")
    print("="*60 + "\n")
    
    app.run(debug=True, port=5001)