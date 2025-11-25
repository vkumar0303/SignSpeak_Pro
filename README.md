# SignSpeak

![SignSpeak Logo](https://github.com/Mukunj-21/SignSpeak/raw/main/Images/Logo.png)

## Breaking Communication Barriers: Real-time Sign Language Detection and Translation

SignSpeak is an advanced, machine learning-powered application that bridges the communication gap between the hearing and deaf communities through real-time sign language recognition and translation.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green.svg)](https://opencv.org/)

## ✨ Features

- **Real-time Sign Language Detection**: Accurately recognizes American Sign Language (ASL) hand gestures through your webcam
- **🤖 AI-Enhanced Translation**: Uses GPT to convert raw signs into fluent, natural sentences
- **💬 Q&A with Sign Language**: Ask questions in sign language and get AI-powered answers using local Ollama models
- **🎨 Multiple Translation Styles**: Choose between casual, neutral, or formal language output
- **Text-to-Speech Output**: Converts detected signs into audible speech for seamless communication
- **Responsive Web Interface**: Access the application from any device with a modern browser
- **High Detection Accuracy**: Powered by a custom-trained Convolutional Neural Network (CNN)
- **Low Latency**: Optimized for real-time performance with minimal delay
- **100% Local AI Processing**: Optional Ollama integration runs entirely on your machine - no cloud required
- **Educational Mode**: Learn sign language with interactive tutorials and practice exercises

## 🖼️ Screenshots

<div align="center">
  <img src="https://github.com/Mukunj-21/SignSpeak/raw/main/Images/Interface.png" alt="SignSpeak Interface" width="600"/>
  <p><i>SignSpeak Web Interface with Real-time Detection</i></p>
</div>

## 🚀 Installation

### Prerequisites

- Python 3.10 (Recommended)
- Webcam or camera device
- Internet connection (for initial setup)
- OpenAI API key (optional, for cloud-based AI translation)
- Ollama (optional, for local AI Q&A - see [OLLAMA_QA_SETUP.md](OLLAMA_QA_SETUP.md))

### Quick Install

1. Clone the repository:
   ```bash
   git clone https://github.com/Mukunj-21/SignSpeak.git
   cd SignSpeak
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   cd backend
   pip install -r requirements.txt
   ```

4. (Optional) Set up AI enhancement:
   ```bash
   cp .env.example .env
   # Edit .env and add your OpenAI API key
   # Get one at: https://platform.openai.com/api-keys
   ```

5. Start the application:
   ```bash
   python app.py
   ```

6. Open your browser and visit:
   ```
   http://localhost:5000
   ```

📖 **For detailed setup instructions**: See [QUICKSTART.md](QUICKSTART.md)  
🤖 **For AI features documentation**: See [AI_FEATURES.md](AI_FEATURES.md)

## 📚 How It Works

SignSpeak uses a deep learning approach combined with generative AI to recognize and translate sign language:

1. **Hand Detection**: OpenCV processes webcam input to isolate hand regions
2. **Feature Extraction**: Key points and hand shapes are identified and normalized
3. **Classification**: Our custom CNN model classifies the hand gesture into corresponding letters/words
4. **AI Enhancement**: GPT transforms raw sign sequences into natural, fluent sentences
5. **Translation**: Recognized signs are converted to text with multiple style options (casual/formal)
6. **Speech Output**: Natural-sounding text-to-speech converts enhanced sentences to audio
7. **User Interface**: Results are displayed in real-time through our Flask-powered web interface

### Example Workflow

```
1. User signs: M-E → G-O → S-T-O-R-E
2. Raw detection: "ME GO STORE"
3. AI enhancement: "I'm going to the store." (casual) or "I am going to the store." (formal)
4. Speech output: Natural voice speaks the enhanced sentence
```

## 🧠 Model Architecture

Our CNN model was trained on a dataset of over 20,000 sign language images covering the ASL alphabet and common phrases. The architecture includes:

- Input layer for normalized hand images (64x64x1)
- 4 convolutional layers with max-pooling
- 2 fully connected layers
- Dropout layers to prevent overfitting
- SoftMax output layer for multi-class classification

## 🛠️ Technology Stack

- **Backend**: Python, Flask
- **Machine Learning**: TensorFlow, Keras, OpenCV, MediaPipe
- **AI Enhancement**: OpenAI GPT-3.5/GPT-4 (cloud), Ollama (local)
- **Frontend**: HTML, CSS, JavaScript, Bootstrap
- **Data Processing**: NumPy, Pandas
- **Deployment**: Docker support for easy deployment

## � Q&A Feature with Ollama

SignSpeak now includes a powerful **local AI Q&A system** using Ollama! This allows you to:

- **Ask questions in sign language** and get intelligent AI-powered answers
- **100% Local processing** - no cloud services required, completely private
- **Conversation history** - maintains context for follow-up questions
- **Multiple AI models** - choose from Llama 2, Mistral, Phi, and more
- **Text-to-speech responses** - answers are spoken aloud automatically

### Quick Setup

1. **Install Ollama**:
   ```bash
   # macOS/Linux
   curl -fsSL https://ollama.com/install.sh | sh
   
   # Or download from: https://ollama.com/download
   ```

2. **Pull a model**:
   ```bash
   ollama pull llama2  # Recommended for beginners
   ```

3. **Start Ollama**:
   ```bash
   ollama serve
   ```

4. **Start SignSpeak** and begin asking questions!

For detailed setup instructions, model comparisons, and troubleshooting, see [OLLAMA_QA_SETUP.md](OLLAMA_QA_SETUP.md).

## �📋 Future Roadmap

- [x] Basic ASL alphabet recognition
- [x] Web interface implementation
- [x] Real-time processing optimization
- [x] AI-powered sentence enhancement with GPT
- [x] Multiple translation styles (casual/formal/neutral)
- [x] Natural text-to-speech output
- [x] Local AI Q&A system with Ollama
- [ ] Support for full ASL grammar and syntax
- [ ] Multi-language support (Spanish, French, etc.)
- [ ] Mobile application development
- [ ] Offline mode functionality
- [ ] Multi-language sign language support
- [ ] Integration with AR/VR platforms
- [ ] Conversation history and analytics
- [ ] Custom vocabulary training

## 👥 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [ASL Dataset Contributors](https://www.kaggle.com/datasets/grassknoted/asl-alphabet)
- All open-source libraries and tools used in this project

## 📬 Contact

Project Creator: Mukunj - [GitHub Profile](https://github.com/Mukunj-21)

Project Link: [https://github.com/Mukunj-21/SignSpeak](https://github.com/Mukunj-21/SignSpeak)

---

<p align="center">Made with ❤️ for the deaf and hard-of-hearing community</p>
