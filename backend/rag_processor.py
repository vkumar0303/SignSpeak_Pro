"""
RAG (Retrieval Augmented Generation) Processor
Handles document upload, parsing, embedding, and context retrieval for Q&A
"""

import os
import logging
from typing import List, Dict, Optional, Tuple
from datetime import datetime
import hashlib

# Document parsing libraries
try:
    import PyPDF2
    pdf_available = True
except ImportError:
    pdf_available = False
    logging.warning("PyPDF2 not installed. PDF support disabled. Install with: pip install PyPDF2")

try:
    from docx import Document
    docx_available = True
except ImportError:
    docx_available = False
    logging.warning("python-docx not installed. DOCX support disabled. Install with: pip install python-docx")

# Vector storage and embedding
try:
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    embedding_available = True
except ImportError:
    embedding_available = False
    logging.warning("sklearn not installed. Install with: pip install scikit-learn")


class RAGProcessor:
    """Process documents and provide context for Q&A"""
    
    def __init__(self, upload_folder: str = "./uploads"):
        self.upload_folder = upload_folder
        self.documents: Dict[str, Dict] = {}  # Store parsed documents
        self.vectorizer = TfidfVectorizer(max_features=500, stop_words='english') if embedding_available else None
        self.document_vectors = {}
        
        # Create upload folder if it doesn't exist
        os.makedirs(upload_folder, exist_ok=True)
        
        logging.info("✅ RAG Processor initialized")
        logging.info(f"   PDF support: {'✅' if pdf_available else '❌'}")
        logging.info(f"   DOCX support: {'✅' if docx_available else '❌'}")
        logging.info(f"   Embedding support: {'✅' if embedding_available else '❌'}")
    
    def parse_txt(self, file_path: str) -> str:
        """Parse text file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            logging.error(f"Error parsing TXT: {e}")
            return ""
    
    def parse_pdf(self, file_path: str) -> str:
        """Parse PDF file"""
        if not pdf_available:
            return "PDF parsing not available. Please install PyPDF2."
        
        try:
            text = []
            with open(file_path, 'rb') as f:
                pdf_reader = PyPDF2.PdfReader(f)
                for page in pdf_reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text.append(page_text)
            return "\n".join(text)
        except Exception as e:
            logging.error(f"Error parsing PDF: {e}")
            return ""
    
    def parse_docx(self, file_path: str) -> str:
        """Parse DOCX file"""
        if not docx_available:
            return "DOCX parsing not available. Please install python-docx."
        
        try:
            doc = Document(file_path)
            text = []
            for paragraph in doc.paragraphs:
                if paragraph.text.strip():
                    text.append(paragraph.text)
            return "\n".join(text)
        except Exception as e:
            logging.error(f"Error parsing DOCX: {e}")
            return ""
    
    def parse_doc(self, file_path: str) -> str:
        """Parse DOC file (legacy Word format)"""
        # DOC files are binary and require more complex parsing
        # For now, return error message
        return "Legacy DOC format not supported. Please convert to DOCX or PDF."
    
    def chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        """Split text into overlapping chunks for better context retrieval"""
        if not text:
            return []
        
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk = ' '.join(words[i:i + chunk_size])
            if chunk:
                chunks.append(chunk)
        
        return chunks
    
    def upload_document(self, file_path: str, original_filename: str) -> Dict:
        """
        Upload and process a document
        
        Args:
            file_path: Path to the uploaded file
            original_filename: Original filename
            
        Returns:
            Dict with status and document info
        """
        try:
            # Determine file type
            ext = os.path.splitext(original_filename)[1].lower()
            
            # Parse document based on type
            if ext == '.txt':
                text = self.parse_txt(file_path)
            elif ext == '.pdf':
                text = self.parse_pdf(file_path)
            elif ext == '.docx':
                text = self.parse_docx(file_path)
            elif ext == '.doc':
                text = self.parse_doc(file_path)
            else:
                return {
                    'status': 'error',
                    'message': f'Unsupported file type: {ext}'
                }
            
            if not text:
                return {
                    'status': 'error',
                    'message': 'Failed to extract text from document'
                }
            
            # Generate document ID
            doc_id = hashlib.md5(f"{original_filename}{datetime.now().isoformat()}".encode()).hexdigest()
            
            # Chunk the text
            chunks = self.chunk_text(text)
            
            # Store document info
            self.documents[doc_id] = {
                'id': doc_id,
                'filename': original_filename,
                'text': text,
                'chunks': chunks,
                'uploaded_at': datetime.now().isoformat(),
                'word_count': len(text.split()),
                'chunk_count': len(chunks)
            }
            
            # Create embeddings if available
            if embedding_available and chunks:
                try:
                    # Fit vectorizer on all chunks
                    all_chunks = []
                    for doc in self.documents.values():
                        all_chunks.extend(doc['chunks'])
                    
                    self.vectorizer.fit(all_chunks)
                    
                    # Transform current document chunks
                    self.document_vectors[doc_id] = self.vectorizer.transform(chunks)
                except Exception as e:
                    logging.warning(f"Failed to create embeddings: {e}")
            
            logging.info(f"📄 Document uploaded: {original_filename} ({len(chunks)} chunks)")
            
            return {
                'status': 'success',
                'doc_id': doc_id,
                'filename': original_filename,
                'word_count': len(text.split()),
                'chunk_count': len(chunks)
            }
            
        except Exception as e:
            logging.error(f"Error uploading document: {e}")
            return {
                'status': 'error',
                'message': str(e)
            }
    
    def retrieve_context(self, query: str, doc_id: Optional[str] = None, top_k: int = 3) -> str:
        """
        Retrieve relevant context from documents based on query
        
        Args:
            query: User's question
            doc_id: Optional specific document ID to search in
            top_k: Number of top chunks to retrieve
            
        Returns:
            Combined context string
        """
        if not self.documents:
            return ""
        
        try:
            # If specific document requested
            if doc_id and doc_id in self.documents:
                docs_to_search = {doc_id: self.documents[doc_id]}
            else:
                docs_to_search = self.documents
            
            # If embeddings available, use vector similarity
            if embedding_available and self.vectorizer:
                all_chunks = []
                chunk_sources = []  # Track which doc each chunk came from
                
                for did, doc in docs_to_search.items():
                    for chunk in doc['chunks']:
                        all_chunks.append(chunk)
                        chunk_sources.append((did, chunk))
                
                if not all_chunks:
                    return ""
                
                # Transform query
                query_vector = self.vectorizer.transform([query])
                
                # Get vectors for all chunks
                chunk_vectors = self.vectorizer.transform(all_chunks)
                
                # Calculate similarities
                similarities = cosine_similarity(query_vector, chunk_vectors)[0]
                
                # Get top-k most similar chunks
                top_indices = np.argsort(similarities)[-top_k:][::-1]
                
                relevant_chunks = []
                for idx in top_indices:
                    if similarities[idx] > 0.1:  # Minimum similarity threshold
                        relevant_chunks.append(all_chunks[idx])
                
                return "\n\n".join(relevant_chunks)
            
            else:
                # Fallback: use keyword matching
                query_words = set(query.lower().split())
                chunk_scores = []
                
                for doc in docs_to_search.values():
                    for chunk in doc['chunks']:
                        chunk_words = set(chunk.lower().split())
                        score = len(query_words & chunk_words)
                        if score > 0:
                            chunk_scores.append((score, chunk))
                
                # Sort by score and get top-k
                chunk_scores.sort(reverse=True, key=lambda x: x[0])
                relevant_chunks = [chunk for score, chunk in chunk_scores[:top_k]]
                
                return "\n\n".join(relevant_chunks)
                
        except Exception as e:
            logging.error(f"Error retrieving context: {e}")
            return ""
    
    def delete_document(self, doc_id: str) -> bool:
        """Delete a document from storage"""
        if doc_id in self.documents:
            del self.documents[doc_id]
            if doc_id in self.document_vectors:
                del self.document_vectors[doc_id]
            logging.info(f"🗑️  Document deleted: {doc_id}")
            return True
        return False
    
    def get_documents(self) -> List[Dict]:
        """Get list of all uploaded documents"""
        return [
            {
                'id': doc['id'],
                'filename': doc['filename'],
                'uploaded_at': doc['uploaded_at'],
                'word_count': doc['word_count'],
                'chunk_count': doc['chunk_count']
            }
            for doc in self.documents.values()
        ]
    
    def get_document_info(self, doc_id: str) -> Optional[Dict]:
        """Get information about a specific document"""
        if doc_id in self.documents:
            doc = self.documents[doc_id]
            return {
                'id': doc['id'],
                'filename': doc['filename'],
                'uploaded_at': doc['uploaded_at'],
                'word_count': doc['word_count'],
                'chunk_count': doc['chunk_count'],
                'preview': doc['text'][:500] + '...' if len(doc['text']) > 500 else doc['text']
            }
        return None
    
    def clear_all_documents(self):
        """Clear all documents"""
        self.documents.clear()
        self.document_vectors.clear()
        logging.info("🗑️  All documents cleared")
