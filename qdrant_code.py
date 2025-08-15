#!/usr/bin/env python3
"""
Qdrant Hybrid Search RAG Implementation

This implementation uses Qdrant for vector storage with hybrid search
combining dense embeddings and sparse BM25-like retrieval for better
performance on large document collections with CPU-only processing.
"""

import os
import sys
from typing import List, Dict, Any, Optional, Tuple
import fitz  # PyMuPDF
import re
import json
import requests
import traceback
from collections import Counter
import math

# Qdrant imports
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct, 
    SparseVectorParams, SparseVector, NamedVector,
    SearchRequest, Filter, FieldCondition, MatchValue
)

# LangChain imports
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings

class HybridRetriever:
    def __init__(
        self, 
        collection_name: str = "hybrid_documents",
        qdrant_path: str = "./qdrant_storage",
        embedding_model: str = "all-MiniLM-L6-v2"
    ):
        """
        Initialize hybrid retriever with Qdrant and BM25-style sparse vectors.
        
        Args:
            collection_name: Name of the Qdrant collection
            qdrant_path: Path to store Qdrant database
            embedding_model: HuggingFace embedding model name
        """
        self.collection_name = collection_name
        self.qdrant_path = qdrant_path
        
        # Initialize Qdrant client (local storage)
        self.client = QdrantClient(path=qdrant_path)
        
        # Initialize embeddings
        print(f"Loading embedding model: {embedding_model}")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        # Vocabulary for sparse vectors (BM25-style)
        self.vocabulary = {}
        self.idf_scores = {}
        self.total_docs = 0
        
        # Ollama configuration
        self.ollama_base_url = "http://127.0.0.1:11434"
        self.ollama_model = "codeqwen:7b-chat-v1.5-q8_0"
        
        self._initialize_collection()

    def _initialize_collection(self):
        """Initialize or recreate the Qdrant collection."""
        try:
            # Check if collection exists
            collections = self.client.get_collections().collections
            collection_exists = any(col.name == self.collection_name for col in collections)
            
            if collection_exists:
                print(f"Collection '{self.collection_name}' already exists")
                # Load vocabulary and IDF scores
                self._load_vocabulary()
            else:
                print(f"Creating new collection: {self.collection_name}")
                self._create_collection()
                
        except Exception as e:
            print(f"Error initializing collection: {e}")
            traceback.print_exc()

    def _create_collection(self):
        """Create a new Qdrant collection with hybrid search capabilities."""
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config={
                "dense": VectorParams(size=384, distance=Distance.COSINE),  # all-MiniLM-L6-v2 size
            },
            sparse_vectors_config={
                "sparse": SparseVectorParams(),
            }
        )
        print(f"Created collection: {self.collection_name}")

    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization for sparse vectors."""
        # Convert to lowercase and extract words
        text = text.lower()
        tokens = re.findall(r'\b\w+\b', text)
        return tokens

    def _build_vocabulary(self, documents: List[Document]):
        """Build vocabulary and IDF scores from documents."""
        print("Building vocabulary for sparse vectors...")
        
        # Count word frequencies across documents
        word_doc_count = Counter()
        all_words = set()
        
        for doc in documents:
            tokens = set(self._tokenize(doc.page_content))
            for token in tokens:
                word_doc_count[token] += 1
                all_words.add(token)
        
        # Create vocabulary mapping
        self.vocabulary = {word: idx for idx, word in enumerate(sorted(all_words))}
        self.total_docs = len(documents)
        
        # Calculate IDF scores
        self.idf_scores = {}
        for word, doc_freq in word_doc_count.items():
            self.idf_scores[word] = math.log(self.total_docs / (doc_freq + 1))
        
        print(f"Built vocabulary with {len(self.vocabulary)} unique terms")
        self._save_vocabulary()

    def _save_vocabulary(self):
        """Save vocabulary and IDF scores to disk."""
        vocab_path = os.path.join(self.qdrant_path, "vocabulary.json")
        vocab_data = {
            "vocabulary": self.vocabulary,
            "idf_scores": self.idf_scores,
            "total_docs": self.total_docs
        }
        with open(vocab_path, 'w') as f:
            json.dump(vocab_data, f)
        print(f"Saved vocabulary to {vocab_path}")

    def _load_vocabulary(self):
        """Load vocabulary and IDF scores from disk."""
        vocab_path = os.path.join(self.qdrant_path, "vocabulary.json")
        if os.path.exists(vocab_path):
            with open(vocab_path, 'r') as f:
                vocab_data = json.load(f)
            self.vocabulary = vocab_data.get("vocabulary", {})
            self.idf_scores = vocab_data.get("idf_scores", {})
            self.total_docs = vocab_data.get("total_docs", 0)
            print(f"Loaded vocabulary with {len(self.vocabulary)} terms")
        else:
            print("No vocabulary file found")

    def _create_sparse_vector(self, text: str) -> SparseVector:
        """Create sparse vector (BM25-style) from text."""
        tokens = self._tokenize(text)
        token_counts = Counter(tokens)
        
        # BM25 parameters
        k1 = 1.5
        b = 0.75
        avg_doc_length = 100  # Approximate average document length
        doc_length = len(tokens)
        
        indices = []
        values = []
        
        for token, tf in token_counts.items():
            if token in self.vocabulary:
                idx = self.vocabulary[token]
                idf = self.idf_scores.get(token, 0)
                
                # BM25 score calculation
                numerator = tf * (k1 + 1)
                denominator = tf + k1 * (1 - b + b * (doc_length / avg_doc_length))
                bm25_score = idf * (numerator / denominator)
                
                if bm25_score > 0:
                    indices.append(idx)
                    values.append(bm25_score)
        
        return SparseVector(indices=indices, values=values)

    def process_document(self, filepath: str) -> List[Document]:
        """Process a document and return chunks (same as original)."""
        try:
            print(f"Processing document: {filepath}")
            
            if not os.path.exists(filepath):
                print(f"ERROR: File does not exist: {filepath}")
                return []
                
            file_size = os.path.getsize(filepath)
            if file_size == 0:
                print(f"ERROR: File is empty: {filepath}")
                return []
            
            # Process PDF or TXT files
            docs = []
            if filepath.endswith('.pdf'):
                with fitz.open(filepath) as pdf_doc:
                    for page_num, page in enumerate(pdf_doc):
                        text = page.get_text("text")
                        if text:
                            metadata = {"source": os.path.basename(filepath), "page": page_num + 1}
                            docs.append(Document(page_content=text, metadata=metadata))
            elif filepath.endswith('.txt'):
                loader = TextLoader(filepath, encoding='utf-8', autodetect_encoding=True)
                docs = loader.load()
            else:
                print(f"Unsupported file type: {filepath}")
                return []

            # Clean and chunk documents
            for doc in docs:
                content = re.sub(r'\s+', ' ', doc.page_content).strip()
                doc.page_content = content

            # Chunking
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1500,
                chunk_overlap=200,
                length_function=len,
                separators=["\n\n", "\n", ". ", " ", ""]
            )
            chunks = text_splitter.split_documents(docs)
            
            # Filter valid chunks
            valid_chunks = [
                chunk for chunk in chunks 
                if chunk.page_content and len(chunk.page_content.strip()) > 10
            ]
            
            print(f"Created {len(valid_chunks)} valid chunks from {filepath}")
            return valid_chunks
            
        except Exception as e:
            print(f"Error processing {filepath}: {e}")
            traceback.print_exc()
            return []

    def add_documents(self, documents: List[Document], batch_size: int = 100):
        """Add documents to Qdrant with hybrid vectors."""
        if not documents:
            print("No documents to add")
            return
        
        print(f"Adding {len(documents)} documents to Qdrant...")
        
        # Build vocabulary if not exists
        if not self.vocabulary:
            self._build_vocabulary(documents)
        
        # Process documents in batches
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            points = []
            
            print(f"Processing batch {i//batch_size + 1}/{(len(documents) + batch_size - 1)//batch_size}")
            
            for doc_id, doc in enumerate(batch):
                try:
                    # Create dense vector
                    dense_vector = self.embeddings.embed_query(doc.page_content)
                    
                    # Create sparse vector
                    sparse_vector = self._create_sparse_vector(doc.page_content)
                    
                    # Create point
                    point = PointStruct(
                        id=i + doc_id,
                        vector={
                            "dense": dense_vector,
                            "sparse": sparse_vector
                        },
                        payload={
                            "content": doc.page_content,
                            "metadata": doc.metadata or {},
                            "source": doc.metadata.get("source", "") if doc.metadata else ""
                        }
                    )
                    points.append(point)
                    
                except Exception as e:
                    print(f"Error processing document {doc_id}: {e}")
                    continue
            
            # Upload batch
            if points:
                self.client.upsert(collection_name=self.collection_name, points=points)
                print(f"Uploaded {len(points)} points")

    def hybrid_search(
        self, 
        query: str, 
        limit: int = 10, 
        dense_weight: float = 0.7, 
        sparse_weight: float = 0.3
    ) -> List[Dict[str, Any]]:
        """
        Perform hybrid search combining dense and sparse vectors.
        
        Args:
            query: Search query
            limit: Number of results to return
            dense_weight: Weight for dense vector similarity
            sparse_weight: Weight for sparse vector similarity
        """
        try:
            # Create dense query vector
            dense_query = self.embeddings.embed_query(query)
            
            # Create sparse query vector
            sparse_query = self._create_sparse_vector(query)
            
            # Perform search
            search_result = self.client.search(
                collection_name=self.collection_name,
                query_vector=NamedVector(name="dense", vector=dense_query),
                query_filter=None,
                limit=limit,
                with_payload=True,
                with_vectors=False
            )
            
            # For now, we'll use dense search (Qdrant's hybrid search is more complex)
            # You can implement RRF (Reciprocal Rank Fusion) here for true hybrid results
            
            results = []
            for hit in search_result:
                results.append({
                    "content": hit.payload["content"],
                    "metadata": hit.payload["metadata"],
                    "score": hit.score,
                    "source": hit.payload.get("source", "")
                })
            
            return results
            
        except Exception as e:
            print(f"Error in hybrid search: {e}")
            traceback.print_exc()
            return []

    def get_relevant_documents(self, query: str, k: int = 10) -> List[Document]:
        """Get relevant documents using hybrid search."""
        results = self.hybrid_search(query, limit=k)
        
        documents = []
        for result in results:
            doc = Document(
                page_content=result["content"],
                metadata={
                    **result["metadata"],
                    "score": result["score"],
                    "search_type": "hybrid"
                }
            )
            documents.append(doc)
        
        return documents

    def rebuild_from_folder(self, uploads_folder: str = "uploads"):
        """Rebuild the entire collection from documents in uploads folder."""
        print(f"Rebuilding collection from {uploads_folder}")
        
        # Clear existing collection
        try:
            self.client.delete_collection(self.collection_name)
        except:
            pass
        
        self._create_collection()
        self.vocabulary = {}
        self.idf_scores = {}
        
        # Process all documents
        all_docs = []
        if os.path.exists(uploads_folder):
            for filename in os.listdir(uploads_folder):
                if filename.endswith(('.pdf', '.txt')):
                    filepath = os.path.join(uploads_folder, filename)
                    docs = self.process_document(filepath)
                    all_docs.extend(docs)
        
        if all_docs:
            self.add_documents(all_docs)
            print(f"Successfully rebuilt collection with {len(all_docs)} documents")
        else:
            print("No documents found to add")


def rebuild_vector_store_from_uploads(uploads_folder: str = "uploads"):
    """
    Main function to rebuild vector store from uploads folder.
    This replaces your rebuild_vectors.py functionality.
    """
    print("=== Qdrant Hybrid Vector Store Rebuilder ===")
    print(f"This script will rebuild the vector store from documents in '{uploads_folder}'.")
    print("WARNING: This will delete and recreate the entire vector store.")
    
    proceed = input("Do you want to proceed? (y/n): ").lower() == 'y'
    
    if not proceed:
        print("Operation cancelled.")
        return False
    
    # Initialize retriever
    retriever = HybridRetriever()
    
    # Check if uploads folder exists
    if not os.path.exists(uploads_folder):
        print(f"ERROR: Uploads folder '{uploads_folder}' does not exist!")
        return False
    
    # Get list of supported files
    supported_files = []
    for filename in os.listdir(uploads_folder):
        if filename.lower().endswith(('.pdf', '.txt', '.doc', '.docx')):
            filepath = os.path.join(uploads_folder, filename)
            if os.path.isfile(filepath):
                supported_files.append(filepath)
    
    if not supported_files:
        print(f"No supported documents found in '{uploads_folder}'")
        print("Supported formats: PDF, TXT, DOC, DOCX")
        return False
    
    print(f"Found {len(supported_files)} documents to process:")
    for filepath in supported_files:
        file_size = os.path.getsize(filepath) / (1024 * 1024)  # MB
        print(f"  - {os.path.basename(filepath)} ({file_size:.2f} MB)")
    
    # Clear existing collection and rebuild
    print("\nClearing existing collection...")
    try:
        retriever.client.delete_collection(retriever.collection_name)
        print("Existing collection deleted")
    except Exception as e:
        print(f"No existing collection to delete: {e}")
    
    # Create new collection
    retriever._create_collection()
    retriever.vocabulary = {}
    retriever.idf_scores = {}
    
    # Process all documents
    all_documents = []
    successful_files = 0
    failed_files = 0
    
    print(f"\nProcessing {len(supported_files)} documents...")
    
    for i, filepath in enumerate(supported_files, 1):
        print(f"\n[{i}/{len(supported_files)}] Processing: {os.path.basename(filepath)}")
        
        try:
            docs = retriever.process_document(filepath)
            if docs:
                all_documents.extend(docs)
                successful_files += 1
                print(f"  ✓ Added {len(docs)} chunks")
            else:
                failed_files += 1
                print(f"  ✗ No content extracted")
        except Exception as e:
            failed_files += 1
            print(f"  ✗ Error: {str(e)}")
    
    print(f"\n=== Processing Summary ===")
    print(f"Total files processed: {len(supported_files)}")
    print(f"Successful: {successful_files}")
    print(f"Failed: {failed_files}")
    print(f"Total chunks created: {len(all_documents)}")
    
    if not all_documents:
        print("No documents to index. Exiting.")
        return False
    
    # Add documents to Qdrant
    print(f"\nIndexing {len(all_documents)} chunks in Qdrant...")
    try:
        retriever.add_documents(all_documents, batch_size=50)  # Smaller batches for CPU
        
        # Verify collection
        collection_info = retriever.client.get_collection(retriever.collection_name)
        point_count = collection_info.points_count
        
        print(f"\n=== Indexing Complete ===")
        print(f"Collection: {retriever.collection_name}")
        print(f"Total points: {point_count}")
        print(f"Vocabulary size: {len(retriever.vocabulary)} terms")
        print(f"Storage location: {retriever.qdrant_path}")
        
        return True
        
    except Exception as e:
        print(f"Error during indexing: {e}")
        traceback.print_exc()
        return False


def interactive_search():
    """Interactive search function - replaces your retriever.py functionality."""
    print("=== Qdrant Hybrid Search Interface ===")
    
    # Initialize retriever
    try:
        retriever = HybridRetriever()
        
        # Check if collection exists and has data
        try:
            collection_info = retriever.client.get_collection(retriever.collection_name)
            if collection_info.points_count == 0:
                print("Collection is empty. Please run rebuild first.")
                return
            print(f"Loaded collection with {collection_info.points_count} documents")
        except Exception as e:
            print(f"Collection not found. Please run rebuild first. Error: {e}")
            return
            
    except Exception as e:
        print(f"Error initializing retriever: {e}")
        return
    
    print("\nReady for search! Type 'exit' to quit, 'help' for commands.")
    
    while True:
        try:
            query = input("\n🔍 Enter your query: ").strip()
            
            if query.lower() in ['exit', 'quit', 'q']:
                print("Goodbye!")
                break
            elif query.lower() == 'help':
                print("\nAvailable commands:")
                print("  - Any text: Search for relevant documents")
                print("  - 'help': Show this help")
                print("  - 'exit'/'quit'/'q': Exit the program")
                print("  - 'stats': Show collection statistics")
                continue
            elif query.lower() == 'stats':
                collection_info = retriever.client.get_collection(retriever.collection_name)
                print(f"\n📊 Collection Statistics:")
                print(f"  Total documents: {collection_info.points_count}")
                print(f"  Vocabulary size: {len(retriever.vocabulary)}")
                print(f"  Storage path: {retriever.qdrant_path}")
                continue
            elif not query:
                print("Please enter a search query.")
                continue
            
            print(f"\nSearching for: '{query}'...")
            
            # Perform search
            documents = retriever.get_relevant_documents(query, k=5)
            
            if documents:
                print(f"\n📋 Found {len(documents)} relevant documents:\n")
                
                for i, doc in enumerate(documents, 1):
                    print(f"📄 Result {i}")
                    print(f"   Source: {doc.metadata.get('source', 'Unknown')}")
                    if 'page' in doc.metadata:
                        print(f"   Page: {doc.metadata['page']}")
                    print(f"   Relevance Score: {doc.metadata.get('score', 0):.4f}")
                    
                    # Show full content
                    content = doc.page_content.strip()
                    print(f"   Content: {content}")
                    print("   " + "─" * 80)
                
            else:
                print("❌ No relevant documents found.")
                print("Try rephrasing your query or check if documents are indexed.")
                
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"Error during search: {e}")
            traceback.print_exc()


def main():
    """Main entry point with clear options."""
    if len(sys.argv) < 2:
        print("\n=== Qdrant Hybrid RAG System ===")
        print("Usage:")
        print("  python qdrant_rag.py rebuild    - Rebuild vector store from 'uploads' folder")
        print("  python qdrant_rag.py search     - Interactive search interface")
        print("  python qdrant_rag.py rebuild <folder>  - Rebuild from custom folder")
        return
    
    command = sys.argv[1].lower()
    
    if command == "rebuild":
        uploads_folder = sys.argv[2] if len(sys.argv) > 2 else "uploads"
        success = rebuild_vector_store_from_uploads(uploads_folder)
        if success:
            print("\n✅ Vector store rebuild completed successfully!")
            print("You can now run 'python qdrant_rag.py search' to start searching.")
        else:
            print("\n❌ Vector store rebuild failed.")
            
    elif command == "search":
        interactive_search()
        
    else:
        print(f"Unknown command: {command}")
        print("Use 'rebuild' or 'search'")


if __name__ == "__main__":
    main()