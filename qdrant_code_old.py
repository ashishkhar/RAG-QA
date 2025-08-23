#!/usr/bin/env python3
"""
Compatible Enhanced Qdrant Hybrid Search RAG Implementation with Number-Aware Search

This version works with current Qdrant client versions by:
1. Using the correct query_points API parameters
2. Proper sparse vector handling compatible with your version
3. Fallback mechanisms for different API versions
4. Enhanced number-aware search capabilities
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
    Filter, FieldCondition, MatchValue
)

# LangChain imports
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings

class NumberAwareHybridRetriever:
    def __init__(
        self, 
        collection_name: str = "hybrid_documents",
        qdrant_path: str = "./qdrant_storage",
        embedding_model: str = "all-MiniLM-L6-v2"
    ):
        """
        Initialize enhanced hybrid retriever with number-aware search capabilities.
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
        
        # Enhanced vocabulary for number-aware sparse vectors
        self.vocabulary = {}
        self.idf_scores = {}
        self.total_docs = 0
        self.number_pattern = re.compile(r'\b\d+(?:\.\d+)?\b')  # Match numbers including decimals
        
        # Ollama configuration
        self.ollama_base_url = "http://127.0.0.1:11434"
        self.ollama_model = "codeqwen:7b-chat-v1.5-q8_0"
        
        self._initialize_collection()

    def _initialize_collection(self):
        """Initialize or recreate the Qdrant collection."""
        try:
            collections = self.client.get_collections().collections
            collection_exists = any(col.name == self.collection_name for col in collections)
            
            if collection_exists:
                print(f"Collection '{self.collection_name}' already exists")
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
                "dense": VectorParams(size=384, distance=Distance.COSINE),
            },
            sparse_vectors_config={
                "sparse": SparseVectorParams(),
            }
        )
        print(f"Created collection: {self.collection_name}")

    def _enhanced_tokenize(self, text: str) -> List[str]:
        """Enhanced tokenization that preserves numbers and creates number-aware tokens."""
        text = text.lower()
        
        # Extract numbers separately to preserve them
        numbers = self.number_pattern.findall(text)
        
        # Regular word tokenization
        words = re.findall(r'\b[a-zA-Z]+\b', text)
        
        # Create enhanced tokens
        tokens = []
        
        # Add regular words
        tokens.extend(words)
        
        # Add numbers as exact tokens
        for num in numbers:
            tokens.append(f"NUM_{num}")  # Prefix numbers to make them distinct
            
        # Add alphanumeric codes (like section numbers, GST codes, etc.)
        codes = re.findall(r'\b[a-zA-Z]*\d+[a-zA-Z]*\b', text)
        for code in codes:
            if code not in numbers:  # Avoid duplicating pure numbers
                tokens.append(f"CODE_{code.lower()}")
        
        # Add bigrams for better context (especially useful for "section 12" type queries)
        words_and_numbers = re.findall(r'\b\w+\b', text.lower())
        for i in range(len(words_and_numbers) - 1):
            bigram = f"{words_and_numbers[i]}_{words_and_numbers[i+1]}"
            tokens.append(f"BIGRAM_{bigram}")
        
        return tokens

    def _build_vocabulary(self, documents: List[Document]):
        """Build enhanced vocabulary with number awareness."""
        print("Building enhanced vocabulary for number-aware sparse vectors...")
        
        word_doc_count = Counter()
        all_words = set()
        
        for doc in documents:
            tokens = set(self._enhanced_tokenize(doc.page_content))
            for token in tokens:
                word_doc_count[token] += 1
                all_words.add(token)
        
        # Create vocabulary mapping
        self.vocabulary = {word: idx for idx, word in enumerate(sorted(all_words))}
        self.total_docs = len(documents)
        
        # Calculate IDF scores with smoothing
        self.idf_scores = {}
        for word, doc_freq in word_doc_count.items():
            # Enhanced IDF calculation with better smoothing
            self.idf_scores[word] = math.log((self.total_docs + 1) / (doc_freq + 1)) + 1
        
        print(f"Built enhanced vocabulary with {len(self.vocabulary)} unique terms")
        
        # Show some statistics about number tokens
        number_tokens = [w for w in self.vocabulary.keys() if w.startswith('NUM_')]
        code_tokens = [w for w in self.vocabulary.keys() if w.startswith('CODE_')]
        bigram_tokens = [w for w in self.vocabulary.keys() if w.startswith('BIGRAM_')]
        
        print(f"  - Number tokens: {len(number_tokens)}")
        print(f"  - Code tokens: {len(code_tokens)}")
        print(f"  - Bigram tokens: {len(bigram_tokens)}")
        
        self._save_vocabulary()

    def _save_vocabulary(self):
        """Save vocabulary and IDF scores to disk."""
        vocab_path = os.path.join(self.qdrant_path, "enhanced_vocabulary.json")
        vocab_data = {
            "vocabulary": self.vocabulary,
            "idf_scores": self.idf_scores,
            "total_docs": self.total_docs
        }
        with open(vocab_path, 'w') as f:
            json.dump(vocab_data, f)
        print(f"Saved enhanced vocabulary to {vocab_path}")

    def _load_vocabulary(self):
        """Load vocabulary and IDF scores from disk."""
        vocab_path = os.path.join(self.qdrant_path, "enhanced_vocabulary.json")
        if os.path.exists(vocab_path):
            with open(vocab_path, 'r') as f:
                vocab_data = json.load(f)
            self.vocabulary = vocab_data.get("vocabulary", {})
            self.idf_scores = vocab_data.get("idf_scores", {})
            self.total_docs = vocab_data.get("total_docs", 0)
            print(f"Loaded enhanced vocabulary with {len(self.vocabulary)} terms")
        else:
            print("No enhanced vocabulary file found")

    def _create_enhanced_sparse_vector(self, text: str) -> SparseVector:
        """Create enhanced sparse vector with better BM25 and number handling."""
        tokens = self._enhanced_tokenize(text)
        token_counts = Counter(tokens)
        
        # Enhanced BM25 parameters
        k1 = 1.2  # Term frequency saturation parameter
        b = 0.75  # Length normalization parameter
        
        # Calculate average document length (approximate)
        avg_doc_length = 200  # Increased for better length normalization
        doc_length = len(tokens)
        
        indices = []
        values = []
        
        for token, tf in token_counts.items():
            if token in self.vocabulary:
                idx = self.vocabulary[token]
                idf = self.idf_scores.get(token, 0)
                
                # Enhanced BM25 score calculation
                numerator = tf * (k1 + 1)
                denominator = tf + k1 * (1 - b + b * (doc_length / avg_doc_length))
                bm25_score = idf * (numerator / denominator)
                
                # Boost scores for exact number/code matches
                if token.startswith('NUM_') or token.startswith('CODE_'):
                    bm25_score *= 2.0  # Give extra weight to numbers and codes
                elif token.startswith('BIGRAM_'):
                    bm25_score *= 1.5  # Moderate boost for bigrams
                
                if bm25_score > 0:
                    indices.append(idx)
                    values.append(bm25_score)
        
        return SparseVector(indices=indices, values=values)

    def process_document(self, filepath: str) -> List[Document]:
        """Process a document and return chunks (enhanced version)."""
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

            # Enhanced text cleaning that preserves numbers
            for doc in docs:
                # Clean while preserving important numerical information
                content = re.sub(r'\s+', ' ', doc.page_content)
                content = content.strip()
                doc.page_content = content

            # Enhanced chunking strategy
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1500,
                chunk_overlap=300,  # Increased overlap to preserve context
                length_function=len,
                separators=["\n\n", "\n", ". ", "; ", ", ", " ", ""]
            )
            chunks = text_splitter.split_documents(docs)
            
            # Enhanced chunk filtering
            valid_chunks = []
            for chunk in chunks:
                content = chunk.page_content.strip()
                if content and len(content) > 10:
                    # Add chunk index for better tracking
                    chunk.metadata["chunk_id"] = len(valid_chunks)
                    valid_chunks.append(chunk)
            
            print(f"Created {len(valid_chunks)} valid chunks from {filepath}")
            return valid_chunks
            
        except Exception as e:
            print(f"Error processing {filepath}: {e}")
            traceback.print_exc()
            return []

    def add_documents(self, documents: List[Document], batch_size: int = 50):
        """Add documents to Qdrant with enhanced hybrid vectors."""
        if not documents:
            print("No documents to add")
            return
        
        print(f"Adding {len(documents)} documents to Qdrant with enhanced indexing...")
        
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
                    
                    # Create enhanced sparse vector
                    sparse_vector = self._create_enhanced_sparse_vector(doc.page_content)
                    
                    # Enhanced payload with searchable fields
                    payload = {
                        "content": doc.page_content,
                        "metadata": doc.metadata or {},
                        "source": doc.metadata.get("source", "") if doc.metadata else "",
                        "page": doc.metadata.get("page", 0) if doc.metadata else 0,
                        "chunk_id": doc.metadata.get("chunk_id", doc_id) if doc.metadata else doc_id
                    }
                    
                    # Extract numbers and codes for exact matching
                    numbers = self.number_pattern.findall(doc.page_content)
                    codes = re.findall(r'\b[a-zA-Z]*\d+[a-zA-Z]*\b', doc.page_content.lower())
                    
                    payload["numbers"] = numbers
                    payload["codes"] = codes
                    
                    # Create point
                    point = PointStruct(
                        id=i + doc_id,
                        vector={
                            "dense": dense_vector,
                            "sparse": sparse_vector
                        },
                        payload=payload
                    )
                    points.append(point)
                    
                except Exception as e:
                    print(f"Error processing document {doc_id}: {e}")
                    continue
            
            # Upload batch
            if points:
                self.client.upsert(collection_name=self.collection_name, points=points)
                print(f"Uploaded {len(points)} points")

    def enhanced_hybrid_search(
        self, 
        query: str, 
        limit: int = 10, 
        dense_weight: float = 0.4, 
        sparse_weight: float = 0.6
    ) -> List[Dict[str, Any]]:
        """
        Enhanced hybrid search with RRF and number awareness - COMPATIBLE VERSION.
        """
        try:
            print(f"Performing enhanced hybrid search for: '{query}'")
            
            # Create query vectors
            dense_query = self.embeddings.embed_query(query)
            sparse_query = self._create_enhanced_sparse_vector(query)
            
            # Check if query contains numbers
            query_numbers = self.number_pattern.findall(query)
            has_numbers = len(query_numbers) > 0
            
            if has_numbers:
                print(f"Detected numbers in query: {query_numbers}")
            
            dense_results = []
            sparse_results = []
            
            # Try modern query_points API first
            try:
                # Dense search using query_points
                dense_response = self.client.query_points(
                    collection_name=self.collection_name,
                    query=dense_query,
                    using="dense",
                    limit=limit * 2,
                    with_payload=True
                )
                dense_results = dense_response.points if hasattr(dense_response, 'points') else dense_response
                print(f"Dense search successful: {len(dense_results)} results")
                
            except Exception as e:
                print(f"Modern query_points failed: {e}")
                # Fallback to deprecated search method
                try:
                    dense_results = self.client.search(
                        collection_name=self.collection_name,
                        query_vector=dense_query,
                        limit=limit * 2,
                        with_payload=True
                    )
                    print(f"Fallback dense search successful: {len(dense_results)} results")
                except Exception as e2:
                    print(f"All dense search methods failed: {e2}")
                    dense_results = []
            
            # Sparse search (only if sparse vector has content)
            if sparse_query.indices and sparse_query.values:
                try:
                    # Try modern query_points with sparse
                    sparse_response = self.client.query_points(
                        collection_name=self.collection_name,
                        query=sparse_query,
                        using="sparse",
                        limit=limit * 2,
                        with_payload=True
                    )
                    sparse_results = sparse_response.points if hasattr(sparse_response, 'points') else sparse_response
                    print(f"Sparse search successful: {len(sparse_results)} results")
                    
                except Exception as e:
                    print(f"Modern sparse query failed: {e}")
                    # Try fallback sparse search
                    try:
                        sparse_results = self.client.search(
                            collection_name=self.collection_name,
                            query_vector=NamedVector(name="sparse", vector=sparse_query),
                            limit=limit * 2,
                            with_payload=True
                        )
                        print(f"Fallback sparse search successful: {len(sparse_results)} results")
                    except Exception as e2:
                        print(f"All sparse search methods failed: {e2}")
                        sparse_results = []
            else:
                print("No sparse vector created (no matching vocabulary terms)")
            
            # Reciprocal Rank Fusion (RRF)
            rrf_scores = {}
            k = 60  # RRF parameter
            
            # Score dense results
            for rank, result in enumerate(dense_results):
                point_id = result.id
                rrf_score = dense_weight / (k + rank + 1)
                
                # Boost score if query contains numbers and document contains those numbers
                if has_numbers and hasattr(result, 'payload') and result.payload:
                    doc_numbers = result.payload.get('numbers', [])
                    if any(num in doc_numbers for num in query_numbers):
                        rrf_score *= 2.0  # Strong boost for exact number matches
                
                rrf_scores[point_id] = rrf_scores.get(point_id, 0) + rrf_score
            
            # Score sparse results
            for rank, result in enumerate(sparse_results):
                point_id = result.id
                rrf_score = sparse_weight / (k + rank + 1)
                
                # Additional boost for sparse matches with numbers
                if has_numbers and hasattr(result, 'payload') and result.payload:
                    doc_numbers = result.payload.get('numbers', [])
                    if any(num in doc_numbers for num in query_numbers):
                        rrf_score *= 1.5
                
                rrf_scores[point_id] = rrf_scores.get(point_id, 0) + rrf_score
            
            # Get all unique results and sort by RRF score
            all_results_map = {}
            for result in dense_results + sparse_results:
                all_results_map[result.id] = result
            
            # Sort by RRF score
            sorted_results = sorted(
                rrf_scores.items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:limit]
            
            # Format final results
            final_results = []
            for point_id, rrf_score in sorted_results:
                if point_id in all_results_map:
                    result = all_results_map[point_id]
                    payload = result.payload if hasattr(result, 'payload') else {}
                    final_results.append({
                        "content": payload.get("content", ""),
                        "metadata": payload.get("metadata", {}),
                        "score": rrf_score,
                        "source": payload.get("source", ""),
                        "page": payload.get("page", 0),
                        "numbers": payload.get("numbers", []),
                        "search_type": "enhanced_hybrid"
                    })
            
            print(f"Enhanced hybrid search returned {len(final_results)} results")
            return final_results
            
        except Exception as e:
            print(f"Error in enhanced hybrid search: {e}")
            traceback.print_exc()
            return []

    def get_relevant_documents(self, query: str, k: int = 10) -> List[Document]:
        """Get relevant documents using enhanced hybrid search."""
        results = self.enhanced_hybrid_search(query, limit=k)
        
        documents = []
        for result in results:
            doc = Document(
                page_content=result["content"],
                metadata={
                    **result["metadata"],
                    "score": result["score"],
                    "search_type": result["search_type"],
                    "source": result["source"],
                    "page": result["page"],
                    "numbers": result.get("numbers", [])
                }
            )
            documents.append(doc)
        
        return documents

    def rebuild_from_folder(self, uploads_folder: str = "uploads"):
        """Rebuild the entire collection from documents in uploads folder."""
        print(f"Rebuilding enhanced collection from {uploads_folder}")
        
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
            print(f"Successfully rebuilt enhanced collection with {len(all_docs)} documents")
        else:
            print("No documents found to add")


def rebuild_enhanced_vector_store(uploads_folder: str = "uploads"):
    """Rebuild vector store with enhanced number-aware capabilities."""
    print("=== Enhanced Qdrant Number-Aware Vector Store Rebuilder (COMPATIBLE VERSION) ===")
    print(f"This will rebuild the vector store with enhanced number search from '{uploads_folder}'.")
    print("Features: Number-aware tokenization, enhanced BM25, hybrid search with RRF")
    
    proceed = input("Do you want to proceed? (y/n): ").lower() == 'y'
    if not proceed:
        print("Operation cancelled.")
        return False
    
    # Initialize enhanced retriever
    retriever = NumberAwareHybridRetriever()
    
    # Check uploads folder
    if not os.path.exists(uploads_folder):
        print(f"ERROR: Uploads folder '{uploads_folder}' does not exist!")
        return False
    
    # Get supported files
    supported_files = []
    for filename in os.listdir(uploads_folder):
        if filename.lower().endswith(('.pdf', '.txt')):
            filepath = os.path.join(uploads_folder, filename)
            if os.path.isfile(filepath):
                supported_files.append(filepath)
    
    if not supported_files:
        print(f"No supported documents found in '{uploads_folder}'")
        return False
    
    print(f"Found {len(supported_files)} documents to process:")
    for filepath in supported_files:
        file_size = os.path.getsize(filepath) / (1024 * 1024)
        print(f"  - {os.path.basename(filepath)} ({file_size:.2f} MB)")
    
    # Rebuild collection
    retriever.rebuild_from_folder(uploads_folder)
    
    print("\n✅ Enhanced vector store rebuild completed!")
    print("New features available:")
    print("  - Number-aware search (e.g., 'section 12', 'GST 18%')")
    print("  - Enhanced BM25 sparse vectors")
    print("  - Hybrid search with RRF")
    print("  - Better handling of codes and identifiers")
    
    return True


def interactive_enhanced_search():
    """Enhanced interactive search with number awareness - COMPATIBLE VERSION."""
    print("=== Enhanced Qdrant Number-Aware Search Interface (COMPATIBLE) ===")
    print("Optimized for numerical queries and exact term matching!")
    
    try:
        retriever = NumberAwareHybridRetriever()
        
        # Check collection
        try:
            collection_info = retriever.client.get_collection(retriever.collection_name)
            if collection_info.points_count == 0:
                print("Collection is empty. Please run rebuild first.")
                return
            print(f"Loaded enhanced collection with {collection_info.points_count} documents")
        except Exception as e:
            print(f"Collection not found. Please run rebuild first. Error: {e}")
            return
            
    except Exception as e:
        print(f"Error initializing enhanced retriever: {e}")
        return
    
    print("\nReady for enhanced search! Type 'exit' to quit, 'help' for commands.")
    print("Try queries like: 'section 12 igst', 'GST rate 18', 'rule 88A', etc.")
    
    while True:
        try:
            query = input("\n🔍 Enter your query: ").strip()
            
            if query.lower() in ['exit', 'quit', 'q']:
                print("Goodbye!")
                break
            elif query.lower() == 'help':
                print("\nAvailable commands:")
                print("  - Any text: Search for relevant documents (optimized for numbers)")
                print("  - 'help': Show this help")
                print("  - 'exit'/'quit'/'q': Exit the program")
                print("  - 'stats': Show collection statistics")
                print("\nSearch tips:")
                print("  - Use exact numbers: 'section 12', 'rate 18', 'rule 88A'")
                print("  - Combine terms: 'section 12 igst credit'")
                print("  - Try codes: 'GST 18%', 'CGST section 49'")
                continue
            elif query.lower() == 'stats':
                collection_info = retriever.client.get_collection(retriever.collection_name)
                print(f"\n📊 Enhanced Collection Statistics:")
                print(f"  Total documents: {collection_info.points_count}")
                print(f"  Enhanced vocabulary size: {len(retriever.vocabulary)}")
                
                # Show number-related statistics
                num_tokens = len([w for w in retriever.vocabulary.keys() if w.startswith('NUM_')])
                code_tokens = len([w for w in retriever.vocabulary.keys() if w.startswith('CODE_')])
                bigram_tokens = len([w for w in retriever.vocabulary.keys() if w.startswith('BIGRAM_')])
                
                print(f"  Number tokens: {num_tokens}")
                print(f"  Code tokens: {code_tokens}")
                print(f"  Context bigrams: {bigram_tokens}")
                print(f"  Storage path: {retriever.qdrant_path}")
                continue
            elif not query:
                print("Please enter a search query.")
                continue
            
            print(f"\nSearching with enhanced number-aware algorithm (COMPATIBLE)...")
            
            # Detect if query contains numbers
            numbers_in_query = retriever.number_pattern.findall(query)
            if numbers_in_query:
                print(f"🔢 Detected numbers: {numbers_in_query} (optimizing search...)")
            
            # Perform enhanced search
            documents = retriever.get_relevant_documents(query, k=5)
            
            if documents:
                print(f"\n📋 Found {len(documents)} relevant documents:\n")
                
                for i, doc in enumerate(documents, 1):
                    print(f"📄 Result {i}")
                    print(f"   Source: {doc.metadata.get('source', 'Unknown')}")
                    if 'page' in doc.metadata:
                        print(f"   Page: {doc.metadata['page']}")
                    print(f"   Enhanced Score: {doc.metadata.get('score', 0):.4f}")
                    
                    # Show numbers found in document if any
                    doc_numbers = doc.metadata.get('numbers', [])
                    if doc_numbers:
                        print(f"   Numbers in doc: {doc_numbers[:10]}...")  # Show first 10 numbers
                    
                    # Highlight query terms in content (simple highlighting)
                    content = doc.page_content.strip()
                    
                    # Simple highlighting for numbers
                    for num in numbers_in_query:
                        content = re.sub(
                            f'\\b{re.escape(num)}\\b', 
                            f"**{num}**", 
                            content, 
                            flags=re.IGNORECASE
                        )
                    
                    print(f"   Content: {content}")
                    print("   " + "─" * 80)
                
            else:
                print("❌ No relevant documents found.")
                print("Try rephrasing your query or check if documents are indexed.")
                if numbers_in_query:
                    print(f"Note: Your query contained numbers {numbers_in_query}")
                    print("Make sure these numbers exist in your documents.")
                
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"Error during search: {e}")
            traceback.print_exc()

def main():
    """Main entry point for enhanced system - FIXED VERSION."""
    if len(sys.argv) < 2:
        print("\n=== Enhanced Qdrant Number-Aware RAG System (FIXED) ===")
        print("Usage:")
        print("  python qdrant_code rebuild    - Rebuild with enhanced number search")
        print("  python qdrant_code search     - Interactive enhanced search")
        print("  python qdrant_code rebuild <folder>  - Rebuild from custom folder")
        print("\nFixes Applied:")
        print("  ✅ Fixed deprecated API usage")
        print("  ✅ Proper SparseVector handling")
        print("  ✅ Better error handling and fallbacks")
        print("  ✅ Compatible with current Qdrant client")
        print("\nEnhancements:")
        print("  ✅ Number-aware tokenization")
        print("  ✅ Enhanced BM25 sparse vectors")  
        print("  ✅ True hybrid search with RRF")
        print("  ✅ Better handling of codes and identifiers")
        return
    
    command = sys.argv[1].lower()
    
    if command == "rebuild":
        uploads_folder = sys.argv[2] if len(sys.argv) > 2 else "uploads"
        success = rebuild_enhanced_vector_store(uploads_folder)
        if success:
            print("You can now run 'python qdrant_code search' to start searching.")
        
    elif command == "search":
        interactive_enhanced_search()
        
    else:
        print(f"Unknown command: {command}")
        print("Use 'rebuild' or 'search'")


if __name__ == "__main__":
    main()