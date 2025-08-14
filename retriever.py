#!/usr/bin/env python3
"""
Simplified Retriever

This retriever uses offline HuggingFace embeddings and optionally uses
Ollama for keyword extraction. It works with a single vector store.
"""

from typing import List, Dict, Any, Optional
from langchain.docstore.document import Document
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import os
import json
import requests
import traceback

class Retriever:
    def __init__(self, vector_store_path: str = "Vector_store"):
        """
        Initialize retriever with vector store.
        
        Args:
            vector_store_path: Path to the vector store directory
        """
        self.vector_store_path = vector_store_path
        print(f"Initializing retriever using vector store at '{self.vector_store_path}'")

        # Use offline HuggingFace embeddings
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.vector_store = None
        self.load_vector_store()
        
        # Ollama configuration
        self.ollama_base_url = "http://127.0.0.1:11434"
        self.ollama_model = "codeqwen:7b-chat-v1.5-q8_0"
        
    def load_vector_store(self):
        """Load the vector store if it exists."""
        try:
            if os.path.exists(self.vector_store_path) and os.path.isdir(self.vector_store_path):
                # Check if index files exist
                index_file = os.path.join(self.vector_store_path, "index.faiss")
                pkl_file = os.path.join(self.vector_store_path, "index.pkl")
                if not os.path.exists(index_file) or not os.path.exists(pkl_file):
                     print(f"Warning: index.faiss or index.pkl missing in {self.vector_store_path}")
                     self.vector_store = None
                     return
                     
                print(f"Loading existing vector store from {self.vector_store_path}...")
                self.vector_store = FAISS.load_local(
                    self.vector_store_path, 
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
                     
                print(f"Vector store loaded successfully with {self.vector_store.index.ntotal} vectors")
            else:
                print(f"No existing vector store directory found at {self.vector_store_path}")
                self.vector_store = None
        except Exception as e:
            print(f"Error loading vector store from {self.vector_store_path}: {e}")
            traceback.print_exc()
            self.vector_store = None

    def _get_keywords_from_ollama(self, query: str) -> List[str]:
        """Uses Ollama to extract keywords from the query."""
        try:
            prompt = (
                f"Extract key terms and phrases from this query that would be useful for document search. "
                f"Return only the keywords separated by commas, no other text.\n\n"
                f"Query: {query}\n"
                f"Keywords:"
            )
            
            payload = {
                "model": self.ollama_model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.1,
                    "num_predict": 50
                }
            }
            
            response = requests.post(
                f"{self.ollama_base_url}/api/generate",
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=300
            )
            
            if response.status_code == 200:
                result = response.json()
                keywords_text = result.get("response", "").strip()
                if keywords_text:
                    keywords = [k.strip() for k in keywords_text.split(',') if k.strip()]
                    print(f"Extracted keywords: {keywords}")
                    return keywords
            else:
                print(f"Ollama request failed with status {response.status_code}")
                
        except Exception as e:
            print(f"Error getting keywords from Ollama: {e}")
            
        return []

    def _keyword_scan(self, keywords: List[str], max_docs: int = 5) -> List[Document]:
        """Scan documents for keyword matches."""
        if not keywords or not self.vector_store:
            return []
            
        matched_docs = []
        try:
            # Get all documents from the vector store
            docstore = self.vector_store.docstore
            if hasattr(docstore, '_dict'):
                print(f"Scanning {len(docstore._dict)} documents for keywords: {keywords}")
                
                for doc_id, document in docstore._dict.items():
                    if len(matched_docs) >= max_docs:
                                    break
                            
                    if isinstance(document, Document):
                        content_lower = document.page_content.lower()
                        
                        # Check if any keyword is present
                        matched_keywords = []
                        for keyword in keywords:
                            if keyword.lower() in content_lower:
                                matched_keywords.append(keyword)
                        
                        if matched_keywords:
                            # Add metadata about the match
                            if document.metadata is None:
                                document.metadata = {}
                            document.metadata['matched_keywords'] = matched_keywords
                            document.metadata['match_type'] = 'keyword_scan'
                            matched_docs.append(document)
                            
                print(f"Found {len(matched_docs)} documents with keyword matches")
        except Exception as e:
            print(f"Error during keyword scan: {e}")
            
        return matched_docs

    def get_relevant_documents(self, user_query: str, k: int = 10, use_keywords: bool = True) -> List[Document]:
        """
        Get relevant documents using semantic search and optional keyword extraction.
        
        Args:
            user_query: The user's question
            k: Total number of documents to return
            use_keywords: Whether to use keyword extraction and scanning
            
        Returns:
            List of relevant Document objects
        """
        if not self.vector_store:
            print("Vector store not initialized or is empty")
            return []
            
        candidate_docs = []
        
        try:
            print(f"\n--- Starting search for query: '{user_query}' ---")
            
            # Step 1: Optional keyword extraction and scanning
            if use_keywords:
                keywords = self._get_keywords_from_ollama(user_query)
                if keywords:
                    candidate_docs.extend(self._keyword_scan(keywords, max_docs=5))
            
            # Step 2: Semantic search
            print(f"Performing semantic search (k=5)")
            semantic_docs = self.vector_store.similarity_search(user_query, k=5)
            
            # Add metadata to semantic docs before combining
            for doc in semantic_docs:
                if doc.metadata is None: doc.metadata = {}
                if 'match_type' not in doc.metadata:
                    doc.metadata['match_type'] = 'semantic'
            candidate_docs.extend(semantic_docs)
            
            # Step 3: Deduplicate and limit results, preserving order (keyword docs first)
            final_docs = []
            seen_content_hashes = set()
            for doc in candidate_docs:
                content_hash = hash(doc.page_content)
                if content_hash not in seen_content_hashes:
                    final_docs.append(doc)
                    seen_content_hashes.add(content_hash)
                    if len(final_docs) >= k:
                        break
            
            print(f"\n--- Found {len(final_docs)} unique documents ---")
            
            return final_docs
            
        except Exception as e:
            print(f"Error in get_relevant_documents: {e}")
            traceback.print_exc()
            return []

if __name__ == "__main__":
    retriever = Retriever()
    
    if not retriever.vector_store:
        print("\nCould not load vector store. Please run rebuild_vectors.py first.")
    else:
        while True:
            query = input("\nEnter your query (or 'exit' to quit): ")
            if query.lower() == 'exit':
                break
            
            documents = retriever.get_relevant_documents(query)
            
            if documents:
                print("\n--- Top Document Chunks ---")
                for i, doc in enumerate(documents):
                    print(f"\n--- Document {i+1} ---")
                    print(f"Source: {doc.metadata.get('source', 'N/A')}")
                    print(f"Match Type: {doc.metadata.get('match_type', 'N/A')}")
                    if 'matched_keywords' in doc.metadata:
                        print(f"Matched Keywords: {doc.metadata['matched_keywords']}")
                    print("\nContent:")
                    print(doc.page_content)
                    print("-" * 25)
            else:
                print("No relevant documents found.")
