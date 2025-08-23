#!/usr/bin/env python3
"""
Enhanced Qdrant Hybrid Search RAG with Advanced PDF Processing

Key Improvements:
1. Uses pdfplumber for better PDF extraction (tables, layouts)
2. Converts content to structured markdown format
3. Preserves table structures and formatting
4. Enhanced number-aware search with structured content
5. Better handling of financial/legal document structures
"""

import os
import sys
from typing import List, Dict, Any, Optional, Tuple
import pdfplumber  # Better PDF processing
import pandas as pd
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


class AdvancedPDFProcessor:
    """Advanced PDF processing with pdfplumber and markdown conversion."""
    
    def __init__(self):
        self.number_pattern = re.compile(r'\b\d+(?:\.\d+)?\b')
        self.currency_pattern = re.compile(r'[₹$€£¥]\s*\d+(?:,\d{3})*(?:\.\d{2})?|\d+(?:,\d{3})*(?:\.\d{2})?\s*[₹$€£¥]')
        self.percentage_pattern = re.compile(r'\d+(?:\.\d+)?%')
        
    def extract_tables_as_markdown(self, page) -> List[str]:
        """Extract tables from a page and convert to markdown format."""
        markdown_tables = []
        
        try:
            tables = page.extract_tables()
            
            for table_idx, table in enumerate(tables):
                if not table or len(table) < 2:  # Skip empty or single-row tables
                    continue
                    
                # Create DataFrame for better processing
                df = pd.DataFrame(table[1:], columns=table[0])
                
                # Clean the DataFrame
                df = df.dropna(how='all').fillna('')
                df = df.applymap(lambda x: str(x).strip() if x else '')
                
                # Filter out completely empty rows
                df = df[df.astype(str).apply(lambda row: row.str.strip().str.len().sum() > 0, axis=1)]
                
                if len(df) > 0:
                    # Convert to markdown
                    markdown_table = df.to_markdown(index=False)
                    
                    # Add table metadata
                    table_header = f"\n### Table {table_idx + 1}\n"
                    markdown_tables.append(table_header + markdown_table + "\n")
                    
        except Exception as e:
            print(f"Warning: Error extracting tables: {e}")
            
        return markdown_tables
    
    def extract_text_with_formatting(self, page) -> str:
        """Extract text while preserving some formatting cues."""
        try:
            # Get text with layout information
            text = page.extract_text()
            
            if not text:
                return ""
            
            # Try to identify headers (lines with fewer words, often in caps)
            lines = text.split('\n')
            formatted_lines = []
            
            for line in lines:
                line = line.strip()
                if not line:
                    formatted_lines.append("")
                    continue
                
                # Detect potential headers (short lines, mostly caps, no ending punctuation)
                words = line.split()
                if (len(words) <= 8 and 
                    line.isupper() and 
                    not line.endswith('.') and 
                    not any(c.isdigit() for c in line[:10])):  # Not starting with numbers
                    formatted_lines.append(f"## {line}")
                elif (len(words) <= 12 and 
                      sum(1 for c in line if c.isupper()) / len(line) > 0.6 and
                      not line.endswith('.') and
                      len(line) > 5):
                    formatted_lines.append(f"### {line}")
                else:
                    formatted_lines.append(line)
            
            return '\n'.join(formatted_lines)
            
        except Exception as e:
            print(f"Warning: Error in text formatting: {e}")
            return page.extract_text() or ""
    
    def process_pdf_to_markdown(self, filepath: str) -> List[Document]:
        """Process PDF and convert to structured markdown format."""
        documents = []
        
        try:
            print(f"Processing PDF with advanced extraction: {filepath}")
            
            with pdfplumber.open(filepath) as pdf:
                total_pages = len(pdf.pages)
                print(f"Found {total_pages} pages")
                
                for page_num, page in enumerate(pdf.pages):
                    print(f"Processing page {page_num + 1}/{total_pages}")
                    
                    # Extract formatted text
                    text_content = self.extract_text_with_formatting(page)
                    
                    # Extract tables as markdown
                    table_content = self.extract_tables_as_markdown(page)
                    
                    # Combine content
                    page_markdown = f"# Page {page_num + 1}\n\n"
                    
                    if text_content:
                        page_markdown += text_content + "\n\n"
                    
                    if table_content:
                        page_markdown += "## Tables\n\n"
                        page_markdown += "\n".join(table_content)
                    
                    # Only add if there's meaningful content
                    if len(page_markdown.strip()) > 20:
                        metadata = {
                            "source": os.path.basename(filepath),
                            "page": page_num + 1,
                            "format": "markdown",
                            "extraction_method": "pdfplumber"
                        }
                        
                        # Add content type indicators
                        has_tables = len(table_content) > 0
                        has_numbers = bool(self.number_pattern.search(page_markdown))
                        has_currency = bool(self.currency_pattern.search(page_markdown))
                        has_percentages = bool(self.percentage_pattern.search(page_markdown))
                        
                        metadata.update({
                            "has_tables": has_tables,
                            "has_numbers": has_numbers,
                            "has_currency": has_currency,
                            "has_percentages": has_percentages,
                            "table_count": len(table_content)
                        })
                        
                        documents.append(Document(
                            page_content=page_markdown,
                            metadata=metadata
                        ))
                    
        except Exception as e:
            print(f"Error processing PDF {filepath}: {e}")
            traceback.print_exc()
            
        print(f"Extracted {len(documents)} pages with structured content")
        return documents


class EnhancedHybridRetriever:
    """Enhanced hybrid retriever with advanced PDF processing and markdown support."""
    
    def __init__(
        self, 
        collection_name: str = "enhanced_documents",
        qdrant_path: str = "./qdrant_storage",
        embedding_model: str = "all-MiniLM-L6-v2"
    ):
        self.collection_name = collection_name
        self.qdrant_path = qdrant_path
        
        # Initialize Qdrant client
        self.client = QdrantClient(path=qdrant_path)
        
        # Initialize embeddings
        print(f"Loading embedding model: {embedding_model}")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        # Initialize PDF processor
        self.pdf_processor = AdvancedPDFProcessor()
        
        # Enhanced vocabulary for structured content
        self.vocabulary = {}
        self.idf_scores = {}
        self.total_docs = 0
        
        # Enhanced patterns for structured content
        self.number_pattern = re.compile(r'\b\d+(?:\.\d+)?\b')
        self.currency_pattern = re.compile(r'[₹$€£¥]\s*\d+(?:,\d{3})*(?:\.\d{2})?|\d+(?:,\d{3})*(?:\.\d{2})?\s*[₹$€£¥]')
        self.percentage_pattern = re.compile(r'\d+(?:\.\d+)?%')
        self.header_pattern = re.compile(r'^#{1,6}\s+(.+)$', re.MULTILINE)
        self.table_pattern = re.compile(r'\|.*\|', re.MULTILINE)
        
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
        """Create a new Qdrant collection."""
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

    def _enhanced_tokenize_markdown(self, text: str) -> List[str]:
        """Enhanced tokenization for markdown content with structure awareness."""
        text_lower = text.lower()
        tokens = []
        
        # Extract markdown headers and give them extra weight
        headers = self.header_pattern.findall(text)
        for header in headers:
            header_tokens = re.findall(r'\b\w+\b', header.lower())
            for token in header_tokens:
                tokens.extend([f"HEADER_{token}", f"IMPORTANT_{token}"])
        
        # Extract table content and mark it specially
        table_lines = self.table_pattern.findall(text)
        for table_line in table_lines:
            table_tokens = re.findall(r'\b\w+\b', table_line.lower())
            for token in table_tokens:
                tokens.append(f"TABLE_{token}")
        
        # Regular word tokenization
        words = re.findall(r'\b[a-zA-Z]+\b', text_lower)
        tokens.extend(words)
        
        # Enhanced number extraction
        numbers = self.number_pattern.findall(text)
        for num in numbers:
            tokens.extend([f"NUM_{num}", f"EXACT_{num}"])
        
        # Currency amounts
        currencies = self.currency_pattern.findall(text)
        for curr in currencies:
            clean_curr = re.sub(r'[^\d.]', '', curr)
            if clean_curr:
                tokens.extend([f"CURRENCY_{clean_curr}", f"AMOUNT_{clean_curr}"])
        
        # Percentages
        percentages = self.percentage_pattern.findall(text)
        for perc in percentages:
            clean_perc = perc.replace('%', '')
            tokens.extend([f"PERCENT_{clean_perc}", f"RATE_{clean_perc}"])
        
        # Alphanumeric codes (sections, rules, etc.)
        codes = re.findall(r'\b[a-zA-Z]*\d+[a-zA-Z]*\b', text)
        for code in codes:
            if code not in numbers:
                tokens.extend([f"CODE_{code.lower()}", f"SECTION_{code.lower()}"])
        
        # Context bigrams with enhanced weighting
        words_and_numbers = re.findall(r'\b\w+\b', text_lower)
        for i in range(len(words_and_numbers) - 1):
            bigram = f"{words_and_numbers[i]}_{words_and_numbers[i+1]}"
            tokens.append(f"CONTEXT_{bigram}")
        
        return tokens

    def _build_enhanced_vocabulary(self, documents: List[Document]):
        """Build vocabulary optimized for structured markdown content."""
        print("Building enhanced vocabulary for structured content...")
        
        word_doc_count = Counter()
        all_words = set()
        
        for doc in documents:
            tokens = set(self._enhanced_tokenize_markdown(doc.page_content))
            for token in tokens:
                word_doc_count[token] += 1
                all_words.add(token)
        
        # Create vocabulary mapping
        self.vocabulary = {word: idx for idx, word in enumerate(sorted(all_words))}
        self.total_docs = len(documents)
        
        # Enhanced IDF calculation with structure-aware scoring
        self.idf_scores = {}
        for word, doc_freq in word_doc_count.items():
            base_idf = math.log((self.total_docs + 1) / (doc_freq + 1)) + 1
            
            # Boost important token types
            if word.startswith(('HEADER_', 'IMPORTANT_')):
                base_idf *= 1.8
            elif word.startswith(('TABLE_', 'CURRENCY_', 'AMOUNT_')):
                base_idf *= 1.6
            elif word.startswith(('NUM_', 'EXACT_', 'PERCENT_', 'RATE_')):
                base_idf *= 1.5
            elif word.startswith(('CODE_', 'SECTION_')):
                base_idf *= 1.4
            elif word.startswith('CONTEXT_'):
                base_idf *= 1.2
            
            self.idf_scores[word] = base_idf
        
        print(f"Built enhanced vocabulary with {len(self.vocabulary)} unique terms")
        
        # Statistics
        header_tokens = len([w for w in self.vocabulary if w.startswith('HEADER_')])
        table_tokens = len([w for w in self.vocabulary if w.startswith('TABLE_')])
        number_tokens = len([w for w in self.vocabulary if w.startswith('NUM_')])
        currency_tokens = len([w for w in self.vocabulary if w.startswith('CURRENCY_')])
        
        print(f"  - Header tokens: {header_tokens}")
        print(f"  - Table tokens: {table_tokens}")
        print(f"  - Number tokens: {number_tokens}")
        print(f"  - Currency tokens: {currency_tokens}")
        
        self._save_vocabulary()

    def _save_vocabulary(self):
        """Save vocabulary to disk."""
        vocab_path = os.path.join(self.qdrant_path, "enhanced_markdown_vocabulary.json")
        vocab_data = {
            "vocabulary": self.vocabulary,
            "idf_scores": self.idf_scores,
            "total_docs": self.total_docs
        }
        with open(vocab_path, 'w') as f:
            json.dump(vocab_data, f)
        print(f"Saved vocabulary to {vocab_path}")

    def _load_vocabulary(self):
        """Load vocabulary from disk."""
        vocab_path = os.path.join(self.qdrant_path, "enhanced_markdown_vocabulary.json")
        if os.path.exists(vocab_path):
            with open(vocab_path, 'r') as f:
                vocab_data = json.load(f)
            self.vocabulary = vocab_data.get("vocabulary", {})
            self.idf_scores = vocab_data.get("idf_scores", {})
            self.total_docs = vocab_data.get("total_docs", 0)
            print(f"Loaded vocabulary with {len(self.vocabulary)} terms")
        else:
            print("No vocabulary file found")

    def _create_structured_sparse_vector(self, text: str) -> SparseVector:
        """Create sparse vector optimized for structured content."""
        tokens = self._enhanced_tokenize_markdown(text)
        token_counts = Counter(tokens)
        
        # Enhanced BM25 parameters for structured content
        k1 = 1.5  # Higher term frequency saturation for structured content
        b = 0.7   # Balanced length normalization
        
        avg_doc_length = 300  # Adjusted for markdown content
        doc_length = len(tokens)
        
        indices = []
        values = []
        
        for token, tf in token_counts.items():
            if token in self.vocabulary:
                idx = self.vocabulary[token]
                idf = self.idf_scores.get(token, 0)
                
                # Enhanced BM25 score
                numerator = tf * (k1 + 1)
                denominator = tf + k1 * (1 - b + b * (doc_length / avg_doc_length))
                bm25_score = idf * (numerator / denominator)
                
                if bm25_score > 0:
                    indices.append(idx)
                    values.append(bm25_score)
        
        return SparseVector(indices=indices, values=values)

    def process_document(self, filepath: str) -> List[Document]:
        """Process document with enhanced PDF handling."""
        try:
            print(f"Processing document: {filepath}")
            
            if not os.path.exists(filepath):
                print(f"ERROR: File does not exist: {filepath}")
                return []
                
            file_size = os.path.getsize(filepath)
            if file_size == 0:
                print(f"ERROR: File is empty: {filepath}")
                return []
            
            docs = []
            
            # Use enhanced PDF processing
            if filepath.endswith('.pdf'):
                docs = self.pdf_processor.process_pdf_to_markdown(filepath)
            elif filepath.endswith('.txt'):
                loader = TextLoader(filepath, encoding='utf-8', autodetect_encoding=True)
                raw_docs = loader.load()
                # Convert plain text to basic markdown
                for doc in raw_docs:
                    doc.page_content = f"# {os.path.basename(filepath)}\n\n{doc.page_content}"
                    doc.metadata["format"] = "markdown"
                    doc.metadata["extraction_method"] = "text_loader"
                docs = raw_docs
            else:
                print(f"Unsupported file type: {filepath}")
                return []

            # Enhanced chunking for markdown content
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=2000,  # Larger chunks for structured content
                chunk_overlap=400,  # More overlap to preserve structure
                length_function=len,
                separators=["\n## ", "\n### ", "\n\n", "\n", ". ", " ", ""]
            )
            
            chunks = text_splitter.split_documents(docs)
            
            # Enhanced chunk processing
            valid_chunks = []
            for chunk in chunks:
                content = chunk.page_content.strip()
                if content and len(content) > 20:
                    # Extract structured elements for metadata
                    has_tables = bool(self.table_pattern.search(content))
                    has_headers = bool(self.header_pattern.search(content))
                    numbers = self.number_pattern.findall(content)
                    currencies = self.currency_pattern.findall(content)
                    percentages = self.percentage_pattern.findall(content)
                    
                    # Enhanced metadata
                    chunk.metadata.update({
                        "chunk_id": len(valid_chunks),
                        "has_tables": has_tables,
                        "has_headers": has_headers,
                        "number_count": len(numbers),
                        "currency_count": len(currencies),
                        "percentage_count": len(percentages),
                        "content_type": "structured" if (has_tables or has_headers) else "text"
                    })
                    
                    valid_chunks.append(chunk)
            
            print(f"Created {len(valid_chunks)} structured chunks from {filepath}")
            return valid_chunks
            
        except Exception as e:
            print(f"Error processing {filepath}: {e}")
            traceback.print_exc()
            return []

    def add_documents(self, documents: List[Document], batch_size: int = 30):
        """Add documents with enhanced structured processing."""
        if not documents:
            print("No documents to add")
            return
        
        print(f"Adding {len(documents)} documents with structured processing...")
        
        # Build vocabulary if not exists
        if not self.vocabulary:
            self._build_enhanced_vocabulary(documents)
        
        # Process in batches
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            points = []
            
            print(f"Processing batch {i//batch_size + 1}/{(len(documents) + batch_size - 1)//batch_size}")
            
            for doc_id, doc in enumerate(batch):
                try:
                    # Create vectors
                    dense_vector = self.embeddings.embed_query(doc.page_content)
                    sparse_vector = self._create_structured_sparse_vector(doc.page_content)
                    
                    # Enhanced payload with structured data
                    numbers = self.number_pattern.findall(doc.page_content)
                    currencies = self.currency_pattern.findall(doc.page_content)
                    percentages = self.percentage_pattern.findall(doc.page_content)
                    headers = [h.strip() for h in self.header_pattern.findall(doc.page_content)]
                    
                    payload = {
                        "content": doc.page_content,
                        "metadata": doc.metadata or {},
                        "source": doc.metadata.get("source", "") if doc.metadata else "",
                        "page": doc.metadata.get("page", 0) if doc.metadata else 0,
                        "numbers": numbers[:50],  # Limit to first 50
                        "currencies": currencies[:20],
                        "percentages": percentages[:20],
                        "headers": headers[:10],
                        "format": doc.metadata.get("format", "text") if doc.metadata else "text",
                        "has_tables": doc.metadata.get("has_tables", False) if doc.metadata else False,
                        "content_type": doc.metadata.get("content_type", "text") if doc.metadata else "text"
                    }
                    
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
                print(f"Uploaded {len(points)} points with structured data")

    def enhanced_hybrid_search(
        self, 
        query: str, 
        limit: int = 10,
        dense_weight: float = 0.1,
        sparse_weight: float = 0.9
    ) -> List[Dict[str, Any]]:
        """Enhanced hybrid search for structured content."""
        try:
            print(f"Performing structured hybrid search for: '{query}'")
            
            # Create query vectors
            dense_query = self.embeddings.embed_query(query)
            sparse_query = self._create_structured_sparse_vector(query)
            
            # Analyze query for structured elements
            query_numbers = self.number_pattern.findall(query)
            query_currencies = self.currency_pattern.findall(query)
            query_percentages = self.percentage_pattern.findall(query)
            
            has_structured_content = bool(query_numbers or query_currencies or query_percentages)
            
            if has_structured_content:
                print(f"🔢 Structured content detected:")
                if query_numbers: print(f"   Numbers: {query_numbers}")
                if query_currencies: print(f"   Currencies: {query_currencies}")
                if query_percentages: print(f"   Percentages: {query_percentages}")
            
            # Perform searches
            dense_results = []
            sparse_results = []
            
            # Dense search
            try:
                dense_response = self.client.query_points(
                    collection_name=self.collection_name,
                    query=dense_query,
                    using="dense",
                    limit=limit * 2,
                    with_payload=True
                )
                dense_results = dense_response.points if hasattr(dense_response, 'points') else dense_response
            except Exception as e:
                print(f"Dense search failed, trying fallback: {e}")
                try:
                    dense_results = self.client.search(
                        collection_name=self.collection_name,
                        query_vector=dense_query,
                        limit=limit * 2,
                        with_payload=True
                    )
                except Exception as e2:
                    print(f"All dense search methods failed: {e2}")
            
            # Sparse search
            if sparse_query.indices and sparse_query.values:
                try:
                    sparse_response = self.client.query_points(
                        collection_name=self.collection_name,
                        query=sparse_query,
                        using="sparse",
                        limit=limit * 2,
                        with_payload=True
                    )
                    sparse_results = sparse_response.points if hasattr(sparse_response, 'points') else sparse_response
                except Exception as e:
                    print(f"Sparse search failed, trying fallback: {e}")
                    try:
                        sparse_results = self.client.search(
                            collection_name=self.collection_name,
                            query_vector=NamedVector(name="sparse", vector=sparse_query),
                            limit=limit * 2,
                            with_payload=True
                        )
                    except Exception as e2:
                        print(f"All sparse search methods failed: {e2}")
            
            # Enhanced RRF with structured content boosting
            rrf_scores = {}
            k = 60
            
            # Score dense results
            for rank, result in enumerate(dense_results):
                point_id = result.id
                rrf_score = dense_weight / (k + rank + 1)
                
                # Boost for structured content matches
                if has_structured_content and hasattr(result, 'payload') and result.payload:
                    doc_numbers = result.payload.get('numbers', [])
                    doc_currencies = result.payload.get('currencies', [])
                    doc_percentages = result.payload.get('percentages', [])
                    
                    # Exact number matches
                    if query_numbers and any(num in doc_numbers for num in query_numbers):
                        rrf_score *= 2.5
                    
                    # Currency matches
                    if query_currencies and any(curr in doc_currencies for curr in query_currencies):
                        rrf_score *= 2.0
                    
                    # Percentage matches
                    if query_percentages and any(perc in doc_percentages for perc in query_percentages):
                        rrf_score *= 2.0
                    
                    # Table content bonus
                    if result.payload.get('has_tables', False):
                        rrf_score *= 1.3
                
                rrf_scores[point_id] = rrf_scores.get(point_id, 0) + rrf_score
            
            # Score sparse results with additional structured boosting
            for rank, result in enumerate(sparse_results):
                point_id = result.id
                rrf_score = sparse_weight / (k + rank + 1)
                
                # Similar boosting logic for sparse results
                if has_structured_content and hasattr(result, 'payload') and result.payload:
                    doc_numbers = result.payload.get('numbers', [])
                    doc_currencies = result.payload.get('currencies', [])
                    doc_percentages = result.payload.get('percentages', [])
                    
                    if query_numbers and any(num in doc_numbers for num in query_numbers):
                        rrf_score *= 2.0
                    if query_currencies and any(curr in doc_currencies for curr in query_currencies):
                        rrf_score *= 1.8
                    if query_percentages and any(perc in doc_percentages for perc in query_percentages):
                        rrf_score *= 1.8
                
                rrf_scores[point_id] = rrf_scores.get(point_id, 0) + rrf_score
            
            # Compile results
            all_results_map = {}
            for result in dense_results + sparse_results:
                all_results_map[result.id] = result
            
            # Sort and format
            sorted_results = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:limit]
            
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
                        "currencies": payload.get("currencies", []),
                        "percentages": payload.get("percentages", []),
                        "headers": payload.get("headers", []),
                        "has_tables": payload.get("has_tables", False),
                        "content_type": payload.get("content_type", "text"),
                        "search_type": "enhanced_structured_hybrid"
                    })
            
            print(f"Enhanced structured search returned {len(final_results)} results")
            return final_results
            
        except Exception as e:
            print(f"Error in enhanced hybrid search: {e}")
            traceback.print_exc()
            return []

    def get_relevant_documents(self, query: str, k: int = 10) -> List[Document]:
        """Get relevant documents using enhanced structured search."""
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
                    "numbers": result.get("numbers", []),
                    "currencies": result.get("currencies", []),
                    "percentages": result.get("percentages", []),
                    "headers": result.get("headers", []),
                    "has_tables": result.get("has_tables", False),
                    "content_type": result.get("content_type", "text")
                }
            )
            documents.append(doc)
        
        return documents

    def rebuild_from_folder(self, uploads_folder: str = "uploads"):
        """Rebuild collection with enhanced processing."""
        print(f"Rebuilding with enhanced PDF processing from {uploads_folder}")
        
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
            print(f"Successfully rebuilt collection with {len(all_docs)} structured documents")
        else:
            print("No documents found to add")


def rebuild_enhanced_vector_store(uploads_folder: str = "uploads"):
    """Rebuild vector store with enhanced PDF processing and markdown support."""
    print("=== Enhanced PDF Processing with pdfplumber & Markdown ===")
    print(f"Rebuilding from '{uploads_folder}' with advanced features:")
    print("✓ pdfplumber for better table extraction")
    print("✓ Structured markdown conversion")
    print("✓ Table-aware tokenization")
    print("✓ Enhanced number/currency detection")
    print("✓ Header and structure preservation")
    
    proceed = input("\nDo you want to proceed with enhanced processing? (y/n): ").lower() == 'y'
    if not proceed:
        print("Operation cancelled.")
        return False
    
    # Check for required packages
    try:
        import pdfplumber
        import pandas as pd
    except ImportError as e:
        print(f"ERROR: Missing required package. Please install:")
        print("pip install pdfplumber pandas")
        print(f"ImportError: {e}")
        return False
    
    # Initialize enhanced retriever
    retriever = EnhancedHybridRetriever()
    
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
    
    print(f"\nFound {len(supported_files)} documents to process:")
    for filepath in supported_files:
        file_size = os.path.getsize(filepath) / (1024 * 1024)
        print(f"  - {os.path.basename(filepath)} ({file_size:.2f} MB)")
    
    # Rebuild with enhanced processing
    retriever.rebuild_from_folder(uploads_folder)
    
    print("\n✅ Enhanced processing completed!")
    print("New capabilities:")
    print("  🔸 Table content extraction and search")
    print("  🔸 Structured markdown format")
    print("  🔸 Header-aware chunking")
    print("  🔸 Currency and percentage detection")
    print("  🔸 Enhanced number matching")
    print("  🔸 Document structure preservation")
    
    return True


def interactive_enhanced_search():
    """Interactive search with enhanced structured content support."""
    print("=== Enhanced Structured Content Search Interface ===")
    print("Optimized for tables, numbers, currencies, and structured data!")
    
    try:
        retriever = EnhancedHybridRetriever()
        
        # Check collection
        try:
            collection_info = retriever.client.get_collection(retriever.collection_name)
            if collection_info.points_count == 0:
                print("Collection is empty. Please run rebuild first.")
                return
            print(f"Loaded collection with {collection_info.points_count} structured documents")
        except Exception as e:
            print(f"Collection not found. Please run rebuild first. Error: {e}")
            return
            
    except Exception as e:
        print(f"Error initializing retriever: {e}")
        return
    
    print("\nReady for enhanced search! Type 'exit' to quit, 'help' for commands.")
    print("Try queries like:")
    print("  📊 'table with tax rates'")
    print("  💰 'GST 18% calculation'") 
    print("  📋 'section 12 requirements'")
    print("  🔢 'total amount ₹50000'")
    
    while True:
        try:
            query = input("\n🔍 Enter your query: ").strip()
            
            if query.lower() in ['exit', 'quit', 'q']:
                print("Goodbye!")
                break
            elif query.lower() == 'help':
                print("\nAvailable commands:")
                print("  - Any text: Search structured documents")
                print("  - 'help': Show this help")
                print("  - 'exit'/'quit'/'q': Exit")
                print("  - 'stats': Show collection statistics")
                print("  - 'examples': Show search examples")
                print("\nSearch features:")
                print("  📊 Table content: 'tax table', 'rate schedule'")
                print("  💰 Currency amounts: '₹1000', '$500', 'amount 25000'")
                print("  📈 Percentages: '18% GST', 'rate 12%'")
                print("  🔢 Exact numbers: 'section 80C', 'rule 114'")
                print("  📝 Headers: 'chapter 5', 'provisions of'")
                continue
            elif query.lower() == 'examples':
                print("\n📚 Enhanced Search Examples:")
                print("Financial Documents:")
                print("  • 'GST rate table for services'")
                print("  • 'section 80C deduction limit ₹150000'")
                print("  • 'TDS rate 10% on salary'")
                print("Legal Documents:")
                print("  • 'rule 88A provisions'")
                print("  • 'penalty under section 271'")
                print("  • 'compliance requirements table'")
                print("Structured Content:")
                print("  • 'tax calculation example'")
                print("  • 'due dates schedule'")
                print("  • 'exemption limits 2024'")
                continue
            elif query.lower() == 'stats':
                collection_info = retriever.client.get_collection(retriever.collection_name)
                print(f"\n📊 Enhanced Collection Statistics:")
                print(f"  Total documents: {collection_info.points_count}")
                print(f"  Vocabulary size: {len(retriever.vocabulary)}")
                
                # Enhanced statistics
                header_tokens = len([w for w in retriever.vocabulary.keys() if w.startswith('HEADER_')])
                table_tokens = len([w for w in retriever.vocabulary.keys() if w.startswith('TABLE_')])
                number_tokens = len([w for w in retriever.vocabulary.keys() if w.startswith('NUM_')])
                currency_tokens = len([w for w in retriever.vocabulary.keys() if w.startswith('CURRENCY_')])
                percent_tokens = len([w for w in retriever.vocabulary.keys() if w.startswith('PERCENT_')])
                
                print(f"  📝 Header tokens: {header_tokens}")
                print(f"  📊 Table tokens: {table_tokens}")
                print(f"  🔢 Number tokens: {number_tokens}")
                print(f"  💰 Currency tokens: {currency_tokens}")
                print(f"  📈 Percentage tokens: {percent_tokens}")
                print(f"  📁 Storage: {retriever.qdrant_path}")
                continue
            elif not query:
                print("Please enter a search query.")
                continue
            
            print(f"\nSearching with enhanced structured processing...")
            
            # Analyze query structure
            numbers = retriever.number_pattern.findall(query)
            currencies = retriever.currency_pattern.findall(query)
            percentages = retriever.percentage_pattern.findall(query)
            
            if numbers or currencies or percentages:
                print("🎯 Structured content detected:")
                if numbers: print(f"   🔢 Numbers: {numbers}")
                if currencies: print(f"   💰 Currencies: {currencies}")
                if percentages: print(f"   📈 Percentages: {percentages}")
            
            # Perform search
            documents = retriever.get_relevant_documents(query, k=5)
            
            if documents:
                print(f"\n📋 Found {len(documents)} relevant documents:\n")
                
                for i, doc in enumerate(documents, 1):
                    print(f"📄 Result {i}")
                    print(f"   Source: {doc.metadata.get('source', 'Unknown')}")
                    if 'page' in doc.metadata and doc.metadata['page']:
                        print(f"   Page: {doc.metadata['page']}")
                    print(f"   Score: {doc.metadata.get('score', 0):.4f}")
                    print(f"   Type: {doc.metadata.get('content_type', 'text')}")
                    
                    # Show structured content indicators
                    if doc.metadata.get('has_tables', False):
                        print("   📊 Contains tables")
                    
                    doc_numbers = doc.metadata.get('numbers', [])
                    if doc_numbers:
                        print(f"   🔢 Numbers: {doc_numbers[:5]}{'...' if len(doc_numbers) > 5 else ''}")
                    
                    doc_currencies = doc.metadata.get('currencies', [])
                    if doc_currencies:
                        print(f"   💰 Currencies: {doc_currencies[:3]}{'...' if len(doc_currencies) > 3 else ''}")
                    
                    doc_percentages = doc.metadata.get('percentages', [])
                    if doc_percentages:
                        print(f"   📈 Percentages: {doc_percentages[:3]}{'...' if len(doc_percentages) > 3 else ''}")
                    
                    doc_headers = doc.metadata.get('headers', [])
                    if doc_headers:
                        print(f"   📝 Headers: {doc_headers[:2]}{'...' if len(doc_headers) > 2 else ''}")
                    
                    # Show content preview with basic highlighting
                    content = doc.page_content.strip()
                    
                    # Simple highlighting for structured elements
                    for num in numbers:
                        content = re.sub(f'\\b{re.escape(num)}\\b', f"**{num}**", content, flags=re.IGNORECASE)
                    
                    for curr in currencies:
                        # Escape special regex characters in currency
                        escaped_curr = re.escape(curr)
                        content = re.sub(escaped_curr, f"**{curr}**", content, flags=re.IGNORECASE)
                    
                    for perc in percentages:
                        content = re.sub(f'\\b{re.escape(perc)}\\b', f"**{perc}**", content, flags=re.IGNORECASE)
                    
                    # Limit content length for display
                    if len(content) > 800:
                        content = content[:800] + "..."
                    
                    print(f"   Content: {content}")
                    print("   " + "─" * 80)
                
            else:
                print("❌ No relevant documents found.")
                print("💡 Search tips:")
                print("  - Try broader terms: 'tax' instead of 'specific tax code'")
                print("  - Include context: 'GST calculation' instead of just 'GST'")
                print("  - Check spelling and try synonyms")
                if numbers or currencies or percentages:
                    print("  - Your query contains structured data - ensure it exists in documents")
                
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"Error during search: {e}")
            traceback.print_exc()


def main():
    """Main entry point for enhanced PDF processing system."""
    if len(sys.argv) < 2:
        print("\n=== Enhanced PDF Processing RAG System ===")
        print("🚀 Advanced Features:")
        print("  ✓ pdfplumber for superior table extraction")
        print("  ✓ Structured markdown conversion") 
        print("  ✓ Table-aware search and indexing")
        print("  ✓ Enhanced currency/percentage detection")
        print("  ✓ Header-based document structure")
        print("  ✓ Number-aware hybrid search")
        print("\nUsage:")
        print("  python enhanced_qdrant_processor.py rebuild    - Rebuild with enhanced processing")
        print("  python enhanced_qdrant_processor.py search     - Interactive structured search")
        print("  python enhanced_qdrant_processor.py rebuild <folder>  - Rebuild from custom folder")
        print("\n📋 Requirements:")
        print("  pip install pdfplumber pandas qdrant-client langchain-community")
        print("  pip install sentence-transformers PyMuPDF")
        return
    
    command = sys.argv[1].lower()
    
    if command == "rebuild":
        uploads_folder = sys.argv[2] if len(sys.argv) > 2 else "uploads"
        success = rebuild_enhanced_vector_store(uploads_folder)
        if success:
            print("\n🎉 Ready to search! Run:")
            print("python enhanced_qdrant_processor.py search")
        
    elif command == "search":
        interactive_enhanced_search()
        
    else:
        print(f"Unknown command: {command}")
        print("Use 'rebuild' or 'search'")


if __name__ == "__main__":
    main()