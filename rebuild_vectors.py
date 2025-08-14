#!/usr/bin/env python3
"""
Vector Store Rebuilder

This script rebuilds the vector store from documents in the uploads folder.
It processes the documents, chunks them, creates embeddings, and saves
the indexed data into the vector store directory.
"""

import os
import sys
from typing import List
import fitz  # PyMuPDF
from langchain.docstore.document import Document
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import re

# Define the folder paths
UPLOADS_FOLDER = 'uploads'
VECTOR_STORE_FOLDER = 'Vector_store'

def allowed_file(filename):
    """Check if file type is allowed."""
    ALLOWED_EXTENSIONS = {'pdf', 'txt', 'doc', 'docx'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def process_document(filepath: str) -> List[Document]:
    """Process a document and return chunks."""
    try:
        print(f"Processing document: {filepath}")
        
        # Check if file exists and is readable
        if not os.path.exists(filepath):
            print(f"ERROR: File does not exist: {filepath}")
            return []
            
        if not os.access(filepath, os.R_OK):
            print(f"ERROR: File is not readable: {filepath}")
            return []
            
        # Get file size and check if it's not empty
        file_size = os.path.getsize(filepath)
        if file_size == 0:
            print(f"ERROR: File is empty: {filepath}")
            return []
            
        print(f"File info: {filepath}, size: {file_size} bytes")
            
        # Choose loader based on file extension
        if filepath.endswith('.pdf'):
            try:
                docs = []
                print(f"Using PyMuPDF (fitz) for {filepath}")
                with fitz.open(filepath) as pdf_doc:
                    print(f"PDF has {len(pdf_doc)} pages.")
                    for page_num, page in enumerate(pdf_doc):
                        text = page.get_text("text") # Extract plain text
                        if text: # Only add if text is extracted
                            metadata = {"source": os.path.basename(filepath), "page": page_num + 1}
                            docs.append(Document(page_content=text, metadata=metadata))
                        else:
                            print(f"Warning: No text found on page {page_num + 1} of {filepath}")
                if not docs:
                    print(f"Warning: No text extracted from PDF {filepath} using PyMuPDF.")
                    return [] # Return empty list if no text could be extracted
                print(f"Extracted text from {len(docs)} PDF pages.")
            except Exception as e:
                print(f"Error processing PDF with PyMuPDF: {e}")
                # Consider adding alternative methods or logging details
                import traceback
                traceback.print_exc()
                return [] # Return empty on error
        elif filepath.endswith('.txt'):
            try:
                # Use explicit encoding with error handling
                loader = TextLoader(filepath, encoding='utf-8', autodetect_encoding=True)
                print(f"Using TextLoader for {filepath} with encoding detection")
            except Exception as e:
                print(f"Error with TextLoader: {e}, trying with different encoding")
                try:
                    # Fallback to Latin-1 which rarely fails
                    loader = TextLoader(filepath, encoding='latin-1')
                    print(f"Using TextLoader with latin-1 encoding")
                except Exception as e2:
                    print(f"Error with fallback loader: {e2}")
                    raise
        else:
            error_msg = f"Unsupported file type: {filepath}"
            print(f"ERROR: {error_msg}")
            raise ValueError(error_msg)
        
        # Load the document content (handled above for PDF, loader.load() only for TXT now)
        if not filepath.endswith('.pdf'):
             print(f"Loading document content...")
             docs = loader.load()
             print(f"Successfully loaded {len(docs)} document pages/sections")
        
        # Clean loaded document texts to remove non-ASCII and normalize whitespace
        # Note: PyMuPDF extraction might be cleaner already, but applying this consistently
        for doc in docs:
            content = doc.page_content
            # normalize whitespace
            content = re.sub(r'\s+', ' ', content)
            doc.page_content = content.strip()
        
        if not docs:
            print(f"WARNING: No content extracted from {filepath}")
            return []
            
        # Set chunk parameters
        chunk_size = 1500  # Increased chunk size for better context
        chunk_overlap = 500  # Significantly increased overlap for better context preservation
        
        print(f"Using chunk strategy: recursive, overlap: {chunk_overlap}")
        
        # Use recursive character text splitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        chunks = text_splitter.split_documents(docs)
            
        print(f"Document {filepath} processed into {len(chunks)} chunks")
        
        # Verify chunks have content
        valid_chunks = []
        for i, chunk in enumerate(chunks):
            if not chunk.page_content or len(chunk.page_content.strip()) < 10:
                print(f"WARNING: Chunk {i} has insufficient content, skipping")
                continue
                
            # Add metadata to track embedding status and source
            if not hasattr(chunk, 'metadata') or chunk.metadata is None:
                chunk.metadata = {}
                
            # Ensure source is set
            if 'source' not in chunk.metadata:
                chunk.metadata['source'] = os.path.basename(filepath)
                
            chunk.metadata['embedding_success'] = True
            valid_chunks.append(chunk)
            
        print(f"Final valid chunks: {len(valid_chunks)} out of {len(chunks)}")
        
        if not valid_chunks:
            print(f"WARNING: No valid chunks created from {filepath}")
            
        return valid_chunks
        
    except Exception as e:
        print(f"Error processing document {filepath}: {e}")
        import traceback
        traceback.print_exc()
        return []

def rebuild_vector_store():
    """
    Rebuild vector store from documents in the uploads folder.
    """
    try:
        all_docs = []
        file_count = 0
        processed_count = 0
        error_count = 0

        print(f"Looking for documents in: {UPLOADS_FOLDER}")
        
        # Check if the uploads directory exists and has files
        if not os.path.exists(UPLOADS_FOLDER) or not os.listdir(UPLOADS_FOLDER):
            print(f"No documents found in {UPLOADS_FOLDER}")
            return False
            
        # Clear the existing vector store
        if os.path.exists(VECTOR_STORE_FOLDER):
            for file_name in os.listdir(VECTOR_STORE_FOLDER):
                file_path = os.path.join(VECTOR_STORE_FOLDER, file_name)
                if os.path.isfile(file_path):
                    try:
                        os.remove(file_path)
                        print(f"Removed old vector store file: {file_path}")
                    except Exception as e:
                        print(f"Error removing file {file_path}: {e}")
        
        # Process documents in uploads folder
        for filename in os.listdir(UPLOADS_FOLDER):
            filepath = os.path.join(UPLOADS_FOLDER, filename)
            
            # Skip directories and non-document files
            if os.path.isdir(filepath) or not allowed_file(filename):
                continue
                
            file_count += 1
            print(f"Processing file {file_count}: {filename}")
            
            try:
                # Process the document
                docs = process_document(filepath)
                
                if docs:
                    all_docs.extend(docs)
                    processed_count += 1
                    print(f"Successfully processed {filename}, added {len(docs)} chunks")
                else:
                    error_count += 1
                    print(f"No content extracted from {filename}")
            except Exception as e:
                error_count += 1
                print(f"Error processing {filename}: {str(e)}")
                import traceback
                traceback.print_exc()
        
        # Create vector store if documents were found
        if all_docs:
            print(f"Creating vector store with {len(all_docs)} total chunks from {processed_count} documents")
            
            # Initialize HuggingFace embeddings for offline use
            embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            
            # Process in batches to respect API limits
            batch_size = 80  # Below Cohere's limit of 96 texts per request
            
            texts = [doc.page_content for doc in all_docs]
            metadatas = [doc.metadata for doc in all_docs]
                        
            # Create new vector store
            new_vector_store = FAISS.from_texts(
                texts, 
                embeddings, 
                metadatas=metadatas
            )
            
            # Save new vector store
            os.makedirs(VECTOR_STORE_FOLDER, exist_ok=True)
            new_vector_store.save_local(VECTOR_STORE_FOLDER)
            
            print(f"Vector store rebuilt successfully with {len(all_docs)} chunks")
            
            # Print vector store info
            total_size = 0
            for file_name in os.listdir(VECTOR_STORE_FOLDER):
                file_path = os.path.join(VECTOR_STORE_FOLDER, file_name)
                if os.path.isfile(file_path):
                    file_size = os.path.getsize(file_path)
                    total_size += file_size
                    print(f"Vector store file: {file_name}, size: {file_size/1024/1024:.2f} MB")
            
            print(f"Total vector store size: {total_size/1024/1024:.2f} MB")
            return True
        else:
            print("No documents processed, vector store not updated")
            return False
            
    except Exception as e:
        print(f"Error rebuilding vector store: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=== Vector Store Rebuilder ===")
    print(f"This script will rebuild the vector store from documents in '{UPLOADS_FOLDER}'.")
    print(f"The new vector store will be saved in '{VECTOR_STORE_FOLDER}'.")
    print("WARNING: This will delete and recreate the entire vector store.")
    
    proceed = input("Do you want to proceed? (y/n): ").lower() == 'y'
    
    if proceed:
        print("Rebuilding vector store...")
        success = rebuild_vector_store()
        if success:
            print("Vector store rebuild completed successfully!")
        else:
            print("Vector store rebuild failed.")
    else:
        print("Operation cancelled.") 