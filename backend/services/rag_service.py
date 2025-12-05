import json
from typing import List, Optional, Dict, Any
from functools import lru_cache
import logging
import os

from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

from backend.config import get_settings
from backend.models import RAGDocument, SearchResult

logger = logging.getLogger(__name__)

class RAGService:
    """FAISS-based RAG service for knowledge base"""
    
    def __init__(self):
        self.settings = get_settings()
        self.embedding_model = None
        self.index = None
        self.documents = []
        self.is_initialized = False
    
    async def initialize(self):
        """Initialize embedding model and FAISS index"""
        try:
            # Load embedding model
            self.embedding_model = SentenceTransformer(
                self.settings.EMBEDDING_MODEL
            )
            
            # Load knowledge base
            kb_data = self._load_knowledge_base()
            self._build_index(kb_data)
            
            self.is_initialized = True
            logger.info("RAG Service initialized successfully")
        except Exception as e:
            logger.error(f"RAG initialization error: {e}")
            raise
    
    def _load_knowledge_base(self) -> dict:
        """Load restaurant_kb.json"""
        try:
            with open(self.settings.KB_FILE, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.warning("Knowledge base file not found, using empty KB")
            return {"menu": [], "faqs": [], "restaurant_info": {}}
    
    def _build_index(self, kb_data: dict):
        """Build FAISS index from knowledge base"""
        self.documents = []
        embeddings = []
        
        # Add menu items
        for item in kb_data.get("menu", []):
            content = f"{item.get('name', '')} {item.get('description', '')}"
            self.documents.append(RAGDocument(
                id=f"menu_{item.get('id', 'unknown')}",
                content=content,
                source="menu",
                metadata=item
            ))
        
        # Add FAQs
        for faq in kb_data.get("faqs", []):
            self.documents.append(RAGDocument(
                id=f"faq_{len(self.documents)}",
                content=f"{faq.get('question', '')} {faq.get('answer', '')}",
                source="faq",
                metadata=faq
            ))
        
        # Generate embeddings
        if self.documents:
            texts = [doc.content for doc in self.documents]
            embeddings = self.embedding_model.encode(texts)
            
            # Create FAISS index
            dimension = embeddings.shape[1]
            self.index = faiss.IndexFlatL2(dimension)
            self.index.add(embeddings.astype('float32'))
    
    async def search_menu(self, query: str, top_k: int = None) -> List[SearchResult]:
        """Search menu items"""
        if not self.is_initialized:
            await self.initialize()
        
        top_k = top_k or self.settings.TOP_K_RESULTS
        
        # Encode query
        query_embedding = self.embedding_model.encode([query]).astype('float32')
        
        # Search
        distances, indices = self.index.search(query_embedding, top_k)
        
        results = []
        for dist, idx in zip(distances[0], indices):
            if idx >= 0 and idx < len(self.documents):
                doc = self.documents[idx]
                results.append(SearchResult(
                    document=doc,
                    score=float(1 / (1 + dist)),  # Convert distance to similarity
                    metadata=doc.metadata
                ))
        
        return results
    
    async def get_restaurant_info(self, topic: str = None) -> str:
        """Get restaurant info"""
        try:
            kb_data = self._load_knowledge_base()
            info = kb_data.get("restaurant_info", {})
            
            if topic:
                return str(info.get(topic, f"No info found for {topic}"))
            
            return json.dumps(info, indent=2)
        except Exception as e:
            return f"Error retrieving info: {str(e)}"

@lru_cache()
def get_rag_service() -> RAGService:
    """Get cached RAG service instance"""
    return RAGService()
