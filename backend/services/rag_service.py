import json
import pickle
from typing import List, Optional
from functools import lru_cache
import numpy as np

from sentence_transformers import SentenceTransformer
import faiss

from config import get_settings
from models import RAGDocument, SearchResult

class RAGService:
    """
    RAG service with FAISS vector store for semantic search on the knowledge base.
    """
    
    def __init__(self):
        self.settings = get_settings()
        self.model: Optional[SentenceTransformer] = None
        self.index: Optional[faiss.Index] = None
        self.documents: List[RAGDocument] = []
        self._initialized = False
    
    async def initialize(self):
        """Initialize embedding model and build or load the FAISS index."""
        if self._initialized:
            return
        
        print("🔄 Initializing RAG Service...")
        
        # Load embedding model
        print(f"  > Loading embedding model: {self.settings.EMBEDDING_MODEL}")
        self.model = SentenceTransformer(self.settings.EMBEDDING_MODEL, device=self.settings.STT_DEVICE)
        
        # Check if index already exists
        index_file = f"{self.settings.FAISS_INDEX}.index"
        docs_file = f"{self.settings.FAISS_INDEX}.pkl"

        try:
            print("  > Attempting to load existing FAISS index...")
            self.index = faiss.read_index(index_file)
            with open(docs_file, 'rb') as f:
                self.documents = pickle.load(f)
            print(f"  ✓ FAISS index and documents loaded successfully with {self.index.ntotal} vectors.")
        except (FileNotFoundError, RuntimeError):
            print("  > Index not found. Building a new one from the knowledge base.")
            with open(self.settings.KB_FILE, 'r', encoding='utf-8') as f:
                kb_data = json.load(f)
            
            self.documents = self._kb_to_documents(kb_data)
            print(f"  > Loaded {len(self.documents)} documents from KB.")
            
            await self._build_index()

        self._initialized = True
        print("✅ RAG Service initialized.")
    
    def _kb_to_documents(self, kb_data: dict) -> List[RAGDocument]:
        """Convert knowledge base JSON into a list of RAG documents."""
        documents = []
        info = kb_data['restaurant_info']
        documents.append(RAGDocument(id="info_general", content=f"{info['name']} is an {info['cuisine']} restaurant at {info['location']}. Phone: {info['phone']}.", metadata={"type": "restaurant_info"}))
        
        hours_text = "Operating hours are: " + ", ".join([f"{day}: {hours}" for day, hours in info['hours'].items()])
        documents.append(RAGDocument(id="info_hours", content=hours_text, metadata={"type": "restaurant_info"}))

        for policy, value in info['policies'].items():
            documents.append(RAGDocument(id=f"policy_{policy}", content=f"Policy on {policy.replace('_', ' ')}: {value}", metadata={"type": "policy"}))

        for item in kb_data['menu']:
            content = f"Menu item: {item['name']} ({item['category']}) costs {item['price']}. Description: {item['description']}."
            if item.get('dietary'):
                content += f" Dietary info: {', '.join(item['dietary'])}."
            documents.append(RAGDocument(id=item['id'], content=content, metadata={"type": "menu", "name": item['name'], "price": item['price']}))

        for faq in kb_data['faqs']:
            documents.append(RAGDocument(id=f"faq_{faq['question']}", content=f"Question: {faq['question']} Answer: {faq['answer']}", metadata={"type": "faq"}))
        
        return documents
    
    async def _build_index(self):
        """Build FAISS index from documents and save to disk."""
        print("  > Building FAISS index...")
        texts = [doc.content for doc in self.documents]
        embeddings = self.model.encode(texts, show_progress_bar=True)
        
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(np.array(embeddings, dtype='float32'))
        
        print(f"  ✓ FAISS index built with {self.index.ntotal} vectors.")
        self._save_index()
    
    def _save_index(self):
        """Save the FAISS index and documents to disk."""
        index_file = f"{self.settings.FAISS_INDEX}.index"
        docs_file = f"{self.settings.FAISS_INDEX}.pkl"
        try:
            faiss.write_index(self.index, index_file)
            with open(docs_file, 'wb') as f:
                pickle.dump(self.documents, f)
            print(f"  ✓ Index saved to {index_file} and {docs_file}")
        except Exception as e:
            print(f"  ❌ Failed to save index: {e}")
    
    async def search(self, query: str, top_k: Optional[int] = None) -> List[SearchResult]:
        """Perform a semantic search against the knowledge base."""
        if not self._initialized:
            await self.initialize()
        
        k = top_k or self.settings.TOP_K_RESULTS
        query_embedding = self.model.encode([query])
        
        distances, indices = self.index.search(np.array(query_embedding, dtype='float32'), k)
        
        results = []
        for i, idx in enumerate(indices[0]):
            if idx != -1:
                doc = self.documents[idx]
                score = 1.0 / (1.0 + distances[0][i]) # Convert L2 distance to a similarity score
                results.append(SearchResult(content=doc.content, score=score, metadata=doc.metadata))
        return results

@lru_cache()
def get_rag_service() -> RAGService:
    """Get a cached singleton instance of the RAGService."""
    service = RAGService()
    return service