from langchain.retrievers import BM25Retriever
from langchain.schema import Document
from collections import defaultdict

# Using Reciprocal Rank Fusion with Dense and BM25 ranking
class HybridRetriever:

    def __init__(self, vector_db, docs):

        self.vector_db = vector_db

        self.bm25 = BM25Retriever.from_documents(docs)
        self.bm25.k = 6

        self.rrf_k = 60


    def _rrf(self, dense_docs, bm25_docs):

        scores = defaultdict(float)
        doc_lookup = {}

        # Dense ranking
        for rank, doc in enumerate(dense_docs):
            key = doc.page_content
            doc_lookup[key] = doc
            scores[key] += 1 / (self.rrf_k + rank + 1)

        # BM25 ranking
        for rank, doc in enumerate(bm25_docs):
            key = doc.page_content
            doc_lookup[key] = doc
            scores[key] += 1 / (self.rrf_k + rank + 1)

        # Sort by score
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        docs = [doc_lookup[text] for text, _ in ranked]

        return docs


    def retrieve(self, query):

        # Dense retrieval
        dense_docs = self.vector_db.similarity_search(query, k=6)

        # BM25 retrieval
        bm25_docs = self.bm25.invoke(query)

        # Fuse rankings
        fused_docs = self._rrf(dense_docs, bm25_docs)

        # Keep top results
        docs = fused_docs[:5]

        context = "\n\n".join([doc.page_content for doc in docs])

        return context, docs