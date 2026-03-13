from langchain.retrievers import BM25Retriever
from langchain.schema import Document


class HybridRetriever:

    def __init__(self, vector_db, docs):

        self.vector_db = vector_db

        self.bm25 = BM25Retriever.from_documents(docs)
        self.bm25.k = 6

    def retrieve(self, query):

        # Dense retrieval
        dense_docs = self.vector_db.similarity_search(query, k=6)

        # BM25 retrieval
        bm25_docs = self.bm25.invoke(query)

        # Combine results
        combined = dense_docs + bm25_docs

        # Remove duplicates
        seen = set()
        unique_docs = []

        for doc in combined:
            text = doc.page_content[:450]

            if text not in seen:
                seen.add(text)
                unique_docs.append(doc)
        docs = unique_docs[:5]
        context = "\n\n".join([doc.page_content for doc in docs])
        return context,docs