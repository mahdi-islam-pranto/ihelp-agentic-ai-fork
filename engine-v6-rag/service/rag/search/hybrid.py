from service.rag.search.fuzzy import fuzzy_retriever
from service.rag.search.semantic import semantic_retriever

def hybrid_langchain_retriever(query, lc_documents, vectorstore, k=5):
    # Step 1: Fuzzy (best for names & typos)
    fuzzy_results = fuzzy_retriever(query, lc_documents, k)

    if fuzzy_results:
        return {
            "strategy": "fuzzy",
            "documents": fuzzy_results
        }

    # Step 2: Semantic fallback
    semantic_results = semantic_retriever(vectorstore, query, k)

    return {
        "strategy": "semantic",
        "documents": semantic_results
    }
