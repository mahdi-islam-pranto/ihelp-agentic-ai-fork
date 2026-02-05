

def semantic_retriever(vectorstore, query, k=5):
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k}
    )

    return retriever.get_relevant_documents(query)
