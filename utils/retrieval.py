def retrieve_context(vector_db, query, k=4):

    docs = vector_db.similarity_search(query, k=k)

    context = "\n\n".join([doc.page_content for doc in docs])

    return context, docs