import faiss
import numpy as np
from .embedding import embed_query

class FAISSRetriever:
    def __init__(self, index_file, use_gpu=False):
        self.index = faiss.read_index(index_file)
        if use_gpu:
            # Check if the index is of a type that can be moved to GPU
            if isinstance(self.index, faiss.IndexHNSWFlat):
                print("Warning: IndexHNSWFlat does not support GPU. Using CPU instead.")
            else:
                try:
                    res = faiss.StandardGpuResources()
                    self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
                except RuntimeError as e:
                    print(f"Warning: Failed to move index to GPU: {e}. Using CPU instead.")
    
    def retrieve(self, query, top_k=5):
        """
        Retrieves the top-k most similar documents to the query.
        """
        query_embedding = embed_query(query)
        query_embedding = np.array(query_embedding).astype('float32')
        _, indices = self.index.search(query_embedding, top_k)
        return indices

# Example Usage:
# faiss_retriever = FAISSRetriever("faiss_index.index", use_gpu=False)
# results = faiss_retriever.retrieve("query text", top_k=5)
# print(results)
