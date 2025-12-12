import faiss
import numpy as np
from .embedding import chunk_and_embed

class FAISSIndexer:
    def __init__(self, dimension, use_gpu=False):
        self.dimension = dimension
        self.use_gpu = use_gpu
        if self.use_gpu:
            cpu_index = faiss.IndexFlatL2(dimension)
            self.index = faiss.index_cpu_to_gpu(faiss.StandardGpuResources(), 0, cpu_index)
        else:
            self.index = faiss.IndexHNSWFlat(dimension, 32)  # 32 is the number of neighbors to keep during search
    
    def add_documents(self, documents):
        """
        Adds a list of documents to the FAISS index.
        """
        embeddings = chunk_and_embed(documents)
        self.index.add(np.array(embeddings).astype('float32'))
    
    def save_index(self, index_file):
        faiss.write_index(self.index, index_file)
    
    def load_index(self, index_file):
        self.index = faiss.read_index(index_file)

# Example Usage:
# faiss_indexer = FAISSIndexer(dimension=768, use_gpu=False)
# faiss_indexer.add_documents(["Document 1 text", "Document 2 text"])
# faiss_indexer.save_index("faiss_index.index")
