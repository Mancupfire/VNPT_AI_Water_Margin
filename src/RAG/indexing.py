import faiss
from langchain_community.vectorstores import FAISS
from .embedding import CustomEmbeddings

class FAISSIndexer:
    def __init__(self, use_gpu=False):
        self.embeddings = CustomEmbeddings()
        self.vector_store = None
        self.use_gpu = use_gpu

    def add_documents(self, documents):
        if self.vector_store is None:
            self.vector_store = FAISS.from_texts(documents, self.embeddings)
            if self.use_gpu:
                self.vector_store.index = faiss.index_cpu_to_gpu(faiss.StandardGpuResources(), 0, self.vector_store.index)
        else:
            self.vector_store.add_texts(documents)

    def save_index(self, folder_path):
        if self.use_gpu:
            self.vector_store.index = faiss.index_gpu_to_cpu(self.vector_store.index)
        self.vector_store.save_local(folder_path)

    def load_index(self, folder_path):
        self.vector_store = FAISS.load_local(folder_path, self.embeddings, allow_dangerous_deserialization=True)
        if self.use_gpu:
            self.vector_store.index = faiss.index_cpu_to_gpu(faiss.StandardGpuResources(), 0, self.vector_store.index)

    def retrieve(self, query, top_k=4):
        """
        Retrieves the top-k most similar documents to the query and returns their content.
        """
        retriever = self.vector_store.as_retriever(search_kwargs={'k': top_k})
        results = retriever.invoke(query)
        return [doc.page_content for doc in results]
