from haystack.document_stores import MilvusDocumentStore
from haystack.nodes import EmbeddingRetriever

import indexer_interface
import os

class MilvusIndexer(indexer_interface.IndexerInterface):
    def __init__(self, milvus_host, milvus_port, sql_url,
                 recreate_index, model_name, embedding_dim) -> None:
        self.milvus_host = milvus_host
        self.milvus_port = milvus_port
        self.sql_url = sql_url
        if not sql_url.startswith('postgres'):
            if not os.path.exists(self.sql_url):
                os.makedirs(self.sql_url)
            self.sql_url = f"sqlite:///{os.path.join(self.sql_url, 'document_store.db')}"

        self.recreate_index = recreate_index
        self.model_name = model_name
        self.embedding_dim = embedding_dim
        self.document_store = self._init_document_store()
        self.retriever = self._init_retriever()

    def _init_document_store(self):
        return MilvusDocumentStore(
            sql_url=self.sql_url,
            host=self.milvus_host,
            port=self.milvus_port,
            index="cc_abstracts",
            index_type="IVF_FLAT",
            similarity='dot_product',
            duplicate_documents='skip',
            embedding_dim=self.embedding_dim,
            recreate_index=self.recreate_index
        )

    def _init_retriever(self, progress_bar=True):
        return EmbeddingRetriever(
            document_store=self.document_store,
            embedding_model=self.model_name,
            model_format='sentence_transformers',
            # include article title into the embedding
            embed_meta_fields=["title"],
            progress_bar=progress_bar
        )

    def write_documents(self, docs):
        self.document_store.write_documents(docs)

        print('Updating embeddings ...')

        self.document_store.update_embeddings(
            retriever=self.retriever,
            update_existing_embeddings=False
        )

        print(f'current embedding count is {self.document_store.get_embedding_count()}')

    def retrieve_matches_for_a_phrase(self, phrase, top_k=10):
        return self.retriever.retrieve(phrase, top_k=top_k)

    def retrieve_matches_for_phrases(self, phrases, top_k=10):
        return self.retriever.retrieve_batch(phrases, top_k=top_k)
