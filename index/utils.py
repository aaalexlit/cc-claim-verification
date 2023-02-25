import gc
import os

import numpy as np
from multiprocessing import Pool
import pandas as pd
from haystack import Document
from haystack.document_stores import FAISSDocumentStore
from haystack.nodes import EmbeddingRetriever

PATH_TO_FAISS = os.path.abspath("../data/faiss")

if not os.path.exists(PATH_TO_FAISS):
    os.makedirs(PATH_TO_FAISS)

PATH_TO_INDEX = os.path.join(PATH_TO_FAISS, "faiss_index")
PATH_TO_DB = os.path.join(PATH_TO_FAISS, 'faiss_document_store.db')


def parallelize_dataframe(df, func, n_cores=4):
    df_split = np.array_split(df, n_cores)
    pool = Pool(n_cores)
    df = pd.concat(pool.map(func, df_split))
    pool.close()
    pool.join()
    return df


def convert_openalex_claims_to_haystack_document(row):
    return [{'content': claim,
             'meta': {
                 'title': row['title'],
                 'publication_year': row['publication_year'],
                 'authors': row['authors'],
                 'doi': row['doi'],
                 'open_alex_id': row['id']
             }} for claim in row['claims']]


def convert_openalex_abstracts_to_haystack_documents(row):
    meta_information = {
        'title': row['title'],
        'publication_year': row['publication_year'],
        'authors': row['authors'],
        'doi': row['doi'],
        'open_alex_id': row['id']
    }
    return Document(content=row['abstract'],
                    meta=meta_information)


def index_docs_from_csv(filename, docs_extractor, model_name, embedding_dim):
    for docs in docs_extractor(filename):
        add_documents_to_faiss_index(docs, model_name, embedding_dim)
        gc.collect()


def add_documents_to_faiss_index(docs, model_name, embedding_dim):
    print(f'Adding next {len(docs)} docs to the index')
    doc_store = get_faiss_document_store(embedding_dim)
    retriever = get_retriever(doc_store, model_name)
    write_documents(docs, doc_store, retriever)


# TODO: if there's a DB but no index erase the DB
def get_faiss_document_store(embedding_dim):
    if os.path.exists(PATH_TO_INDEX):
        return FAISSDocumentStore.load(index_path=PATH_TO_INDEX)
    else:
        return FAISSDocumentStore(
            sql_url=f"sqlite:///{PATH_TO_DB}",
            return_embedding=True,
            similarity='cosine',
            embedding_dim=embedding_dim,
            duplicate_documents='skip'
        )


def get_retriever(document_store, model_name, progress_bar=True):
    return EmbeddingRetriever(
        document_store=document_store,
        embedding_model=model_name,
        model_format='sentence_transformers',
        # include article title into the embedding
        embed_meta_fields=["title"],
        progress_bar=progress_bar
    )


def write_documents(docs, document_store, retriever):
    document_store.write_documents(docs)

    print('Updating embeddings ...')

    document_store.update_embeddings(
        retriever=retriever,
        update_existing_embeddings=False
    )

    print(f'current embedding count is {document_store.get_embedding_count()}')
    print('Saving document store')
    document_store.save(index_path=PATH_TO_INDEX)


def retrieve_matches_for_a_phrase(phrase, embedding_dim, model_name, top_k=10):
    doc_store = get_faiss_document_store(embedding_dim)
    retriever = get_retriever(doc_store, model_name, progress_bar=False)
    return retriever.retrieve(phrase, top_k=top_k)


def retrieve_matches_for_phrases(phrases, embedding_dim, model_name, top_k=10):
    doc_store = get_faiss_document_store(embedding_dim)
    retriever = get_retriever(doc_store, model_name, progress_bar=True)
    return retriever.retrieve_batch(phrases, top_k=top_k)
