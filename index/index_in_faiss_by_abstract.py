from index.faiss.faiss_indexer import FAISSIndexer
from index import utils

import logging
from timeit import default_timer as timer

logging.basicConfig(format="%(levelname)s - %(name)s -  %(message)s", level=logging.WARNING)

MODEL_NAME = 'sentence-transformers/msmarco-MiniLM-L-6-v3'
# embedding size used by msmarco-MiniLM-L-6-v3
EMBEDDING_DIM = 384

# chunk_size = 1800
# start_from_row = 328 * chunk_size
chunk_size = 100
start_from_row = 0 * chunk_size


def main(args):
    start = timer()
    faiss_indexer = FAISSIndexer('../data/faiss/abstracts_test', MODEL_NAME, EMBEDDING_DIM)
    utils.index_docs_from_csv('../data/OpenAlex/csv/openalex_data_by_abstract_inf_2023-02-18_09-15-53.csv',
                              utils.read_csv_yield_haystack_documents,
                              faiss_indexer,
                              chunk_size,
                              start_from_row)

    end = timer()
    print(end - start)


def get_indexer():
    return FAISSIndexer('../data/faiss', MODEL_NAME, EMBEDDING_DIM)


if __name__ == "__main__":
    main([])
