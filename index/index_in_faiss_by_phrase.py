import gc
import itertools

from timeit import default_timer as timer

import pandas as pd
import logging

import utils
from claim import claim_identifier
from index.faiss.faiss_indexer import FAISSIndexer

logging.basicConfig(format="%(levelname)s - %(name)s -  %(message)s", level=logging.WARNING)
# logging.getLogger("haystack").setLevel(logging.INFO)

MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'
# embedding size used by all-MiniLM-L6-v2
EMBEDDING_DIM = 384

chunk_size = 20
start_from_row = 0 * chunk_size


def convert_abstracts_from_openalex_to_haystack_docs(filename, chunk_size, start_from_row):
    id_col = 'id'
    chunk_number = 1
    for df in pd.read_csv(filename, chunksize=chunk_size, skiprows=range(1, start_from_row)):
        print(f'starting to index chunk number {chunk_number}')
        df.fillna("", inplace=True)
        claims_df = claim_identifier.get_claims_from_texts(df[[id_col, 'abstract']])
        print('Finished extracting claims')
        df = df.merge(claims_df, on=id_col)
        del claims_df
        gc.collect()
        df.drop(columns=['abstract'], inplace=True)
        row_dict = df.to_dict('records')
        chunk_number += 1
        yield list(itertools.chain(*[utils.convert_openalex_claims_to_haystack_document(row) for row in row_dict]))


def main(args):
    start = timer()

    faiss_indexer = FAISSIndexer('../data/faiss/claim_phrases_test', MODEL_NAME, EMBEDDING_DIM)

    utils.index_docs_from_csv('../data/OpenAlex/csv/openalex_data_by_title_inf_2023-02-18_06-26-29.csv',
                              convert_abstracts_from_openalex_to_haystack_docs,
                              faiss_indexer,
                              chunk_size,
                              start_from_row
                              )

    end = timer()
    print(end - start)


if __name__ == "__main__":
    main([])
