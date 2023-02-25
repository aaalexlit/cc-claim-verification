import gc
import itertools

from timeit import default_timer as timer

import pandas as pd
import logging

import utils
from claim import claim_identifier

logging.basicConfig(format="%(levelname)s - %(name)s -  %(message)s", level=logging.WARNING)
# logging.getLogger("haystack").setLevel(logging.INFO)

# pd.set_option('display.max_rows', 500)
# pd.set_option('display.max_columns', 500)
# pd.set_option('display.width', 1000)

MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'
# embedding size used by all-MiniLM-L6-v2
EMBEDDING_DIM = 384


def convert_abstracts_from_openalex_to_haystack_docs(filename):
    id_col = 'id'
    chunk_number = 1
    for df in pd.read_csv(filename, chunksize=20):
        print(f'starting to index chunk number {chunk_number}')
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

    utils.index_docs_from_csv(
        '../data/OpenAlex/csv/openalex_data_by_title_inf_2023-02-18_06-26-29.csv',
        convert_abstracts_from_openalex_to_haystack_docs,
        MODEL_NAME,
        EMBEDDING_DIM)

    end = timer()
    print(end - start)

# print(retrieve_matches_for_a_phrase("elevation of temperature"))
