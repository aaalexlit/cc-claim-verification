import pandas as pd

from index import utils

import logging
from timeit import default_timer as timer

logging.basicConfig(format="%(levelname)s - %(name)s -  %(message)s", level=logging.WARNING)

MODEL_NAME = 'sentence-transformers/msmarco-MiniLM-L-6-v3'
# embedding size used by msmarco-MiniLM-L-6-v3
EMBEDDING_DIM = 384

chunk_size = 1800
start_from_row = 328 * chunk_size


def convert_openalex_abstracts_to_haystack_documents(filename):
    chunk_number = 1
    for df in pd.read_csv(filename, chunksize=chunk_size, skiprows=start_from_row):
        print(f'starting to index chunk number {chunk_number}')
        df.fillna("", inplace=True)
        row_dict = df.to_dict('records')
        chunk_number += 1
        yield [utils.convert_openalex_abstracts_to_haystack_documents(row)
               for row in row_dict]


def main(args):
    start = timer()

    utils.index_docs_from_csv(
        '../data/OpenAlex/csv/openalex_data_by_abstract_inf_2023-02-18_09-15-53.csv',
        convert_openalex_abstracts_to_haystack_documents,
        MODEL_NAME,
        EMBEDDING_DIM)

    end = timer()
    print(end - start)


def get_abstracts_matching_claims(claims, top_k=10, debug=False):
    start = timer()
    all_matches = utils.retrieve_matches_for_phrases(claims,
                                                     EMBEDDING_DIM,
                                                     MODEL_NAME,
                                                     top_k=top_k)
    if debug:
        for claim_n, matches in enumerate(all_matches):
            print(f"Claim:\n{claims[claim_n]}\n")
            for i, match in enumerate(matches):
                print(f'Evidence {i}:\n',
                      f'Similarity: {match.score:.3f}\n'
                      f'Quote: {match.content}\n',
                      f'Article Title: {match.meta.get("title", "")}\n',
                      f'DOI: {match.meta.get("doi", "")}\n',
                      f'year: {match.meta.get("publication_year", "")}\n', )
    end = timer()
    print(f"Took {(end - start):.0f} seconds")
    return all_matches

# get_abstracts_matching_claims(['CO2 is not the cause of our current warming trend.'])
#
# get_abstracts_matching_claims(["CO2 is not the cause of our current warming trend.",
#                                 "Arctic sea ice has expanded in recent years",
#                                 "Polar bears’ population is growing and is not threatened by climate change.",
#                                 "CO2 is good for plant life"])
