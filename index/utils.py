import numpy as np
from multiprocessing import Pool
import pandas as pd
from haystack import Document
from timeit import default_timer as timer
from claim import climate_identifier
from codetiming import Timer

import indexer_interface


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


def read_csv_yield_haystack_documents(filename, chunk_size, start_from_row):
    chunk_number = 1
    for df in pd.read_csv(filename, chunksize=chunk_size, skiprows=range(1, start_from_row)):
        print(f'starting to index chunk number {chunk_number}')
        df.fillna("", inplace=True)
        row_dict = df.to_dict('records')
        chunk_number += 1
        yield [convert_openalex_abstracts_to_haystack_documents(row)
               for row in row_dict]


def index_docs_from_csv(filename, docs_extractor,
                        indexer: indexer_interface.IndexerInterface,
                        chunk_size, start_from_row,
                        check_climate_related=True):
    for docs in docs_extractor(filename, chunk_size, start_from_row):
        if check_climate_related:
            docs = filter_climate_related(docs)
        indexer.write_documents(docs)


def filter_climate_related(docs):
    abstracts = list(map(lambda doc: doc.content, docs))
    labels, _ = climate_identifier.is_about_climate(abstracts)
    return [doc for label, doc in zip(labels, docs) if label == 'Yes']


@Timer(text="get_abstracts_matching_claims elapsed time: {seconds:.0f} s")
def get_abstracts_matching_claims(claims,
                                  indexer: indexer_interface.IndexerInterface,
                                  top_k=10, debug=False):
    start = timer()
    all_matches = indexer.retrieve_matches_for_phrases(claims,
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
