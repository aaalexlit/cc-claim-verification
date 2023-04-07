import os.path
import time

import pandas
import pandas as pd
import requests
import json
import numpy as np
from multiprocessing import Pool
from pathlib import Path
from codetiming import Timer

from sensitive_info import S2_API_KEY

pd.set_option('display.max_columns', None)

BASE_DIR = Path(__file__).resolve(strict=True).parent.parent.parent
input_filename = 'openalex_data_by_abstract_inf_2023-02-18_09-15-53.csv'
input_file = f'{BASE_DIR}/data/OpenAlex/csv/{input_filename}'
chunk_size = 200
start_from_row = 3127 * chunk_size
add_info_path = f'{BASE_DIR}/data/sem_scholar/csv'
add_info_filename = input_filename.replace('openalex', 'mod')


@Timer(f"with chunk size: {chunk_size}")
def main():
    chunk_number = 3128
    Path(add_info_path).mkdir(parents=True, exist_ok=True)
    add_info_file = os.path.join(add_info_path, add_info_filename)
    for df in pd.read_csv(input_file, chunksize=chunk_size, skiprows=range(1, start_from_row)):
        print(f'starting to fetch info for chunk number {chunk_number}')
        df.dropna(subset=['doi'], inplace=True)

        res = parallelize_dataframe(df, modify_and_add_info)
        if chunk_number == 1:
            res.to_csv(add_info_file, index=False)
        else:
            res.to_csv(add_info_file, mode='a', header=False, index=False)
        chunk_number += 1


def modify_and_add_info(df):
    if df.empty:
        return df
    modify_original(df)
    additional_info_df = retrieve_from_semanticscholar_by_doi(list(df.doi))
    res = pd.merge(df, additional_info_df, on='doi')
    res['abstract'] = res.apply(lambda r: r.abstract_y if r.abstract_y else r.abstract_x, axis=1)
    res.drop(columns=['abstract_x', 'abstract_y'], inplace=True)
    return res


def modify_original(df):
    # remove rows with empty DOI
    df.drop(columns=["authors"], inplace=True)
    df.fillna("", inplace=True)
    # remove link from openalex
    df.loc[:, "id"] = df.apply(lambda r: r.id[21:], axis=1)
    # remove link from doi
    df.loc[:, "doi"] = df.apply(lambda r: r.doi[16:], axis=1)
    df.rename(columns={"id": "openalex_id",
                       "publication_year": "year"}, inplace=True)


def retrieve_from_semanticscholar_by_doi(dois, retries_left=3):
    url = "https://api.semanticscholar.org/graph/v1/paper/batch?fields=title,abstract,citationCount,influentialCitationCount"
    headers = {}
    if S2_API_KEY is not None:
        headers["x-api-key"] = S2_API_KEY
    request_body = {"ids": dois}
    r = requests.post(url, headers=headers, json=request_body)
    if r.status_code != 200 and retries_left > 0:
        if not ('error' in r.json() and r.json()['error'] == 'No valid paper ids given'):
            time.sleep(60)
            print(f"retrying {retries_left}")
        return retrieve_from_semanticscholar_by_doi(dois, retries_left - 1)
    elif r.status_code != 200:
        print("no retries left")
        print(json.dumps(r.json(), indent=4))
    return parse_response_to_df(r, dois)


def parse_response_to_df(resp, list_of_dois):
    keep_fields = ['paperId', 'abstract', 'citationCount', 'influentialCitationCount']
    work_dicts = []
    if 'error' not in resp.json():
        for work, doi in zip(resp.json(), list_of_dois):
            if work:
                work_min = {k: work[k] for k in keep_fields}
                work_min['doi'] = doi
                work_dicts.append(work_min)
            else:
                work_dicts.append({'paperId': '',
                                   'abstract': '',
                                   'citationCount': '',
                                   'influentialCitationCount': '',
                                   'doi': doi})
    else:
        print("Error!!!", json.dumps(resp.json(), indent=4), list_of_dois)
        for doi in list_of_dois:
            work_dicts.append({'paperId': '',
                               'abstract': '',
                               'citationCount': '',
                               'influentialCitationCount': '',
                               'doi': doi})
    return pandas.DataFrame(work_dicts)


def parallelize_dataframe(df, func, n_cores=4):
    df_split = np.array_split(df, n_cores)
    pool = Pool(n_cores)
    df = pd.concat(pool.map(func, df_split))
    pool.close()
    pool.join()
    return df


if __name__ == '__main__':
    main()
    # print(retrieve_from_semanticscholar_by_doi(['10.1093/icesjms/fsr195', '10.1175/jamc-d-11-0256.1',
    #                                             '10.1111/j.1365-2699.2012.02690.x']))
