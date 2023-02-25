import math
from datetime import datetime

import pandas as pd
import os

import openalex_utils
import openalex_settings
from sensitive_info import email

headers = {}
if email is not None:
    headers["email"] = email

MAIN_DIR = "../data/OpenAlex"
JSON_DIR = f"{MAIN_DIR}/json"
CSV_DIR = f"{MAIN_DIR}/csv"

for folder in [MAIN_DIR, JSON_DIR, CSV_DIR]:
    if not os.path.exists(folder):
        os.makedirs(folder)


def main(args):
    queries_by = ["title", "abstract"]
    max_requests = math.inf

    for query_by in queries_by:
        url = openalex_utils.query_to_openalex_url(openalex_settings.concept_groups, query_by)

        works_list = openalex_utils.get_paginated_results(url,
                                                          f"{JSON_DIR}/by_{query_by}",
                                                          max_requests,
                                                          headers,
                                                          resume=True)
        df = pd.DataFrame.from_dict(works_list)

        df.to_csv(
            f"{CSV_DIR}/openalex_data_by_{query_by}_{max_requests}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.csv",
            index=False)
        print(df.shape)
        print(df.head())


if __name__ == "__main__":
    main([])
