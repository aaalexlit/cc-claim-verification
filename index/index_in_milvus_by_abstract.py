from index import utils
import os
import logging
from timeit import default_timer as timer

from index.milvus.milvus_indexer import MilvusIndexer

logging.basicConfig(format="%(levelname)s - %(name)s -  %(message)s", level=logging.WARNING)

MODEL_NAME = 'sebastian-hofstaetter/distilbert-dot-tas_b-b256-msmarco'
EMBEDDING_DIM = 768

# chunk_size = 1800
# start_from_row = 328 * chunk_size
chunk_size = 100
start_from_row = 0 * chunk_size

sqlite_url = '../data/milvus'
remote_sqlite_url = '../data/milvus/remote'
postgres_url = "postgresql://pgadmin:pass@localhost:5432/postgres"
remote_milvus_host = sensitive_info.zillis_milvus_host
remote_milvus_port = "19537"

remote_index_type = "AUTOINDEX"
local_index_type = "IVF_FLAT"


def main(args):
    start = timer()

    milvus_indexer = MilvusIndexer(
        milvus_host=remote_milvus_host,
        milvus_port=remote_milvus_port,
        sql_url=remote_sqlite_url,
        recreate_index=False,
        model_name=MODEL_NAME,
        embedding_dim=EMBEDDING_DIM,
        index_type=remote_index_type)
    utils.index_docs_from_csv('../data/OpenAlex/csv/openalex_data_by_abstract_inf_2023-02-18_09-15-53.csv',
                              utils.read_csv_yield_haystack_documents,
                              milvus_indexer,
                              chunk_size,
                              start_from_row)

    end = timer()
    print(end - start)


def get_indexer(remote=False):
    if remote:
        return MilvusIndexer(
            milvus_host=remote_milvus_host,
            milvus_port=remote_milvus_port,
            sql_url=sqlite_url,
            recreate_index=False,
            model_name=MODEL_NAME,
            embedding_dim=EMBEDDING_DIM,
            index_type=remote_index_type
        )
    else:
        return MilvusIndexer(
            milvus_host="localhost",
            milvus_port=19530,
            sql_url=sqlite_url,
            recreate_index=False,
            model_name=MODEL_NAME,
            embedding_dim=EMBEDDING_DIM,
            index_type=local_index_type
        )


if __name__ == "__main__":
    main([])
