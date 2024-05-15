import os

import numpy as np
from qdrant_client import QdrantClient
import logging
from config import config


logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s: %(message)s', datefmt='%Y-%m-%d %I:%M:%S', level=logging.DEBUG)
logger = logging.getLogger(__name__)

# qdrant_url=config["QDRANT_URL"]
# qdrant_api_key=config["QDRANT_API_KEY"]


# qdrant_client = QdrantClient(
#     url=qdrant_url, 
#     api_key=qdrant_api_key,
# )


def fetch_all_vectors(_qdrant_client, _collection_name):
    response = _qdrant_client.count(
        collection_name=_collection_name,
    )
    count = response.count

    records = []
    response = _qdrant_client.scroll(
        collection_name=_collection_name,
        limit=count,
        with_payload=True,
        with_vectors=True,
    )
    records.extend(response[0])
    return records


def extracting(records):
    vectors = []
    payloads = []
    for record in records:
        vectors.append(np.asarray(record.vector))
        payloads.append(record.payload['page_content'])
    vectors = np.asarray(vectors)
    return vectors, payloads


if __name__ == "__main__":
    qdrant_url=config["QDRANT_URL"]
    qdrant_api_key=config["QDRANT_API_KEY"]


    qdrant_client = QdrantClient(
        url=qdrant_url, 
        api_key=qdrant_api_key,
    )

    collection_names = ["DE indo", "DE rf", "DE et", "EN outsider"]
    # collection_names = ["EN outsider"]
    for collection_name in collection_names:
        records = fetch_all_vectors(qdrant_client, collection_name)
        vectors, payloads = extracting(records)
        print(vectors, payloads)
