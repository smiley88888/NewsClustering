import os
from downloadQdrant import fetch_all_vectors, extracting
from config import config
from qdrant_client import QdrantClient
import pandas as pd
import numpy as np
import matplotlib.pyplot as mtp

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score


if __name__ == "__main__":
    qdrant_url=config["QDRANT_URL"]
    qdrant_api_key=config["QDRANT_API_KEY"]


    qdrant_client = QdrantClient(
        url=qdrant_url, 
        api_key=qdrant_api_key,
        timeout=1000,
    )

    # collection_names = ["DE indo_multilingual-e5-large-instruct", "DE rf_multilingual-e5-large-instruct", "DE et_multilingual-e5-large-instruct", "EN outsider_multilingual-e5-large-instruct"]
    collection_names = ["EN outsider_multilingual-e5-large-instruct"]
    # collection_names = ["EN outsider"]
    for collection_name in collection_names:
        print(f"----- collection name = {collection_name} -----")
        records = fetch_all_vectors(qdrant_client, collection_name)
        vectors, payloads = extracting(records)

        range_n_clusters = np.arange(2, 100)
        silhouette_coefficients=[]
        for n_clusters in range_n_clusters:
            clusterer = KMeans(n_clusters=n_clusters, random_state=10)
            cluster_labels = clusterer.fit_predict(vectors)

            # The silhouette_score gives the average value for all the samples.
            # This gives a perspective into the density and separation of the formed
            # clusters
            silhouette_avg = silhouette_score(vectors, cluster_labels)
            silhouette_coefficients.append(silhouette_avg)
            print("For n_clusters =", n_clusters, "The average silhouette_score is :", silhouette_avg)

        mtp.figure(figsize=(10,5))
        mtp.plot(range_n_clusters, silhouette_coefficients)
        mtp.xlabel("k")
        mtp.ylabel("silhouette coefficient")
        mtp.legend()
        mtp.grid(True)
        mtp.show()
        




