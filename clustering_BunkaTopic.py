import os
from downloadQdrant import fetch_all_vectors, extracting
from config import config
from qdrant_client import QdrantClient
import pandas as pd
import numpy as np
import matplotlib.pyplot as mtp

from bunkatopics import Bunka
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.preprocessing import normalize


def bunka_clustering(X, payloads, n_clusters, prefix):
    ids = [f"doc_{idx}" for idx, payload in enumerate(payloads)]

    pre_computed_embeddings = [{'doc_id': doc_id, 'embedding': embedding} for doc_id, embedding in zip(ids, X)]

    bunka = Bunka()
    bunka.fit(docs=payloads, ids = ids, pre_computed_embeddings = pre_computed_embeddings)
    bunka.get_topics(n_clusters=n_clusters, name_length=4)

    writer = pd.ExcelWriter(f'result/{prefix}_{n_clusters}.xlsx', engine='openpyxl')
    for index, row in bunka.df_topics_.iterrows():
        # print(f"Index: {index}, topic_id: {row['topic_id']}, topic_name: {row['topic_name']}, size: {row['size']}, percent: {row['percent']}")
        data=[]
        for doc in bunka.docs:
            if doc.topic_id == row['topic_id']:
                data.append(doc.content)
        
        df = pd.DataFrame(data)
        df.to_excel(writer, sheet_name=f"{index}_{row['topic_name']}")
    writer.close()


def process(vectors, payloads, silhouette_metric="euclidean", prefix=""):
    range_n_clusters = np.arange(10, 101)
    silhouette_coefficients=[]

    ids = [f"doc_{idx}" for idx, payload in enumerate(payloads)]
    pre_computed_embeddings = [{'doc_id': doc_id, 'embedding': embedding} for doc_id, embedding in zip(ids, vectors)]

    for n_clusters in range_n_clusters:
        bunka = Bunka()
        bunka.fit(docs=payloads, ids = ids, pre_computed_embeddings = pre_computed_embeddings)
        bunka.get_topics(n_clusters=n_clusters, name_length=4)
        
        cluster_labels=[]
        for doc in bunka.docs:
            # for index, topic in enumerate(bunka.topics):
            #     if doc.topic_id == topic.topic_id:
            #         cluster_labels.append(index)
            #         break
            label = doc.topic_id.replace("bt-", "")
            label = int(label)
            cluster_labels.append(label)
        cluster_labels=np.array(cluster_labels)
        silhouette_avg = silhouette_score(vectors, cluster_labels, metric=silhouette_metric)
        silhouette_coefficients.append(silhouette_avg)
        print("For n_clusters =", n_clusters, "The average silhouette_score is :", silhouette_avg)

    mtp.figure(figsize=(16,9))
    mtp.plot(range_n_clusters, silhouette_coefficients)
    mtp.xlabel("k")
    mtp.ylabel("silhouette coefficient")
    mtp.legend()
    mtp.grid(True)
    # mtp.show()
    mtp.savefig(f'result/{prefix}_bunka_{silhouette_metric}.png', dpi=300)
    mtp.close()

    optimal_clusters = range_n_clusters[np.argmax(silhouette_coefficients)]
    print("optimal_clusters=", optimal_clusters)

    bunka_clustering(vectors, payloads, optimal_clusters, f'{prefix}_bunka_{silhouette_metric}')



if __name__ == "__main__":
    qdrant_url=config["QDRANT_URL"]
    qdrant_api_key=config["QDRANT_API_KEY"]


    qdrant_client = QdrantClient(
        url=qdrant_url, 
        api_key=qdrant_api_key,
        timeout=1000,
    )

    metrics = ['euclidean', 'l1', 'l2', 'manhattan', 'cosine']

    # collection_names = ["DE indo_multilingual-e5-large-instruct", "DE rf_multilingual-e5-large-instruct", "DE et_multilingual-e5-large-instruct", "EN outsider_multilingual-e5-large-instruct"]
    # collection_names = ["DE indo_multilingual-e5-large-instruct", "EN outsider_multilingual-e5-large-instruct"]
    # collection_names = ["Indo_A_multilingual-e5-large-instruct", "Indo_B_multilingual-e5-large-instruct"]
    # collection_names = ["EN outsider_multilingual-e5-large-instruct", "EN outsider_pre_multilingual-e5-large-instruct", "DE indo_multilingual-e5-large-instruct", "DE indo_pre_multilingual-e5-large-instruct"]
    collection_names = ["DE indo_multilingual-e5-large-instruct", "DE indo_pre_multilingual-e5-large-instruct"]
    for collection_name in collection_names:
        print(f"----- collection name = {collection_name} -----")
        records = fetch_all_vectors(qdrant_client, collection_name)
        vectors, payloads = extracting(records)
        normalized_vectors = normalize(vectors)

        # for silhouette_metric in metrics:
        #     process(vectors, payloads, silhouette_metric=silhouette_metric, prefix=collection_name)

        process(normalized_vectors, payloads, silhouette_metric='cosine', prefix=collection_name)


