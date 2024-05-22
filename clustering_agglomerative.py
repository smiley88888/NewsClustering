import os
from downloadQdrant import fetch_all_vectors, extracting
from config import config
from qdrant_client import QdrantClient
import pandas as pd
import numpy as np
import matplotlib.pyplot as mtp

from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram
from sklearn.metrics import silhouette_samples, silhouette_score
# from bunkatopics import Bunka



def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)


def agglomerative_clustering(X, payloads, n_clusters, prefix):
    clusterer = AgglomerativeClustering(n_clusters=n_clusters)
    cluster_labels = clusterer.fit_predict(X=X)

    writer = pd.ExcelWriter(f'result/{prefix}_{n_clusters}.xlsx', engine='openpyxl')
    for cluster_idx in np.arange(min(cluster_labels), max(cluster_labels)+1):
        data = []
        for idx, label in enumerate(cluster_labels):
            if label == cluster_idx:
                data.append(payloads[idx])
        df = pd.DataFrame(data)
        df.to_excel(writer, sheet_name=f"{cluster_idx}")
    writer.close()


def process(vectors, payloads, agglomerative_metric="euclidean", agglomerative_linkage="ward", silhouette_metric="euclidean", prefix=""):
    range_n_clusters = np.arange(10, 101)
    silhouette_coefficients=[]

    for n_clusters in range_n_clusters:
        clusterer = AgglomerativeClustering(n_clusters=n_clusters, metric=agglomerative_metric, linkage=agglomerative_linkage)
        cluster_labels = clusterer.fit_predict(vectors)

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
    mtp.savefig(f'result/{prefix}_hierarchical+{agglomerative_metric}+{agglomerative_linkage}_{silhouette_metric}.png', dpi=300)
    mtp.close()

    optimal_clusters = range_n_clusters[np.argmax(silhouette_coefficients)]
    print("optimal_clusters=", optimal_clusters)

    agglomerative_clustering(vectors, payloads, optimal_clusters, f'{prefix}_hierarchical+{agglomerative_metric}+{agglomerative_linkage}_{silhouette_metric}')



if __name__ == "__main__":
    qdrant_url=config["QDRANT_URL"]
    qdrant_api_key=config["QDRANT_API_KEY"]


    qdrant_client = QdrantClient(
        url=qdrant_url, 
        api_key=qdrant_api_key,
        timeout=1000,
    )
    
    metrics = ['euclidean', 'l1', 'l2', 'manhattan', 'cosine']
    linkages = ['ward', 'complete', 'average', 'single']
    # collection_names = ["DE indo_multilingual-e5-large-instruct", "DE rf_multilingual-e5-large-instruct", "DE et_multilingual-e5-large-instruct", "EN outsider_multilingual-e5-large-instruct"]
    # collection_names = ["DE indo_multilingual-e5-large-instruct", "EN outsider_multilingual-e5-large-instruct"]
    # collection_names = ["EN outsider_multilingual-e5-large-instruct", "EN outsider_pre_multilingual-e5-large-instruct", "DE indo_multilingual-e5-large-instruct", "DE indo_pre_multilingual-e5-large-instruct"]
    collection_names = ["EN outsider_pre_multilingual-e5-large-instruct", "DE indo_multilingual-e5-large-instruct", "DE indo_pre_multilingual-e5-large-instruct"]
    # collection_names = ["Indo_A_multilingual-e5-large-instruct", "Indo_B_multilingual-e5-large-instruct"]
    for collection_name in collection_names:
        print(f"----- collection name = {collection_name} -----")
        records = fetch_all_vectors(qdrant_client, collection_name)
        vectors, payloads = extracting(records)

        # for agglomerative_metric in metrics:
        #     for agglomerative_linkage in linkages:
        #         if agglomerative_linkage == 'ward' and agglomerative_metric != 'euclidean':
        #             continue
        #         for silhouette_metric in metrics:
        #             process(vectors, payloads, agglomerative_metric=agglomerative_metric, agglomerative_linkage=agglomerative_linkage, silhouette_metric=silhouette_metric, prefix=collection_name)

        process(vectors, payloads, agglomerative_metric='cosine', agglomerative_linkage='average', silhouette_metric='cosine', prefix=collection_name)
        process(vectors, payloads, agglomerative_metric='euclidean', agglomerative_linkage='ward', silhouette_metric='euclidean', prefix=collection_name)


