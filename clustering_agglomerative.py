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
from bunkatopics import Bunka



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


def agglomerative_clustering(X, payloads, n_clusters, collection_name):
    clusterer = AgglomerativeClustering(n_clusters=n_clusters)
    cluster_labels = clusterer.fit_predict(X=X)

    writer = pd.ExcelWriter(f'result/{collection_name}_multilang E5 large instruct+AgglomerativeClustering+Silhoette_{n_clusters}.xlsx', engine='openpyxl')
    for cluster_idx in np.arange(min(cluster_labels), max(cluster_labels)+1):
        data = []
        for idx, label in enumerate(cluster_labels):
            if label == cluster_idx:
                data.append(payloads[idx])
        df = pd.DataFrame(data)
        df.to_excel(writer, sheet_name=f"{cluster_idx}")
    writer.close()


def bunka_clustering(X, payloads, n_clusters, collection_name):
    ids = [f"doc_{idx}" for idx, payload in enumerate(payloads)]

    pre_computed_embeddings = [{'doc_id': doc_id, 'embedding': embedding} for doc_id, embedding in zip(ids, X)]

    bunka = Bunka()
    bunka.fit(docs=payloads, ids = ids, pre_computed_embeddings = pre_computed_embeddings)
    bunka.get_topics(n_clusters=n_clusters, name_length=4)

    writer = pd.ExcelWriter(f'result/{collection_name}_multilang E5 large instruct+AgglomerativeClustering+Silhoette_Bunka_{n_clusters}.xlsx', engine='openpyxl')
    for index, row in bunka.df_topics_.iterrows():
        # print(f"Index: {index}, topic_id: {row['topic_id']}, topic_name: {row['topic_name']}, size: {row['size']}, percent: {row['percent']}")

        data=[]
        for doc in bunka.docs:
            if doc.topic_id == row['topic_id']:
                data.append(doc.content)
        
        df = pd.DataFrame(data)
        df.to_excel(writer, sheet_name=f"{index}_{row['topic_name']}")
    writer.close()



if __name__ == "__main__":
    qdrant_url=config["QDRANT_URL"]
    qdrant_api_key=config["QDRANT_API_KEY"]


    qdrant_client = QdrantClient(
        url=qdrant_url, 
        api_key=qdrant_api_key,
        timeout=1000,
    )

    # collection_names = ["DE indo_multilingual-e5-large-instruct", "DE rf_multilingual-e5-large-instruct", "DE et_multilingual-e5-large-instruct", "EN outsider_multilingual-e5-large-instruct"]
    collection_names = ["DE indo_multilingual-e5-large-instruct"]
    for collection_name in collection_names:
        print(f"----- collection name = {collection_name} -----")
        records = fetch_all_vectors(qdrant_client, collection_name)
        vectors, payloads = extracting(records)

        # model = AgglomerativeClustering(n_clusters=None, distance_threshold=0)
        # model = model.fit(vectors)

        # mtp.title("Hierarchical Clustering Dendrogram")
        # # plot the top three levels of the dendrogram
        # plot_dendrogram(model, truncate_mode="level")
        # mtp.xlabel("Number of points in node (or index of point if no parenthesis).")
        # mtp.show()

        # print(model.labels_, max(model.labels_))
        
        range_n_clusters = np.arange(2, 81)
        silhouette_coefficients=[]

        for n_clusters in range_n_clusters:
            clusterer = AgglomerativeClustering(n_clusters=n_clusters)
            cluster_labels = clusterer.fit_predict(vectors)

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

        optimal_clusters = range_n_clusters[np.argmax(silhouette_coefficients)]
        print("optimal_clusters=", optimal_clusters)

        agglomerative_clustering(vectors, payloads, optimal_clusters, collection_name)
        bunka_clustering(vectors, payloads, optimal_clusters, collection_name)


