import os

from sklearn import datasets
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import hdbscan
import pandas as pd
import logging
from config import config
from downloadQdrant import fetch_all_vectors, extracting
from config import config
from qdrant_client import QdrantClient


logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s: %(message)s', datefmt='%Y-%m-%d %I:%M:%S', level=logging.DEBUG)
logger = logging.getLogger(__name__)


def clustering(vectors, _min_cluster_size=2):
    clusterer = hdbscan.HDBSCAN(algorithm='best', alpha=1.0, approx_min_span_tree=True,
        gen_min_span_tree=False, leaf_size=40, cluster_selection_method='leaf', prediction_data=True,
        metric='euclidean', min_cluster_size=_min_cluster_size, min_samples=None, p=None)

    clusterer.fit(vectors)
    # print(clusterer.labels_)
    # print(clusterer.probabilities_)
    # print(clusterer.labels_.max())

    soft_clusters = hdbscan.all_points_membership_vectors(clusterer)
    # print(soft_clusters)

    category_index = np.zeros(soft_clusters.shape[0], dtype=np.int64)
    category_prob = np.zeros(soft_clusters.shape[0], dtype=np.float64)
    for index, probs in enumerate(soft_clusters):
        max_index = np.argmax(probs)
        category_index[index] = max_index
        category_prob[index] = np.max(probs)
    # print(categories)
    return clusterer.labels_.max(), category_index, category_prob


def generate_report(categories, probs, payloads, file_name):
    writer = pd.ExcelWriter(file_name, engine='openpyxl')

    max_index = np.max(categories)
    for index in np.arange(0, max_index+1):
        matching_indices = [i for i, element in enumerate(categories) if element == index]
        matching_probs = probs[matching_indices]
        sorted_indices_prob = np.argsort(matching_probs)[::-1].tolist()
        
        # print(index, matching_indices, matching_probs, sorted_indices_prob)
        matching_indices = [matching_indices[value] for value in sorted_indices_prob]

        payload = [payloads[value] for value in matching_indices]
        data = { 'text': payload }
        df = pd.DataFrame(data)
        df.to_excel(writer, sheet_name=f"{index}")
    writer.close()


if __name__ == "__main__":
    qdrant_url=config["QDRANT_URL"]
    qdrant_api_key=config["QDRANT_API_KEY"]


    qdrant_client = QdrantClient(
        url=qdrant_url, 
        api_key=qdrant_api_key,
    )

    # collection_names = ["DE 700", "DE 800", "EN 4200"]
    collection_names = ["DE indo", "DE rf", "DE et", "EN outsider"]
    for collection_name in collection_names:
        records = fetch_all_vectors(qdrant_client, collection_name)
        vectors, payloads = extracting(records)
        sz = 2
        while True:
            category_size, categories, probs=clustering(vectors, _min_cluster_size=sz)
            if category_size < 20 and category_size > 10:
                break
            elif category_size < 10: 
                sz = sz - 1
                category_size, categories, probs=clustering(vectors, _min_cluster_size=sz)
                break
            sz = sz + 1
        generate_report(categories, probs, payloads, "result/"+f"{collection_name}_{category_size}.xlsx")

