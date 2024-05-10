import os

from sklearn import datasets
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import hdbscan
import pandas as pd

from langchain.chains import LLMChain
# from langchain_community.embeddings import OpenAIEmbeddings
from langchain.embeddings import HuggingFaceEmbeddings

from langchain_community.vectorstores import Qdrant
# from newsapi import NewsApiClient
from qdrant_client import QdrantClient

from dotenv import load_dotenv

load_dotenv()


qdrant_url="https://ec9d4088-4356-4871-82d3-ec182c2a9187.us-east4-0.gcp.cloud.qdrant.io:6333"
qdrant_api_key="l38XCLdqnaJ2q9VAe2bI_XZXIsp3QWXpRV6gUcd95SufCmi6I7QHCg"

# embeddings = OpenAIEmbeddings()
embeddings = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-base")


def uploadQdrant():
    file_path = "data/data for clustering.xlsx"
    df1 = pd.read_excel(file_path, sheet_name='EN 4200')
    df2 = pd.read_excel(file_path, sheet_name='DE 800')
    df3 = pd.read_excel(file_path, sheet_name='DE 700') 

    docs1 = df1.iloc[:, 0].tolist()
    docs2 = df2.iloc[:, 0].tolist()
    docs3 = df3.iloc[:, 0].tolist()

    doc_store1 = Qdrant.from_texts(texts=docs1, embedding=embeddings, url=qdrant_url, api_key=qdrant_api_key, collection_name="EN 4200", timeout=1000)
    print(doc_store1)

    doc_store2 = Qdrant.from_texts(texts=docs2, embedding=embeddings, url=qdrant_url, api_key=qdrant_api_key, collection_name="DE 800", timeout=1000)
    print(doc_store2)

    doc_store3 = Qdrant.from_texts(texts=docs3, embedding=embeddings, url=qdrant_url, api_key=qdrant_api_key, collection_name="DE 700", timeout=1000)
    print(doc_store3)


# uploadQdrant()

qdrant_client = QdrantClient(
    url=qdrant_url, 
    api_key=qdrant_api_key,
)

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
    # if len(response[0]) == 0:
    #     break
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


def clustering(vectors):
    clusterer = hdbscan.HDBSCAN(algorithm='best', alpha=1.0, approx_min_span_tree=True,
        gen_min_span_tree=False, leaf_size=40, cluster_selection_method='leaf', prediction_data=True,
        metric='euclidean', min_cluster_size=2, min_samples=None, p=None)

    clusterer.fit(vectors)
    print(clusterer.labels_)
    print(clusterer.probabilities_)
    print(clusterer.labels_.max())

    soft_clusters = hdbscan.all_points_membership_vectors(clusterer)
    # print(soft_clusters)

    category_index = np.zeros(soft_clusters.shape[0], dtype=np.int64)
    category_prob = np.zeros(soft_clusters.shape[0], dtype=np.float64)
    for index, probs in enumerate(soft_clusters):
        max_index = np.argmax(probs)
        category_index[index] = max_index
        category_prob[index] = np.max(probs)
    # print(categories)
    return category_index, category_prob


def generate_report(categories, probs, payloads, file_name):
    writer = pd.ExcelWriter(file_name, engine='openpyxl')

    max_index = np.max(categories)
    for index in np.arange(0, max_index+1):
        matching_indices = [i for i, element in enumerate(categories) if element == index]
        matching_probs = probs[matching_indices]
        sorted_indices_prob = np.argsort(matching_probs)[::-1].tolist()
        
        print(index, matching_indices, matching_probs, sorted_indices_prob)
        matching_indices = [matching_indices[value] for value in sorted_indices_prob]

        payload = [payloads[value] for value in matching_indices]
        data = { 'text': payload }
        df = pd.DataFrame(data)
        df.to_excel(writer, sheet_name=f"{index}")
    writer.close()

de700_records = fetch_all_vectors(qdrant_client, "DE 700")
de700_vectors, de700_payloads = extracting(de700_records)
de700_categories, de700_probs=clustering(de700_vectors)
generate_report(de700_categories, de700_probs, de700_payloads, "result/"+"DE 700.xlsx")

de800_records = fetch_all_vectors(qdrant_client, "DE 800")
de800_vectors, de800_payloads = extracting(de800_records)
de800_categories, de800_probs=clustering(de800_vectors)
generate_report(de800_categories, de800_probs, de800_payloads, "result/"+"DE 800.xlsx")

en4200_records = fetch_all_vectors(qdrant_client, "EN 4200")
en4200_vectors, en4200_payloads = extracting(en4200_records)
en4200_categories, en4200_probs=clustering(en4200_vectors)
generate_report(en4200_categories, en4200_probs, en4200_payloads, "result/"+"EN 4200.xlsx")

