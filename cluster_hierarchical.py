import os
from downloadQdrant import fetch_all_vectors, extracting
from config import config
from qdrant_client import QdrantClient
import pandas as pd
import numpy as np
import matplotlib.pyplot as mtp
import scipy.cluster.hierarchy as shc



if __name__ == "__main__":
    qdrant_url=config["QDRANT_URL"]
    qdrant_api_key=config["QDRANT_API_KEY"]


    qdrant_client = QdrantClient(
        url=qdrant_url, 
        api_key=qdrant_api_key,
        timeout=1000,
    )

    # collection_names = ["DE indo_multilingual-e5-large-instruct", "DE rf_multilingual-e5-large-instruct", "DE et_multilingual-e5-large-instruct", "EN outsider_multilingual-e5-large-instruct"]
    # collection_names = ["EN outsider_multilingual-e5-large-instruct"]
    collection_names = ["B2_multilingual-e5-large-instruct"]
    for collection_name in collection_names:
        records = fetch_all_vectors(qdrant_client, collection_name)
        vectors, payloads = extracting(records)
        
        linkage = shc.linkage(vectors, method='ward', metric='euclidean')
        # linkage = shc.linkage(vectors, method='complete', metric='euclidean')
        dendro = shc.dendrogram(linkage)

        mtp.title("Dendrogram Plot")
        mtp.xlabel("Embeddings")
        mtp.ylabel("Euclidean Distances")
        mtp.show()

        distances = linkage[:, 2]
        distances = distances[::-1]
        max_distance = max(distances)
        if distances[0] > max(distances[1], distances[2]) * 2:
            clusters = shc.fcluster(linkage, t=max_distance/5, criterion='distance')
        else:
            clusters = shc.fcluster(linkage, t=max_distance/4, criterion='distance')

        # inconsistent = shc.inconsistent(linkage)
        # coff = inconsistent[:, 3]
        # clusters = shc.fcluster(linkage, t=1.1, criterion='inconsistent')

        print(f'-----{collection_name}::max(clusters)={max(clusters)}-----')

        writer = pd.ExcelWriter(f'result/{collection_name}_{max(clusters)}.xlsx', engine='openpyxl')
        for cluster_idx in np.arange(min(clusters), max(clusters)+1):
            data = []
            for idx, cluster in enumerate(clusters):
                if cluster == cluster_idx:
                    data.append(payloads[idx])
            df = pd.DataFrame(data)
            df.to_excel(writer, sheet_name=f"{cluster_idx}")
        writer.close()

        # clusters = shc.fcluster(linkage, t=max_distance/5, criterion='distance')






