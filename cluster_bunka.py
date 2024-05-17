from bunkatopics import Bunka
from downloadQdrant import fetch_all_vectors, extracting
from config import config
from qdrant_client import QdrantClient
import pandas as pd



if __name__ == "__main__":
    qdrant_url=config["QDRANT_URL"]
    qdrant_api_key=config["QDRANT_API_KEY"]


    qdrant_client = QdrantClient(
        url=qdrant_url, 
        api_key=qdrant_api_key,
        timeout=1000,
    )

    collection_names = ["DE indo", "DE rf", "DE et", "EN outsider"]
    # collection_names = ["EN outsider"]
    for collection_name in collection_names:
        records = fetch_all_vectors(qdrant_client, collection_name)
        vectors, payloads = extracting(records)
        ids = [f"doc_{idx}" for idx, payload in enumerate(payloads)]

        pre_computed_embeddings = [{'doc_id': doc_id, 'embedding': embedding} for doc_id, embedding in zip(ids, vectors)]

        bunka = Bunka()
        bunka.fit(docs=payloads, ids = ids, pre_computed_embeddings = pre_computed_embeddings)


        # from sklearn.cluster import KMeans
        # clustering_model = KMeans(n_clusters=30)
        # bunka.get_topics(name_length=5, custom_clustering_model=clustering_model) # Specify the number of terms to describe each topic
        # print(bunka.df_topics_)

        bunka.get_topics(n_clusters=15, name_length=4)
        print(bunka.df_topics_)

        writer = pd.ExcelWriter('result/'+collection_name+'.xlsx', engine='openpyxl')
        for index, row in bunka.df_topics_.iterrows():
            print(f"Index: {index}, topic_id: {row['topic_id']}, topic_name: {row['topic_name']}, size: {row['size']}, percent: {row['percent']}")

            data=[]
            for doc in bunka.docs:
                if doc.topic_id == row['topic_id']:
                    data.append(doc.content)
            
            df = pd.DataFrame(data)
            df.to_excel(writer, sheet_name=f"{index}")
        writer.close()





