import os
import pandas as pd
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Qdrant
import logging
from config import config


logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s: %(message)s', datefmt='%Y-%m-%d %I:%M:%S', level=logging.DEBUG)
logger = logging.getLogger(__name__)


qdrant_url=config["QDRANT_URL"]
qdrant_api_key=config["QDRANT_API_KEY"]

# embeddings = OpenAIEmbeddings()
# embeddings = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-base")
embeddings = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large-instruct")


def processExcel(file_path):
    # Create an ExcelFile object
    xlsx = pd.ExcelFile(file_path)

    # List all sheet names
    sheet_names = xlsx.sheet_names
    print("Sheet names:", sheet_names)

    for sheet_name in sheet_names:
        df = pd.read_excel(file_path, sheet_name=sheet_name)
        docs = df.iloc[:, 0].tolist()
        doc_store = Qdrant.from_texts(texts=docs, embedding=embeddings, url=qdrant_url, api_key=qdrant_api_key, collection_name=sheet_name+"_multilingual-e5-large-instruct", timeout=1000)
        print(doc_store)
        

if __name__ == "__main__":
    processExcel("data/Data for clustering (EverGrowing).xlsx")

