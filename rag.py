import os
import pandas as pd
from app.common import logger
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings


current_dir = os.path.dirname(os.path.abspath(__file__))
faiss_vdb = os.path.join(
    current_dir, "..", "static", "dbs", "faiss_en_10k"
)
tsv_dir = os.path.join(
    current_dir, "..", "static", "tsv"
)


if os.path.isdir(faiss_vdb):
    logger.info(f"{faiss_vdb} found!")
    db = FAISS.load_local(faiss_vdb, OpenAIEmbeddings(), allow_dangerous_deserialization=True)
else:
    raise FileNotFoundError(f"{faiss_vdb} not found!")

def search(query: str):
    try:
        docs = db.similarity_search(query)
        return docs
    except Exception as e:
        logger.error(f"Failed to do the RAG: {str(e)}")
        return []
    
def to_json(docs: list):
    results = [{"page_content": r.page_content, "metadata": r.metadata} for r in docs]

    for r in results:
        r["table_data"] = pd.read_csv(os.path.join(tsv_dir, r['metadata']['source'].split("\\")[-1])).iloc[r['metadata']['row']].to_dict()
        
    return results
