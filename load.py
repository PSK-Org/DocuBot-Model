from haystack.document_stores import InMemoryDocumentStore
from haystack.utils import fetch_archive_from_http
from haystack.pipelines.standard_pipelines import TextIndexingPipeline
from datasets import load_dataset
import os


def init_store(config):
    document_store = InMemoryDocumentStore(use_bm25=True)
    
    doc_dir = f"data/{config.document_dir}"
    
    return doc_dir, document_store

def download_data(config, doc_dir):
    
    if config.is_downloading_documents:
        fetch_archive_from_http(
            url="https://s3.eu-central-1.amazonaws.com/deepset.ai-farm-qa/datasets/documents/wiki_gameofthrones_txt1.zip",
            output_dir=doc_dir
        )
    

        
        
def load_doc(config, document_store, doc_dir):
    # move out of the output folder
    os.chdir(os.getcwd().split("/outputs")[0])
    print(f"cwd {os.getcwd()}")
    dirs = os.listdir(doc_dir)
    files_to_index = [doc_dir + "/" + f for f in dirs]
    indexing_pipeline = TextIndexingPipeline(document_store)
    indexing_pipeline.run_batch(file_paths=files_to_index)