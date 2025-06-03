from datasets import load_dataset
from setup_eval_env import vector_store, document_processor
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger("IngestRAGBench")

def ingest_hotpotqa_first5():
    ds = load_dataset("rungalileo/ragbench", "hotpotqa", split="test")
    for i in range(5):
        sample = ds[i]
        doc_id = f"hotpotqa_{sample['id']}"
        logger.info(f"Ingesting sample {i+1} with doc_id: {doc_id}")
        for j, doc_text in enumerate(sample.get('documents', [])):
            chunked = document_processor.process_text(doc_text, metadata={"source": f"hotpotqa_{i}_doc{j}"})
            if chunked:
                vector_store.add_documents(doc_id=f"{doc_id}_doc{j}", chunks=[{"text": c.page_content, "metadata": c.metadata} for c in chunked])
                logger.info(f"Added {len(chunked)} chunks for doc {j} of sample {i+1}")
            else:
                logger.warning(f"No chunks produced for doc {j} of sample {i+1}")
    logger.info("Ingestion complete for first 5 hotpotqa samples.")

if __name__ == "__main__":
    ingest_hotpotqa_first5() 