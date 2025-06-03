import torch
import pandas as pd
import logging
from datasets import load_dataset
from .metrics import precision_at_k, recall_at_k, exact_match, faithfulness
from .plot_utils import bar_chart
from .setup_eval_env import rag_pipeline

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger("RQ1Eval")

def evaluate_rq1(num_queries=1, save_plot_path='rq1_bar.png'):
    # Load a RAGBench subset (e.g., hotpotqa test split)
    ds = load_dataset("rungalileo/ragbench", "hotpotqa", split="test")
    methods = ['LLM-only', 'RAG', 'RAG+Summary']
    results = []
    for method in methods:
        # For demo, use the first num_queries samples
        y_true = []
        y_pred = []
        logger.info(f"Evaluating method: {method}")
        for i in range(num_queries):
            sample = ds[i]
            question = sample.get('question', '')
            gt = sample.get('response', '')
            y_true.append(gt)
            doc_id = f"hotpotqa_{sample['id']}_doc0"  # Only use first doc per sample for focused eval
            logger.info(f"Q{i+1}: {question}")
            logger.info(f"GT: {gt}")
            # Use RAG pipeline for answer
            if method == 'LLM-only':
                # Use only LLM, no retrieval (simulate by empty context)
                answer = rag_pipeline.llm_manager.generate(
                    prompt=question, temperature=0.1, max_tokens=128)
            elif method == 'RAG':
                answer = rag_pipeline.generate_answer(
                    question=question,
                    collection_id=doc_id,
                    temperature=0.1, max_tokens=128, use_reranking=True
                ).get("answer", "")
            elif method == 'RAG+Summary':
                # For now, same as RAG (customize if you have summary logic)
                answer = rag_pipeline.generate_answer(
                    question=question,
                    collection_id=doc_id,
                    temperature=0.1, max_tokens=128, use_reranking=True
                ).get("answer", "")
            logger.info(f"Pred: {answer}")
            y_pred.append(answer)
        # For metrics, use string match (EM) and dummy P@5/Recall@5 for now
        em = exact_match(y_true, y_pred)
        # For P@5/Recall@5, simulate top-5 with repeated answer (for real, use reranked docs)
        y_pred_top5 = [[a] * 5 for a in y_pred]
        p_at_5 = precision_at_k(y_true, y_pred_top5, k=5)
        r_at_5 = recall_at_k(y_true, y_pred_top5, k=5)
        faith = faithfulness(y_pred, y_true)
        logger.info(f"Results for {method}: EM={em}, P@5={p_at_5}, R@5={r_at_5}, Faithfulness={faith}")
        results.append({'Method': method, 'P@5': p_at_5, 'R@5': r_at_5, 'EM': em, 'Faithfulness': faith})
    df_results = pd.DataFrame(results)
    df_results.to_csv('rq1_results.csv', index=False)
    bar_chart(df_results, x='Method', y='P@5', title='RQ-1: P@5 for LLM-only vs RAG vs RAG+Summary', save_path=save_plot_path)
    print(f"RQ-1 results saved to rq1_results.csv and {save_plot_path}")

if __name__ == "__main__":
    evaluate_rq1(num_queries=1) 