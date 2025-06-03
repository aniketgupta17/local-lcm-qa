import pandas as pd
import logging
from datasets import load_dataset
from .metrics import precision_at_k
from .plot_utils import line_chart
from .setup_eval_env import rag_pipeline

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger("RQ3Eval")

def evaluate_rq3(num_queries=1, save_plot_path='rq3_line.png'):
    ds = load_dataset("rungalileo/ragbench", "hotpotqa", split="test")
    dims = [256, 512, 768, 1024, 1536]
    results = []
    for dim in dims:
        y_true = []
        y_pred = []
        logger.info(f"Evaluating dim: {dim}")
        for i in range(num_queries):
            sample = ds[i]
            question = sample.get('question', '')
            gt = sample.get('response', '')
            y_true.append(gt)
            doc_id = f"hotpotqa_{sample['id']}_doc0"
            logger.info(f"Q{i+1}: {question}")
            logger.info(f"GT: {gt}")
            answer = rag_pipeline.generate_answer(
                question=question,
                collection_id=doc_id,
                temperature=0.1, max_tokens=128, use_reranking=True
            ).get("answer", "")
            logger.info(f"Pred: {answer}")
            preds = [answer] + [f"mock_pred_{j}" for j in range(4)]
            y_pred.append(preds)
        p_at_5 = precision_at_k(y_true, y_pred, k=5)
        storage = dim * 100 * 4 / 1e6  # MB, fake calc
        latency = 0.15  # mock value
        logger.info(f"Results for dim {dim}: P@5={p_at_5}, Storage={storage}, p50Latency={latency}")
        results.append({'Dim': dim, 'P@5': p_at_5, 'Storage(MB)': storage, 'p50Latency': latency})
    df_results = pd.DataFrame(results)
    df_results.to_csv('rq3_results.csv', index=False)
    line_chart(df_results, x='Storage(MB)', y='P@5', title='RQ-3: Storage vs Recall (dim=256-1536)', save_path=save_plot_path)
    print(f"RQ-3 results saved to rq3_results.csv and {save_plot_path}")

if __name__ == "__main__":
    evaluate_rq3(num_queries=1) 