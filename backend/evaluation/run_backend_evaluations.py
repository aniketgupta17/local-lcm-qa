import os
import json
import time
import logging
import pandas as pd
import numpy as np
from datasets import load_dataset
from backend.evaluation import metrics, plot_utils
from backend.evaluation.backend_api import upload_document, get_answer

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger("RunBackendEvals")

BACKEND_URL = "http://127.0.0.1:8001"
RESULTS_DIR = "backend/evaluation/results"
os.makedirs(RESULTS_DIR, exist_ok=True)

MAX_DOC_SIZE = 4000  # Maximum characters per document (if needed)
MAX_RETRIES = 3
RETRY_DELAY = 2  # seconds

def upload_with_retry(content, filename, max_retries=MAX_RETRIES):
    """Upload document with retry logic."""
    for attempt in range(max_retries):
        try:
            return upload_document(content, filename)
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            logger.warning(f"Upload attempt {attempt + 1} failed: {e}. Retrying in {RETRY_DELAY}s...")
            time.sleep(RETRY_DELAY)

def get_sample_info(sample, domain):
    """Extract relevant fields from sample, with fallbacks for different dataset structures."""
    info = {
        'domain': domain,
        'question': sample.get('question', sample.get('query', '')),
        'ground_truth': sample.get('answer', sample.get('response', sample.get('ground_truth', ''))),
        'context': sample.get('context', sample.get('documents', [''])[0] if isinstance(sample.get('documents', []), list) else '')
    }
    if not info['question'] or not info['ground_truth'] or not info['context']:
        raise ValueError(f"Missing required fields in sample from {domain}")
    return info

def run_evaluation(num_samples_per_domain=1):
    domains = ["hotpotqa", "pubmedqa", "cuad", "finqa", "techqa"]
    all_samples = []
    for domain in domains:
        try:
            ds = load_dataset("rungalileo/ragbench", domain, split="test")
            # Get multiple samples per domain
            samples_count = 0
            for i in range(min(num_samples_per_domain, len(ds))):
                sample = ds[i]
                sample_info = get_sample_info(sample, domain)
                all_samples.append(sample_info)
                samples_count += 1
            logger.info(f"Loaded {samples_count} samples from {domain}")
        except Exception as e:
            logger.error(f"Failed to load {domain}: {e}")
            continue
    logger.info(f"Loaded {len(all_samples)} samples from {len(domains)} domains")
    results = {
        'questions': [],
        'ground_truth': [],
        'predictions': [],
        'contexts': [],
        'domains': [],
        'latencies': [],
        'doc_ids': [],
        'confidences': []
    }
    for sample in all_samples:
        domain = sample['domain']
        question = sample['question']
        ground_truth = sample['ground_truth']
        context = sample['context']
        logger.info(f"\nProcessing {domain} query: {question}")
        logger.info(f"Ground truth: {ground_truth[:100]}...")  # Only show first 100 chars
        logger.info(f"Context length: {len(context)} chars")
        # Only upload the first MAX_DOC_SIZE chars if context is too large
        doc_content = context[:MAX_DOC_SIZE] if len(context) > MAX_DOC_SIZE else context
        try:
            start_time = time.time()
            doc_filename = f"{domain}_{len(results['questions'])}.txt"
            doc_id = upload_with_retry(doc_content, doc_filename)
            logger.info(f"Uploaded document. ID: {doc_id}")
            time.sleep(1)  # Brief pause for backend
            
            # Get answer and confidence score
            answer, confidence = get_answer(
                document_id=doc_id,
                question=question,
                temperature=0.1,
                max_tokens=128,
                hybrid_alpha=0.7
            )
            end_time = time.time()
            
            # Store results
            results['questions'].append(question)
            results['ground_truth'].append(ground_truth)
            results['predictions'].append(answer)
            results['contexts'].append(context)
            results['domains'].append(domain)
            results['latencies'].append(end_time - start_time)
            results['doc_ids'].append(doc_id)
            results['confidences'].append(confidence)
            
            logger.info(f"Prediction: {answer[:100]}{'...' if len(answer) > 100 else ''}")
            logger.info(f"Confidence: {confidence:.2f}")
            logger.info(f"Latency: {end_time - start_time:.2f}s")
        except Exception as e:
            logger.error(f"Error processing {domain} query: {e}")
            continue
    if not results['questions']:
        logger.error("No results collected. Evaluation failed.")
        return None
    predictions_list = [[pred] for pred in results['predictions']]
    eval_metrics = metrics.compute_metrics_with_time(
        y_true=results['ground_truth'],
        y_pred=predictions_list,
        contexts=results['contexts'],
        domains=results['domains'],
        latencies=results['latencies'],
        confidences=results['confidences']
    )
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(RESULTS_DIR, f"evaluation_results_{timestamp}.json")
    summary_stats = {
        'total_queries': len(results['questions']),
        'domains_covered': len(set(results['domains'])),
        'mean_latency': eval_metrics['latency']['mean'],
        'p90_latency': eval_metrics['latency']['p90'],
        'overall_metrics': {
            'hallucination_reduction': eval_metrics['hallucination_scores']['overall'],
            'source_grounding': eval_metrics['source_overlap']['overall'],
            'confidence_calibration': eval_metrics['confidence_calibration']['overall'],
            'exact_match': eval_metrics['exact_match']['overall'],
            'precision@5': eval_metrics['precision@k']['overall'],
            'trace_score': eval_metrics['trace_scores']['overall']
        }
    }
    with open(results_file, 'w') as f:
        json.dump({
            'summary': summary_stats,
            'raw_results': results,
            'metrics': eval_metrics
        }, f, indent=2)
        
    # Always plot all non-zero metrics with multiple visualization types
    plot_utils.plot_metrics_by_domain(eval_metrics, RESULTS_DIR, timestamp)
    
    logger.info("\n=== Evaluation Summary ===")
    logger.info(f"Total queries: {summary_stats['total_queries']}")
    logger.info(f"Domains covered: {summary_stats['domains_covered']}")
    logger.info("\nOverall System Performance:")
    logger.info(f"Hallucination Reduction: {summary_stats['overall_metrics']['hallucination_reduction']:.3f}")
    logger.info(f"Source Grounding: {summary_stats['overall_metrics']['source_grounding']:.3f}")
    logger.info(f"Confidence Calibration: {summary_stats['overall_metrics']['confidence_calibration']:.3f}")
    logger.info(f"TRACe Score: {summary_stats['overall_metrics']['trace_score']:.3f}")
    logger.info("\nMetrics by domain:")
    for domain in set(results['domains']):
        logger.info(f"\n{domain.upper()}:")
        logger.info(f"Exact Match: {eval_metrics['exact_match'].get(domain, 0):.3f}")
        logger.info(f"Precision@5: {eval_metrics['precision@k'].get(domain, 0):.3f}")
        logger.info(f"Hallucination Score: {eval_metrics['hallucination_scores'].get(domain, 0):.3f}")
        logger.info(f"Source Overlap: {eval_metrics['source_overlap'].get(domain, 0):.3f}")
        logger.info(f"Confidence Calibration: {eval_metrics['confidence_calibration'].get(domain, 0):.3f}")
        logger.info(f"TRACe Score: {eval_metrics['trace_scores'].get(domain, 0):.3f}")
    logger.info("\nLatency Stats:")
    logger.info(f"Mean: {eval_metrics['latency']['mean']:.2f}s")
    logger.info(f"P90: {eval_metrics['latency']['p90']:.2f}s")
    logger.info(f"P99: {eval_metrics['latency']['p99']:.2f}s")
    logger.info(f"\nPlots saved to: {os.path.join(RESULTS_DIR, f'plots_{timestamp}')}")
    return results_file

if __name__ == "__main__":
    results_file = run_evaluation(num_samples_per_domain=1)  # 1 sample per domain = 5 total queries
    if results_file:
        logger.info(f"\nResults saved to: {results_file}")
    else:
        logger.error("Evaluation failed to complete.") 