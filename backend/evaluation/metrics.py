import torch
import re
import time
import numpy as np
import string
from collections import defaultdict

def is_not_found(answer):
    """Detect if the answer is a 'not found' or irrelevant response."""
    if not answer or not isinstance(answer, str):
        return True
    answer = answer.strip().lower()
    patterns = [
        r"not found in the provided documents",
        r"i don't have enough information",
        r"the context does not mention",
        r"the context only discusses",
        r"the question refers to a different",
        r"sorry",
        r"no relevant information",
        r"cannot answer",
        r"no answer",
        r"no, ",
        r"not mentioned"
    ]
    return any(re.search(p, answer) for p in patterns)

def preprocess_text(text):
    """Clean text for better matching."""
    if not text or not isinstance(text, str):
        return ""
    # Convert to lowercase and remove punctuation
    text = text.lower()
    text = re.sub(r'\[document \d+\]', '', text)  # Remove document citations
    text = re.sub(r'\[\d+\]', '', text)  # Remove reference markers
    text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
    return text.strip()

def compute_source_overlap(pred, context):
    """Compute how much of the prediction is supported by the source context."""
    if is_not_found(pred) or not context:
        return 0.0
    
    # Preprocess 
    pred = preprocess_text(pred)
    context = preprocess_text(context)
    
    if not pred:
        return 0.0
    
    # For very short answers, use direct substring check
    if len(pred.split()) <= 5:
        return 1.0 if pred in context else 0.0
    
    # Tokenize into n-grams for phrase matching
    def get_ngrams(text, n=3):
        words = text.split()
        if len(words) < n:
            return set([' '.join(words)])
        return set(' '.join(words[i:i+n]) for i in range(len(words)-n+1))
    
    # Use multiple n-gram sizes for better coverage
    overlap_scores = []
    for n in [1, 2, 3]:
        pred_ngrams = get_ngrams(pred, n)
        context_ngrams = get_ngrams(context, n)
        
        if not pred_ngrams:
            continue
        
        overlap = len(pred_ngrams & context_ngrams)
        overlap_scores.append(overlap / len(pred_ngrams))
    
    # Return average overlap across different n-gram sizes
    return np.mean(overlap_scores) if overlap_scores else 0.0

def compute_hallucination_score(pred, ref, context):
    """
    Compute a hallucination score (0-1) where:
    0 = completely hallucinated
    1 = fully grounded in context and matches reference
    """
    if is_not_found(pred):
        # For not found responses, check if there's actually a valid answer
        # in the context. If there is, this is a false negative
        context_ref_overlap = compute_source_overlap(ref, context)
        if context_ref_overlap > 0.5:
            return 0.5  # Penalize for missing valid information
        return 1.0  # Not found is correct when info is truly not present
        
    # Source grounding score (60% weight)
    source_score = compute_source_overlap(pred, context)
    
    # Reference similarity score (40% weight)
    ref_score = compute_trace_score(pred, ref)
    
    # Combine for final score
    return 0.6 * source_score + 0.4 * ref_score

def compute_trace_score(pred, ref):
    """Compute TRACe score (faithfulness + context relevance + completeness)."""
    if is_not_found(pred):
        return 0.0
    
    # Preprocess texts
    pred = preprocess_text(pred)
    ref = preprocess_text(ref)
    
    if not pred or not ref:
        return 0.0
    
    # Compute word overlap for basic faithfulness
    pred_words = set(pred.split())
    ref_words = set(ref.split())
    
    if not pred_words:
        return 0.0
    
    # Compute Jaccard similarity for better balance
    intersection = len(pred_words & ref_words)
    union = len(pred_words | ref_words)
    
    jaccard = intersection / union if union > 0 else 0
    
    # Also compute simple overlap
    overlap = intersection / len(pred_words) if pred_words else 0
    
    # Return weighted combination
    return 0.7 * jaccard + 0.3 * overlap

def extract_citations(text):
    """Extract document citations from text."""
    if not text or not isinstance(text, str):
        return []
    citation_patterns = [
        r'\[Document \d+\]',
        r'\[doc\d+\]',
        r'\[document\d+\]',
        r'\[(?:source|reference) \d+\]'
    ]
    citations = []
    for pattern in citation_patterns:
        citations.extend(re.findall(pattern, text, re.IGNORECASE))
    return citations

def compute_confidence_calibration(pred, ref, confidence_score):
    """
    Compute how well calibrated the model's confidence is with actual performance.
    confidence_score: float between 0-1 indicating model's confidence
    """
    if is_not_found(pred):
        # If model says "not found", confidence should be low
        return 1.0 if confidence_score < 0.5 else (1.0 - confidence_score)
    
    # Check if prediction has citations
    citations = extract_citations(pred)
    has_citations = len(citations) > 0
    
    # Calculate actual performance score
    actual_score = compute_trace_score(pred, ref)
    
    # If answer has citations, adjust expected confidence level
    expected_confidence = actual_score
    if has_citations:
        expected_confidence = max(0.6, expected_confidence)
    
    # Calculate calibration error and convert to score (1 = perfectly calibrated)
    calibration_error = abs(confidence_score - expected_confidence)
    calibration_score = 1.0 - calibration_error
    
    return max(0.0, calibration_score)

def precision_at_k_by_domain(y_true, y_pred, domains, k=5):
    """
    Compute precision@k grouped by domain.
    y_true: list of strings (ground truth answers)
    y_pred: list of lists of strings (each inner list contains k predictions)
    domains: list of strings (domain labels)
    """
    domain_results = defaultdict(list)
    domain_metrics = {}
    
    for true, preds, domain in zip(y_true, y_pred, domains):
        # Ensure preds is a list
        if isinstance(preds, str):
            preds = [preds]
        
        # Take top-k predictions
        preds_k = preds[:k]
        
        # If all top-k are 'not found', count as incorrect
        if all(is_not_found(p) for p in preds_k):
            domain_results[domain].append(0.0)
            continue
            
        # Check if any prediction matches ground truth
        if any(compute_trace_score(p, true) > 0.5 and not is_not_found(p) for p in preds_k):
            domain_results[domain].append(1.0)
        else:
            domain_results[domain].append(0.0)
    
    # Compute average for each domain
    for domain, scores in domain_results.items():
        domain_metrics[domain] = sum(scores) / len(scores) if scores else 0.0
    
    # Also compute overall
    all_scores = [score for scores in domain_results.values() for score in scores]
    domain_metrics['overall'] = sum(all_scores) / len(all_scores) if all_scores else 0.0
    
    return domain_metrics

def compute_metrics_with_time(y_true, y_pred, contexts, domains, latencies, confidences=None, k=5):
    """
    Compute all metrics including latency analysis and hallucination metrics.
    y_true: list of strings (ground truth answers)
    y_pred: list of strings or list of lists of strings (predictions)
    contexts: list of strings (source contexts)
    domains: list of strings (domain labels)
    latencies: list of floats (query latencies in seconds)
    confidences: list of floats (model confidence scores, optional)
    """
    if confidences is None:
        confidences = [0.5] * len(y_true)  # Default confidence if not provided
        
    metrics = {
        'exact_match': exact_match_by_domain(y_true, y_pred, domains),
        'precision@k': precision_at_k_by_domain(y_true, y_pred, domains, k),
        'recall@k': recall_at_k_by_domain(y_true, y_pred, domains, k),
        'trace_scores': defaultdict(list),
        'hallucination_scores': defaultdict(list),
        'confidence_calibration': defaultdict(list),
        'source_overlap': defaultdict(list),
        'latency': {
            'mean': sum(latencies) / len(latencies) if latencies else 0,
            'median': sorted(latencies)[len(latencies)//2] if latencies else 0,
            'min': min(latencies) if latencies else 0,
            'max': max(latencies) if latencies else 0,
            'p90': np.percentile(latencies, 90) if latencies else 0,
            'p99': np.percentile(latencies, 99) if latencies else 0
        }
    }
    
    # Compute scores by domain
    for pred, ref, context, domain, conf in zip(y_pred, y_true, contexts, domains, confidences):
        # If pred is a list, take the first prediction
        if isinstance(pred, list):
            pred = pred[0] if pred else ""
            
        metrics['trace_scores'][domain].append(compute_trace_score(pred, ref))
        metrics['hallucination_scores'][domain].append(compute_hallucination_score(pred, ref, context))
        metrics['confidence_calibration'][domain].append(compute_confidence_calibration(pred, ref, conf))
        metrics['source_overlap'][domain].append(compute_source_overlap(pred, context))
    
    # Compute overall averages BEFORE replacing lists with floats
    for metric_name in ['trace_scores', 'hallucination_scores', 'confidence_calibration', 'source_overlap']:
        all_scores = [score for scores in metrics[metric_name].values() for score in scores]
        metrics[metric_name]['overall'] = sum(all_scores) / len(all_scores) if all_scores else 0
    
    # Average scores by domain (replace lists with floats)
    for metric_name in ['trace_scores', 'hallucination_scores', 'confidence_calibration', 'source_overlap']:
        for domain in list(metrics[metric_name].keys()):
            scores = metrics[metric_name][domain]
            if isinstance(scores, list):
                metrics[metric_name][domain] = sum(scores)/len(scores) if scores else 0
    
    # Add aggregate metrics
    metrics['aggregate_scores'] = {
        'mean_hallucination_reduction': metrics['hallucination_scores']['overall'],
        'mean_source_grounding': metrics['source_overlap']['overall'],
        'mean_confidence_calibration': metrics['confidence_calibration']['overall'],
        'p90_latency': metrics['latency']['p90']
    }
    
    return metrics

def recall_at_k_by_domain(y_true, y_pred, domains, k=5):
    """For single-answer QA, recall@k is same as precision@k."""
    return precision_at_k_by_domain(y_true, y_pred, domains, k)

def exact_match_by_domain(y_true, y_pred, domains):
    """
    Compute exact match score grouped by domain.
    y_true: list of strings (ground truth answers)
    y_pred: list of strings or list of lists of strings (predictions)
    domains: list of strings (domain labels)
    """
    domain_results = defaultdict(list)
    domain_metrics = {}
    
    for t, p, domain in zip(y_true, y_pred, domains):
        # If p is a list, take the first prediction
        if isinstance(p, list):
            p = p[0] if p else ""
            
        if is_not_found(p):
            domain_results[domain].append(0.0)
            continue
        
        # Normalize and compare
        t_norm = preprocess_text(t)
        p_norm = preprocess_text(p)
            
        if t_norm == p_norm or compute_trace_score(p, t) > 0.9:
            domain_results[domain].append(1.0)
        else:
            domain_results[domain].append(0.0)
    
    # Compute average for each domain
    for domain, scores in domain_results.items():
        domain_metrics[domain] = sum(scores) / len(scores) if scores else 0.0
    
    # Also compute overall
    all_scores = [score for scores in domain_results.values() for score in scores]
    domain_metrics['overall'] = sum(all_scores) / len(all_scores) if all_scores else 0.0
    
    return domain_metrics

def precision_at_k(y_true, y_pred, k=5):
    # y_pred: list of list of strings, y_true: list of strings
    correct = 0
    for true, preds in zip(y_true, y_pred):
        preds_k = preds[:k]
        # If all top-k are 'not found', count as incorrect
        if all(is_not_found(p) for p in preds_k):
            continue
        if any(compute_trace_score(p, true) > 0.5 and not is_not_found(p) for p in preds_k):
            correct += 1
    return correct / len(y_true) if y_true else 0.0

def recall_at_k(y_true, y_pred, k=5):
    # For single-answer QA, recall@k is same as precision@k
    return precision_at_k(y_true, y_pred, k)

def exact_match(y_true, y_pred):
    correct = 0
    for t, p in zip(y_true, y_pred):
        if is_not_found(p):
            continue
        t_norm = preprocess_text(t)
        p_norm = preprocess_text(p)
        if t_norm == p_norm or compute_trace_score(p, t) > 0.9:
            correct += 1
    return correct / len(y_true) if y_true else 0.0

def faithfulness(preds, refs):
    # Penalize 'not found' or irrelevant answers
    scores = []
    for p, r in zip(preds, refs):
        if is_not_found(p):
            scores.append(0.0)
        else:
            scores.append(compute_trace_score(p, r))
    return sum(scores) / len(scores) if scores else 0.0

def concept_f1(y_true, y_pred):
    # Extract key concepts and compute F1 score
    scores = []
    for t, p in zip(y_true, y_pred):
        if is_not_found(p):
            scores.append(0.0)
            continue
        
        # Extract noun phrases as proxy for concepts
        t_concepts = set(re.findall(r'\b[A-Z][a-z]*(?:\s+[A-Z][a-z]*)*\b', t))
        p_concepts = set(re.findall(r'\b[A-Z][a-z]*(?:\s+[A-Z][a-z]*)*\b', p))
        
        if not t_concepts or not p_concepts:
            scores.append(compute_trace_score(p, t))
            continue
            
        precision = len(t_concepts & p_concepts) / len(p_concepts) if p_concepts else 0
        recall = len(t_concepts & p_concepts) / len(t_concepts) if t_concepts else 0
        
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        scores.append(f1)
        
    return sum(scores) / len(scores) if scores else 0.0 