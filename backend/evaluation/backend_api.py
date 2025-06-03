import requests
import time
import logging
import re

logger = logging.getLogger("BackendAPI")

BACKEND_URL = "http://127.0.0.1:8001"
UPLOAD_ENDPOINT = f"{BACKEND_URL}/api/upload"
ANSWER_ENDPOINT = f"{BACKEND_URL}/api/answer"


def upload_document(file_content, filename, backend_url=BACKEND_URL, timeout=120):
    """
    Upload a document to the backend. Returns document_id on success.
    file_content: str or bytes
    filename: str
    backend_url: str
    timeout: int (seconds)
    """
    files = {"file": (filename, file_content)}
    logger.info(f"Uploading {filename} to {backend_url}/api/upload ...")
    
    try:
        response = requests.post(
            f"{backend_url}/api/upload",
            files=files,
            timeout=timeout
        )
        response.raise_for_status()
        document_id = response.json().get('document_id')
        logger.info(f"Uploaded {filename}, got document_id: {document_id}")
        return document_id
    except Exception as e:
        logger.error(f"Failed to upload {filename}: {e}")
        raise


def get_answer(document_id, question, backend_url=BACKEND_URL, temperature=0.1, max_tokens=128, hybrid_alpha=0.7):
    """
    Get answer from the backend for a question about a document.
    Returns both the answer text and confidence score if available.
    """
    logger.info(f"Querying answer for doc {document_id} ...")
    
    try:
        response = requests.post(
            f"{backend_url}/api/answer",
            json={
                "document_id": document_id,
                "question": question,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "hybrid_alpha": hybrid_alpha
            }
        )
        response.raise_for_status()
        result = response.json()
        answer = result.get('answer', '')
        
        # Try to extract confidence information
        confidence = 0.5  # Default confidence
        
        # Method 1: Check if confidence is directly provided in the response
        if 'confidence' in result:
            confidence = float(result['confidence'])
        
        # Method 2: Look for confidence patterns in the answer
        elif isinstance(answer, str):
            # Look for confidence expressed as percentage
            percentage_match = re.search(r'confidence:?\s*(\d+)%', answer.lower())
            if percentage_match:
                confidence = float(percentage_match.group(1)) / 100.0
            
            # Look for confidence expressed as decimal
            decimal_match = re.search(r'confidence:?\s*(0\.\d+)', answer.lower())
            if decimal_match:
                confidence = float(decimal_match.group(1))
            
            # Adjust confidence based on answer patterns
            if "not found" in answer.lower() or "i don't have" in answer.lower():
                confidence = min(confidence, 0.3)  # Reduce confidence for "not found" answers
            elif "[Document" in answer and "]" in answer:
                confidence = max(confidence, 0.6)  # Increase confidence for cited answers
        
        logger.info(f"Got answer: {answer[:50]}... (confidence: {confidence:.2f})")
        return answer, confidence
    except Exception as e:
        logger.error(f"Failed to get answer: {e}")
        raise 