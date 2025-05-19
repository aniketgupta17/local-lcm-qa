"""
llm_manager.py - Manages all LLM calls for the RAG system.

This class is responsible for:
- Handling LLM inference (summarization, answering, etc.)
- Managing model loading and device selection
- Supporting HuggingFace API, transformers, and llama.cpp
- Providing async and sync generation methods
"""
import os
import time
import logging
import torch
import numpy as np
from typing import Dict, Any, Optional, List, Union
from pathlib import Path
import threading
import asyncio
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from huggingface_hub import InferenceClient

logger = logging.getLogger(__name__)

class LLMManager:
    """
    Manages all LLM calls for the RAG system (summarization, answering, etc.).
    Supports HuggingFace API, transformers, and llama.cpp.
    """
    def __init__(
        self,
        model_path: str,
        model_type: str = "huggingface_api",
        device: Optional[torch.device] = None,
        context_size: int = 4096,
        cache_dir: str = "./cache",
        quantization_level: str = "q4_k_m",
        max_workers: int = 2,
        offload_kv: bool = False,
        batch_size: int = 1,
        streaming: bool = False
    ):
        """
        Initialize the LLM Manager.
        Args:
            model_path: Path or name of the model
            model_type: Type of model (huggingface_api, huggingface, llama.cpp)
            device: Torch device
            context_size: Context window size
            cache_dir: Directory for caching model outputs
            quantization_level: Quantization level for model optimization
            max_workers: Maximum number of worker threads
            offload_kv: Whether to offload KV cache to CPU
            batch_size: Batch size for inference
            streaming: Whether to enable streaming generation
        """
        self.model_type = model_type
        self.model_path = model_path
        if self.model_type == "huggingface_api":
            logger.info(f"Initializing huggingface_api model from {self.model_path}")
            self.api_client = InferenceClient(model=self.model_path, token=os.getenv("HF_API_TOKEN"))
            self.context_size = context_size
            self.device = device if device else torch.device("cpu")
            self.cache_dir = Path(cache_dir); self.cache_dir.mkdir(exist_ok=True, parents=True)
            self.model_lock = threading.RLock()
            self.executor = ThreadPoolExecutor(max_workers=max_workers)
            return
        self.model_path = model_path
        self.model_type = model_type
        self.device = device if device else self._get_default_device()
        self.context_size = context_size
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True, parents=True)
        self.quantization_level = quantization_level
        self.offload_kv = offload_kv
        self.batch_size = batch_size
        self.streaming = streaming
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.model_lock = threading.RLock()
        self.model = None
        self.tokenizer = None
        self._initialize_model()

    def _get_default_device(self) -> torch.device:
        """Determine the best available device."""
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")

    def _initialize_model(self) -> None:
        """Initialize and load the LLM with optimizations."""
        logger.info(f"Initializing {self.model_type} model from {self.model_path}")
        try:
            if self.model_type == "llama.cpp":
                self._initialize_llamacpp_model()
            elif self.model_type == "huggingface":
                self._initialize_hf_model()
            else:
                raise ValueError(f"Unsupported model type: {self.model_type}")
            logger.info("Model initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing model: {str(e)}")
            raise

    def _initialize_llamacpp_model(self) -> None:
        """Initialize llama.cpp model with optimizations."""
        try:
            from llama_cpp import Llama
            n_gpu_layers = -1 if str(self.device) == "cuda" else 0
            with self.model_lock:
                self.model = Llama(
                    model_path=self.model_path,
                    n_ctx=self.context_size,
                    n_batch=self.batch_size,
                    n_gpu_layers=n_gpu_layers,
                    seed=42,
                    use_mlock=True,
                    verbose=False
                )
            logger.info(f"Loaded llama.cpp model with context size {self.context_size}")
        except ImportError:
            logger.error("llama_cpp package not installed. Run: pip install llama-cpp-python")
            raise

    def _initialize_hf_model(self) -> None:
        """Initialize HuggingFace model with optimizations."""
        try:
            import transformers
            from transformers import AutoModelForCausalLM, AutoTokenizer
            with self.model_lock:
                load_kwargs = {
                    "device_map": "auto",
                    "torch_dtype": torch.float16,
                    "low_cpu_mem_usage": True,
                }
                if self.quantization_level:
                    if self.quantization_level == "int8":
                        load_kwargs["load_in_8bit"] = True
                    elif self.quantization_level == "int4":
                        load_kwargs["load_in_4bit"] = True
                        load_kwargs["bnb_4bit_compute_dtype"] = torch.float16
                        load_kwargs["bnb_4bit_quant_type"] = "nf4"
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_path,
                    use_fast=True
                )
                logger.info("Loading HuggingFace model...")
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_path,
                    **load_kwargs
                )
                if hasattr(self.model, "eval"):
                    self.model.eval()
                logger.info(f"Loaded HuggingFace model on {self.device}")
        except ImportError:
            logger.error("transformers package not installed. Run: pip install transformers")
            raise

    @lru_cache(maxsize=32)
    def _cached_generate(self, prompt: str, temperature: float, max_tokens: int) -> str:
        """Generate text with caching."""
        cache_key = f"{hash(prompt)}_{temperature}_{max_tokens}"
        cache_path = self.cache_dir / f"{cache_key}.txt"
        if cache_path.exists():
            try:
                with open(cache_path, "r") as f:
                    return f.read()
            except Exception as e:
                logger.warning(f"Error reading from cache: {e}")
        # Generate new response
        return self._generate_raw(prompt, temperature, max_tokens)

    def _generate_raw(self, prompt: str, temperature: float, max_tokens: int) -> str:
        """Generate text using the selected model backend."""
        if self.model_type == "llama.cpp":
            return self._generate_llamacpp(prompt, temperature, max_tokens)
        elif self.model_type == "huggingface":
            return self._generate_hf(prompt, temperature, max_tokens)
        elif self.model_type == "huggingface_api":
            return self._generate_hf_api(prompt, temperature, max_tokens)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

    def _generate_llamacpp(self, prompt: str, temperature: float, max_tokens: int) -> str:
        """Generate text using llama.cpp backend."""
        with self.model_lock:
            output = self.model(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                stop=["\n"],
                echo=False
            )
            return output["choices"][0]["text"]

    def _generate_hf(self, prompt: str, temperature: float, max_tokens: int) -> str:
        """Generate text using HuggingFace transformers backend."""
        with self.model_lock:
            input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
            output = self.model.generate(
                input_ids,
                max_length=max_tokens,
                temperature=temperature,
                do_sample=True
            )
            return self.tokenizer.decode(output[0], skip_special_tokens=True)

    def _generate_hf_api(self, prompt: str, temperature: float, max_tokens: int) -> str:
        """Generate text using HuggingFace Inference API, using summarization if supported."""
        summarization_models = ["facebook/bart-large-cnn", "t5-base", "t5-small", "t5-large"]
        if self.model_path in summarization_models:
            try:
                result = self.api_client.summarization(prompt)
                # Always extract the summary string
                if isinstance(result, list) and len(result) > 0 and "summary_text" in result[0]:
                    return result[0]["summary_text"]
                elif isinstance(result, dict) and "summary_text" in result:
                    return result["summary_text"]
                elif isinstance(result, dict) and "generated_text" in result:
                    return result["generated_text"]
                else:
                    logger.error(f"Unexpected summarization result format: {result}")
                    return str(result)
            except Exception as e:
                logger.error(f"Summarization API error: {e}")
                return f"[Summarization error: {e}]"
        # Fallback to text_generation for other models
        try:
            result = self.api_client.text_generation(prompt, max_new_tokens=max_tokens, temperature=temperature)
            return result
        except Exception as e:
            logger.error(f"Text generation API error: {e}")
            return f"[Text generation error: {e}]"

    def generate(self, prompt: str, temperature: float = 0.7, max_tokens: int = 512) -> str:
        """Public method to generate text (sync). Always use text generation endpoint for answering."""
        # Always use text generation for answering, even for summarization models
        if self.model_type == "huggingface_api":
            try:
                result = self.api_client.text_generation(prompt, max_new_tokens=max_tokens, temperature=temperature)
                return result
            except Exception as e:
                logger.error(f"Text generation API error: {e}")
                return f"[Text generation error: {e}]"
        return self._cached_generate(prompt, temperature, max_tokens)

    async def generate_async(self, prompt: str, temperature: float = 0.1, max_tokens: int = 300) -> str:
        """Async method to generate text."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self.executor,
            lambda: self.generate(prompt, temperature, max_tokens)
        )

    def summarize(self, text: str, max_tokens: int = 250) -> str:
        """Summarize a given text using the LLM. Handles empty, too-short, or too-long input gracefully."""
        # Token/length validation
        try:
            import tiktoken
            encoding = tiktoken.get_encoding("cl100k_base")
            num_tokens = len(encoding.encode(text))
        except ImportError:
            num_tokens = len(text.split())
        if not text or len(text.strip()) < 20 or num_tokens < 20:
            logger.warning("Skipping summarization: input text too short or empty.")
            return "[No summary: text too short or empty]"
        if num_tokens > 1024 or len(text) > 4000:
            logger.info("Truncating long chunk for summarization.")
            text = text[:4000]
            try:
                import tiktoken
                encoding = tiktoken.get_encoding("cl100k_base")
                tokens = encoding.encode(text)
                text = encoding.decode(tokens[:1024])
            except ImportError:
                text = ' '.join(text.split()[:1024])
        if self.model_type == "huggingface_api":
            summary = self._generate_hf_api(text, temperature=0.1, max_tokens=max_tokens)
            if summary and "error" not in summary.lower():
                return summary
            else:
                logger.error(f"Summarization failed or returned error: {summary}")
                return "[Summary unavailable due to API error]"
        prompt = (
            "Please provide a clear, concise summary of the following document section, "
            "focusing on the main ideas and key details. Write in complete sentences.\n\n"
            f"Section:\n{text}\n\nSummary:"
        )
        try:
            return self.generate(prompt, max_tokens=max_tokens)
        except Exception as e:
            logger.error(f"Summarization error: {e}")
            return "[Summary unavailable due to error]"

    def generate_combined_summary(self, texts: List[str], max_tokens: int = 300) -> str:
        """Generate a combined summary for a list of texts."""
        combined = "\n".join(texts)
        return self.summarize(combined, max_tokens=max_tokens)