# llm_manager.py - Optimized LLM inference manager
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

# Configure logging
logger = logging.getLogger(__name__)

class LLMManager:
    """
    Optimized LLM Manager for efficient inference with quantization and resource management
    """
    
    def __init__(
        self,
        model_path: str,
        model_type: str = "llama.cpp",
        device: Optional[torch.device] = None,
        context_size: int = 4096,
        cache_dir: str = "./cache",
        quantization_level: str = "q4_k_m",
        max_workers: int = 2,
        offload_kv: bool = False,  # KV cache offloading for large context
        batch_size: int = 1,
        streaming: bool = False
    ):
        """
        Initialize the LLM Manager.
        
        Args:
            model_path: Path to the model file
            model_type: Type of model (llama.cpp, huggingface)
            device: PyTorch device (cuda, cpu, mps)
            context_size: Context window size
            cache_dir: Directory for caching model outputs
            quantization_level: Quantization level for model optimization
            max_workers: Maximum number of worker threads
            offload_kv: Whether to offload KV cache to CPU (for large contexts)
            batch_size: Batch size for inference
            streaming: Whether to enable streaming generation
        """
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
        
        # Resource management
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.model_lock = threading.RLock()
        
        # Model caching
        self.model = None
        self.tokenizer = None
        
        # Initialize the model
        self._initialize_model()
        
    def _get_default_device(self) -> torch.device:
        """Determine the best available device"""
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    
    def _initialize_model(self) -> None:
        """Initialize and load the LLM with optimizations"""
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
        """Initialize llama.cpp model with optimizations"""
        try:
            from llama_cpp import Llama
            
            # Set appropriate parameters based on device
            if str(self.device) == "cuda":
                n_gpu_layers = -1  # Use all layers on GPU
            else:
                n_gpu_layers = 0  # CPU only
            
            with self.model_lock:
                self.model = Llama(
                    model_path=self.model_path,
                    n_ctx=self.context_size,
                    n_batch=self.batch_size,
                    n_gpu_layers=n_gpu_layers,
                    seed=42,
                    use_mlock=True,  # Memory lock for faster inference
                    verbose=False
                )
                
            logger.info(f"Loaded llama.cpp model with context size {self.context_size}")
        except ImportError:
            logger.error("llama_cpp package not installed. Run: pip install llama-cpp-python")
            raise
    
    def _initialize_hf_model(self) -> None:
        """Initialize HuggingFace model with optimizations"""
        try:
            import transformers
            from transformers import AutoModelForCausalLM, AutoTokenizer
            
            with self.model_lock:
                # Configure model loading
                load_kwargs = {
                    "device_map": "auto",
                    "torch_dtype": torch.float16,  # Use half precision
                    "low_cpu_mem_usage": True,
                }
                
                # Add quantization parameters if specified
                if self.quantization_level:
                    if self.quantization_level == "int8":
                        load_kwargs["load_in_8bit"] = True
                    elif self.quantization_level == "int4":
                        load_kwargs["load_in_4bit"] = True
                        load_kwargs["bnb_4bit_compute_dtype"] = torch.float16
                        load_kwargs["bnb_4bit_quant_type"] = "nf4"
                
                # Load tokenizer
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_path,
                    use_fast=True
                )
                
                # Load model
                logger.info("Loading HuggingFace model...")
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_path,
                    **load_kwargs
                )
                
                # Optimize for inference
                if hasattr(self.model, "eval"):
                    self.model.eval()
                
                logger.info(f"Loaded HuggingFace model on {self.device}")
        except ImportError:
            logger.error("transformers package not installed. Run: pip install transformers")
            raise
    
    @lru_cache(maxsize=32)
    def _cached_generate(self, prompt: str, temperature: float, max_tokens: int) -> str:
        """Generate text with caching"""
        # Hash the prompt and parameters to create a cache key
        cache_key = f"{hash(prompt)}_{temperature}_{max_tokens}"
        cache_path = self.cache_dir / f"{cache_key}.txt"
        
        # Check cache first
        if cache_path.exists():
            try:
                with open(cache_path, "r") as f:
                    return f.read()
            except Exception as e:
                logger.warning(f"Error reading from cache: {e}")
        
        # Generate new response
        response = self._generate_raw(prompt, temperature, max_tokens)
        
        # Save to cache
        try:
            with open(cache_path, "w") as f:
                f.write(response)
        except Exception as e:
            logger.warning(f"Error saving to cache: {e}")
        
        return response
    
    def _generate_raw(self, prompt: str, temperature: float, max_tokens: int) -> str:
        """Raw generation without caching"""
        with self.model_lock:
            if self.model_type == "llama.cpp":
                return self._generate_llamacpp(prompt, temperature, max_tokens)
            elif self.model_type == "huggingface":
                return self._generate_hf(prompt, temperature, max_tokens)
            else:
                raise ValueError(f"Unsupported model type: {self.model_type}")
    
    def _generate_llamacpp(self, prompt: str, temperature: float, max_tokens: int) -> str:
        """Generate text using llama.cpp model"""
        try:
            output = self.model(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                echo=False,
                stop=["</s>", "<|im_end|>"]  # Common stop tokens
            )
            
            if isinstance(output, dict):
                return output.get("choices", [{}])[0].get("text", "").strip()
            elif isinstance(output, list):
                return output[0].get("text", "").strip()
            else:
                return str(output).strip()
                
        except Exception as e:
            logger.error(f"Error generating with llama.cpp: {e}")
            raise
    
    def _generate_hf(self, prompt: str, temperature: float, max_tokens: int) -> str:
        """Generate text using HuggingFace model"""
        try:
            # Tokenize input
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            
            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    max_new_tokens=max_tokens,
                    temperature=temperature if temperature > 0 else 1.0,
                    do_sample=temperature > 0,
                    pad_token_id=self.tokenizer.eos_token_id,
                    top_p=0.95,
                    repetition_penalty=1.1
                )
            
            # Decode
            result = self.tokenizer.decode(
                outputs[0][inputs.input_ids.shape[1]:], 
                skip_special_tokens=True
            )
            
            return result.strip()
        except Exception as e:
            logger.error(f"Error generating with HuggingFace: {e}")
            raise
    
    def generate(self, prompt: str, temperature: float = 0.1, max_tokens: int = 300) -> str:
        """
        Generate text using the LLM
        
        Args:
            prompt: Input prompt
            temperature: Temperature parameter for generation
            max_tokens: Maximum number of tokens to generate
            
        Returns:
            Generated text
        """
        start_time = time.time()
        
        try:
            # Use cached generation if possible
            result = self._cached_generate(prompt, temperature, max_tokens)
            
            logger.info(f"Generated {len(result.split())} words in {time.time() - start_time:.2f}s")
            return result
            
        except Exception as e:
            logger.error(f"Error in generate: {e}")
            return f"Error generating text: {str(e)}"
    
    async def generate_async(self, prompt: str, temperature: float = 0.1, max_tokens: int = 300) -> str:
        """Async version of generate"""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self.executor,
            lambda: self.generate(prompt, temperature, max_tokens)
        )
    
    def summarize(self, text: str, max_tokens: int = 200) -> str:
        """
        Generate a summary of the given text
        
        Args:
            text: Text to summarize
            max_tokens: Maximum length of summary
            
        Returns:
            Generated summary
        """
        if not text:
            return ""
        
        # Truncate text if too long
        max_chars = self.context_size * 2  # Rough approximation
        if len(text) > max_chars:
            text = text[:max_chars] + "..."
        
        prompt = f"""Please provide a concise summary of the following text:

{text}

Summary:"""
        
        return self.generate(prompt, temperature=0.1, max_tokens=max_tokens)
    
    def generate_combined_summary(self, texts: List[str], max_tokens: int = 300) -> str:
        """
        Generate a combined summary from multiple text fragments
        
        Args:
            texts: List of text fragments to summarize
            max_tokens: Maximum length of the combined summary
            
        Returns:
            Combined summary
        """
        if not texts:
            return ""
        
        # Join texts with separators
        combined = "\n\n---\n\n".join(
            # Truncate each text if too long
            [t[:2000] + "..." if len(t) > 2000 else t for t in texts]
        )
        
        # Further truncate if still too long
        max_chars = self.context_size * 2  # Rough approximation
        if len(combined) > max_chars:
            combined = combined[:max_chars] + "..."
        
        prompt = f"""Please provide a concise summary that integrates information from these text fragments:

{combined}

Combined summary:"""
        
        return self.generate(prompt, temperature=0.1, max_tokens=max_tokens)