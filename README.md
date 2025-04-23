# Local Document Summarization and QA Pipeline with LLMs

This project implements a local document summarization and question-answering system using large-context language models. The system processes documents, summarizes segments, creates vector embeddings, and answers questions—all locally on your machine without external APIs.

## Features

- Process various document formats (PDF, DOCX, TXT)
- Split documents into logical chunks
- Generate abstractive summaries of each chunk
- Create vector embeddings for efficient retrieval
- Answer questions using retrieved context
- Completely local execution (no data leaves your machine)
- Interactive UI with Streamlit

## Requirements

- Python 3.10+
- Apple Silicon Mac (for optimal Metal acceleration) or equivalent hardware
- Minimum 16GB RAM (recommended: 32GB+ for larger models)
- Downloaded GGUF model file (recommend Yi-34B-200K, Mixtral 8x22B, or similar with 4-bit quantization)

## Setup

1. Clone this repository
2. Download a GGUF quantized model file from Hugging Face
3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
4. Update `models.py` with the path to your downloaded model file
5. Run the application:
   ```
   streamlit run main.py
   ```

## Pipeline Architecture

The system uses LangChain and LangGraph to implement a flexible document processing pipeline:

1. **Document Loading**: Parse various document formats
2. **Text Segmentation**: Split into manageable chunks
3. **Chunk Summarization**: Generate concise summaries
4. **Vector Store Creation**: Embed summaries for retrieval
5. **Question Answering**: Retrieve relevant chunks and generate answers

## Performance Considerations

- Model inference speed depends on your hardware and the model size
- For improved speed on Apple Silicon, ensure Metal support is enabled
- Adjust chunk size based on your model's optimal context window
- Consider adjusting thread count based on your CPU

## Acknowledgements

- LangChain & LangGraph for the workflow framework
- llama.cpp for efficient local LLM inference
- Sentence-Transformers for embeddings
- Hugging Face for model hosting