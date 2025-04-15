# Concise Knowledge-Guided RAG

## Overview
Concise Knowledge-Guided Retrieval-Augmented Generation (cRAG) enhances response generation by integrating a retrieval component with a knowledge-guided filtering approach. This ensures that generated responses are relevant, concise, and knowledge-driven.

## How to Run

   Run Ollama server and pull models using:
   ```bash
   $ ollama serve
   $ ollama pull llama3.2:1b
   $ ollama pull gemma:7b
   ```
   Run evaluation
   ```bash
   $ conda activate rag
   $ python evaluate.py --dataset_name MMLU --k 2 --chunk_size 512
   ```
   or for multi-GPU inference,
   ```bash
   $ conda activate rag
   $ python evaluate_parallel.py --dataset_name MMLU --k 2 --chunk_size 512
   ```