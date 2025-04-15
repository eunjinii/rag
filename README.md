# Concise Knowledge-Guided RAG

Concise Knowledge-Guided Retrieval-Augmented Generation (RAG) enhances response generation by integrating a retrieval component with a knowledge-guided filtering approach. This ensures that generated responses are relevant, concise, and knowledge-driven.

### 1. Start Ollama Server and Pull Required Models
   ```bash
   # Start the Ollama server
   ollama serve

   # Pull required models (only needed once)
   ollama pull llama3.2:1b
   ollama pull gemma:7b
   ```
   
### 2. Run Evaluation on a Single GPU
   ```bash
   conda activate rag
   python evaluate.py --dataset MedMCQA --k 2 --chunk 512
   ```

### 3. Run Evaluation with Multi-GPU Parallelism
   ```bash
   conda activate rag
   python evaluate_parallel.py --dataset MedMCQA --k 2 --chunk 512
   ```