import os
import numpy as np
import pandas as pd
import torch
import faiss

from collections import defaultdict
from sentence_transformers import CrossEncoder
from transformers import AutoModel, AutoTokenizer
from rank_bm25 import BM25Okapi
from langchain.text_splitter import RecursiveCharacterTextSplitter

# import nltk
# nltk.download('punkt')
from nltk.tokenize import sent_tokenize
# _ = sent_tokenize(".") # force initialize punkt tokenizer

class MyRetriever:
    def __init__(
            self,
            raw_data_path:str="",
            retriever_type:str="contriever",
            chunk_size:int=None,
            granularity:str="chunk",
            # device:str="cuda",
        ):

        self.raw_data_path = raw_data_path
        self.retriever_type = retriever_type
        self.chunk_size = chunk_size
        self.granularity = granularity # "chunk", "sentence"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=50,
            length_function=len
        )
        self.texts = self._load_dataset()
        self.reranker = CrossEncoder("nboost/pt-biobert-base-msmarco")
        # self.reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        
        # Dense - Contriever, MedCPT, Specter, RRF-2
        if self.retriever_type in ["contriever", "specter", "longformer", "medcpt", "rrf2"]:
            if self.retriever_type == 'contriever':
                model_name = "facebook/contriever"
            elif self.retriever_type == 'specter':
                model_name = "allenai/specter"
            elif self.retriever_type == 'longformer':
                model_name = "allenai/longformer-base-4096"
            elif self.retriever_type == "medcpt":
                model_name = "ncbi/MedCPT-Article-Encoder"
            else:
                model_name = "facebook/contriever"

            self.model = AutoModel.from_pretrained(model_name).to(self.device)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            text_embedding = self._compute_embeddings(self.texts)
            text_embedding = text_embedding.cpu()
            self.faiss_index = self._create_dense_retriever(text_embedding)
            # print("FAISS index created.")

        # Sparse - BM25, TF-IDF
        self.bm25_index = self._get_bm25_index(self.texts)

    def _compute_embeddings(self, texts):
        def mean_pooling(token_embeddings, mask):
            token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.)
            sentence_embeddings = (token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]).detach()
            return sentence_embeddings
        
        all_embeddings = []
        with torch.no_grad():
            for i in range(0, len(texts), 32):
                batch_texts = texts[i:i+32]
                inputs = self.tokenizer(batch_texts, padding=True, truncation=True, return_tensors="pt", max_length=self.model.config.max_position_embeddings).to(self.device)
                outputs = self.model(**inputs)
                embeddings = mean_pooling(outputs[0], inputs['attention_mask']).cpu()
                embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
                all_embeddings.append(embeddings)
        return torch.cat(all_embeddings, dim=0)

    def _compute_query_embedding(self, query):
        return self._compute_embeddings([query]).numpy()
    
    def _create_dense_retriever(self, text_embedding):
        d = text_embedding.shape[1]
        faiss_index = faiss.IndexFlatIP(d) # [IndexFlatIP, IndexFlatL2]
        faiss_index.add(np.ascontiguousarray(text_embedding.numpy()))
        return faiss_index
    
    def _get_dense_results(self, query, k=5):
        query_embedding = self._compute_query_embedding(query)
        distances, indices = self.faiss_index.search(query_embedding, k)
        faiss_results = []
        for idx_list, dist_list in zip(indices, distances):
            for idx, dist in zip(idx_list, dist_list):
                faiss_results.append((self.texts[idx], dist))
        return faiss_results[:k]
    
    def _get_bm25_index(self, texts):
        tokenized_corpus = [text.lower().split() for text in texts]
        bm25_index = BM25Okapi(tokenized_corpus)
        return bm25_index
    
    def _get_bm25_results(self, query):
        query_tokens = query.lower().split()
        bm25_scores = self.bm25_index.get_scores(query_tokens)
        bm25_results = [(self.texts[i], bm25_scores[i]) for i in range(len(bm25_scores))]
        bm25_results = sorted(bm25_results, key=lambda x: x[1], reverse=True)
        return bm25_results
    
    def _load_dataset(self):
        texts = []
        for root, _, files in os.walk(self.raw_data_path):  # Recursively walk through directories
            for file_name in files:
                file_path = os.path.join(root, file_name)
                if file_name.endswith(".txt"):
                    with open(file_path, "r", encoding="utf-8") as f:
                        text = f.read().strip()
                        if self.chunk_size is not None:
                            texts.extend(self._split_text(text))
                        else:
                            texts.append(text)  # Add the entire text as a single element

                elif file_name.endswith(".csv"):
                    df = pd.read_csv(file_path)
                    if "summary" in df.columns:
                        texts.extend(df["summary"].dropna().tolist())

        if self.chunk_size is not None:
            concatenated_text = " ".join(texts)
            texts = self._split_text(concatenated_text)
        # print(f"Loaded {len(texts)} documents from directory: {self.raw_data_path}")
        return texts

    def _ensemble_scores(self, *results_with_weights):
        """
        Args: results_with_weights: Tuples of (results, weight) where results are lists of (text, score).
        Returns: List of (text, combined_score) tuples sorted by score in descending order.
        """
        combined_scores = defaultdict(float)

        for results, weight in results_with_weights:
            scores_dict = {text: score for text, score in results}
            values = np.array(list(scores_dict.values()))
            if len(values) > 0:
                min_val, max_val = values.min(), values.max()
                if max_val > min_val:
                    scores_dict = {k: (v - min_val) / (max_val - min_val) for k, v in scores_dict.items()}

            for text, score in scores_dict.items():
                combined_scores[text] += weight * score

        sorted_results = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_results
    
    def _split_text(self, texts):
        if self.granularity == "sentence":
            return sent_tokenize(texts)
        return self.text_splitter.split_text(texts)

    def search(self, query:str, k:int=5, rerank:bool=False):
        retrieved_docs = None
        if self.retriever_type in ["contriever", "longformer", "specter", "medcpt"]:
            retrieved_docs = self._get_dense_results(query, k=k)

        elif self.retriever_type == "bm25":
            retrieved_docs = self._get_bm25_results(query)[:k]
        
        elif self.retriever_type == "rrf2":
            faiss_results = self._get_dense_results(query, k=k)
            bm25_results = self._get_bm25_results(query)
            retrieved_docs = self._ensemble_scores((faiss_results, 0.5), (bm25_results, 0.5))[:k]
        else:
            retrieved_docs = self._get_dense_results(query, k=k)

        if rerank:
            retrieved_docs = self.rerank(query, retrieved_docs, top_n=2) # top_n = final value of k
        return retrieved_docs
    
    def rerank(self, question, docs, top_n=2):
        # all_chunks = []
        # for doc in docs:
        #     text = doc.page_content if hasattr(doc, 'page_content') else doc[0]
        #     if len(text) > self.chunk_size:
        #         chunks = self.text_splitter.split_text(text)
        #     else: 
        #         chunks = [text]
        #     all_chunks.extend(chunks)
        pairs = [(question, doc.page_content if hasattr(doc, 'page_content') else doc[0]) for doc in docs]
        # pairs = [(question, chunk) for chunk in all_chunks]
        scores = self.reranker.predict(pairs).tolist()
        scored_docs = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
        return [d[0] for d in scored_docs[:top_n]]