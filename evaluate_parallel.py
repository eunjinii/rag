import os
import torch
import time
import argparse
import pandas as pd
import multiprocessing as mp
import numpy as np
import ast 
import nltk
nltk.download('punkt_tab')

from tqdm import tqdm
from rouge import Rouge
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from bert_score import score
from transformers import AutoTokenizer
from langchain_ollama import ChatOllama
from retriever import MyRetriever
from dataset import QuestionAnswering
from prompt_utils import run_simple_prompt, run_rag_chain

# from nltk.tokenize import sent_tokenize
# _ = sent_tokenize(".") # force initialize punkt tokenizer

VISIBLE_GPUS = list(map(int, os.environ["CUDA_VISIBLE_DEVICES"].split(",")))

bert_tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")
rouge = Rouge()
smooth_fn = SmoothingFunction().method1


def truncate_and_decode(text, tokenizer, max_len=512):
    enc = tokenizer(text, truncation=True, max_length=max_len, return_tensors="pt", add_special_tokens=False)
    return tokenizer.decode(enc["input_ids"][0], skip_special_tokens=True)

def compute_f1_em(pred, gt):
    pred_tokens = pred.strip().split()
    gt_tokens = gt.strip().split()
    common = set(pred_tokens) & set(gt_tokens)
    if not common:
        return 0.0, 0.0
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(gt_tokens)
    f1 = 2 * precision * recall / (precision + recall)
    em = float(pred.strip() == gt.strip())
    return f1, em


def evaluate_partition(samples, retriever_type, dataset_name, tokenizer, k, chunk_size, granularity, is_rerank, return_list, gpu_id):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    torch.cuda.set_device(0) 
    
    slm_model = ChatOllama(model="llama3.2:1b", temperature=0)
    llm_model = ChatOllama(model="gemma:7b", temperature=0)

    if retriever_type not in ["llm_wo_knowledge", "slm_wo_knowledge"]:
        retriever_name = retriever_type.split("_")[-1]
        retriever_path = "random" if retriever_name == "random" else dataset_name.lower()
        retriever = MyRetriever(
            raw_data_path=f"data/text/{retriever_path}",
            retriever_type="contriever" if retriever_name == "random" else retriever_name,
            chunk_size=chunk_size,
            granularity=granularity,
        )
    else:
        retriever = None
    
    correct_answers = 0
    scores_list = []
    time_spent = []

    # options 컬럼 타입 보정 (문자열로 되어 있을 경우 파싱)
    if "options" in samples.columns:
        if samples["options"].apply(lambda x: isinstance(x, str)).all():
            samples["options"] = samples["options"].apply(ast.literal_eval)

    evaluation_mode = "multiple_choice" if "options" in samples.columns and samples["options"].apply(lambda x: isinstance(x, list) and len(x) > 0).all() else "sentence"
    print(f"Evaluation mode: {evaluation_mode}")

    for _, sample in (tqdm(samples.iterrows(), total=len(samples), desc=f"GPU{gpu_id}") if gpu_id == VISIBLE_GPUS[0] else samples.iterrows()):
        question = sample["question"].lower()
        options = list(map(str.lower, sample["options"])) if evaluation_mode == "multiple_choice" else None
        gt_answer = sample["answer"].lower()

        start_time = time.time()
        if retriever_type == "llm_wo_knowledge":
            pred_answer = run_simple_prompt(llm_model, question, options).lower()
        elif retriever_type == "llm_contriever":
            pred_answer = run_rag_chain(llm_model, retriever, question, options, k, is_rerank).lower()
        elif retriever_type == "slm_wo_knowledge":
            pred_answer = run_simple_prompt(slm_model, question, options).lower()
        else:
            pred_answer = run_rag_chain(slm_model, retriever, question, options, k, is_rerank).lower()
        
        end_time = time.time()
        time_spent.append(end_time - start_time)
        
        if evaluation_mode == "multiple_choice":
            if pred_answer in ['a', 'b', 'c', 'd']:
                correct_answers += int(pred_answer == gt_answer)
        else:
            # bleu = sentence_bleu([gt_answer.split()], pred_answer.split(), smoothing_function=smooth_fn)
            # rouge_l = rouge.get_scores(pred_answer, gt_answer)[0]['rouge-l']['f']
            # f1, em = compute_f1_em(pred_answer, gt_answer)
            truncated_pred = truncate_and_decode(pred_answer, tokenizer)
            truncated_gt = truncate_and_decode(gt_answer, tokenizer)
            try:
                _, _, bert_f1 = score([truncated_pred], [truncated_gt], lang='en', model_type='allenai/scibert_scivocab_uncased')
                bert_score_value = bert_f1.item()
            except RuntimeError:
                bert_score_value = 0.0

            scores_list.append({
                # "BLEU": bleu,
                # "ROUGE-L": rouge_l,
                "BERTScore": bert_score_value,
                # "F1": f1,
                # "EM": em,
            })

    return_list.append({
        "correct": correct_answers,
        "scores": scores_list,
        "mode": evaluation_mode,
        "times": time_spent
    })


def evaluate(qna_df, retriever_types, dataset_name, k=10, chunk_size=512, granularity="chunk", is_rerank=False):
    results = []
    total_samples = len(qna_df)
    num_gpus = len(VISIBLE_GPUS)
    qna_chunks = np.array_split(qna_df, num_gpus)

    for retriever_type in retriever_types:
        print(f"\n{retriever_type.upper()} Processing...")

        manager = mp.Manager()
        return_list = manager.list()
        jobs = []

        for i in range(num_gpus):
            p = mp.Process(target=evaluate_partition, args=(
                qna_chunks[i], retriever_type, dataset_name,
                bert_tokenizer, k, chunk_size, granularity, is_rerank, return_list, VISIBLE_GPUS[i]))
            p.start()
            jobs.append(p)

        for p in jobs:
            p.join()

        eval_mode = return_list[0]["mode"]
        all_times = sum([r["times"] for r in return_list], [])
        avg_time = sum(all_times) / len(all_times)

        if eval_mode == "multiple_choice":
            total_correct = sum([r["correct"] for r in return_list])
            acc = total_correct / total_samples * 100
            results.append({
                "Retriever": retriever_type.upper(),
                # "Rerank": is_rerank,
                "Granularity": granularity,
                "TopK (Chunk)": f"{k} ({chunk_size if granularity == 'chunk' else '-'})",
                "Accuracy (%)": f"{acc:.2f}",
                "Time (s)": f"{avg_time:.4f}",
            })
            print(f"{retriever_type}, {k}({chunk_size if granularity == 'chunk' else '-'}), Accuracy: {acc:.2f}, Time(s): {avg_time:.4f}")
        else:
            all_scores = sum([r["scores"] for r in return_list], [])
            # mean_bleu = sum(s["BLEU"] for s in all_scores) / total_samples
            # mean_rouge = sum(s["ROUGE-L"] for s in all_scores) / total_samples
            mean_bert = sum(s["BERTScore"] for s in all_scores) / total_samples
            # mean_f1 = sum(s["F1"] for s in all_scores) / total_samples
            # mean_em = sum(s["EM"] for s in all_scores) / total_samples

            results.append({
                "Retriever": retriever_type.upper(),
                # "Rerank": is_rerank,
                "Granularity": granularity,
                "TopK (Chunk)": f"{k} ({chunk_size if granularity == 'chunk' else '-'})",
                # "BLEU": f"{mean_bleu:.4f}",
                # "ROUGE-L": f"{mean_rouge:.4f}",
                "BERTScore": f"{mean_bert:.4f}",
                # "F1": f"{mean_f1:.4f}",
                # "EM": f"{mean_em:.4f}",
                "Time (s)": f"{avg_time:.4f}",
            })
            print(f"{retriever_type}, {k}({chunk_size if granularity == 'chunk' else '-'}), BERTScore: {mean_bert:.4f}, Time(s): {avg_time:.4f}")

    return pd.DataFrame(results)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    
    parser = argparse.ArgumentParser(description="Evaluating the model.")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--k", type=int, default=2)
    parser.add_argument("--chunk", type=int, default=512)
    parser.add_argument("--gran", type=str, default="chunk")
    parser.add_argument("--rerank", action="store_true")
    args = parser.parse_args()

    qna_df = QuestionAnswering(args.dataset).get_question_answering_dataframe()

#  ["llm_wo_knowledge", "llm_contriever", "slm_wo_knowledge", "slm_random", "slm_contriever", "slm_specter", "slm_longformer", "slm_medcpt", "slm_bm25", "slm_rrf2"]
    retriever_types = ["slm_random", "slm_contriever", "slm_specter"]
    results_df = evaluate(
        qna_df,
        retriever_types,
        args.dataset,
        args.k,
        args.chunk,
        args.gran,
        args.rerank
    )

    print(f"\nEvaluation Result:")
    print(results_df.to_markdown(index=False))
