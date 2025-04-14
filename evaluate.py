import os
import time
import argparse
import pandas as pd

from tqdm import tqdm
from rouge import Rouge
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from bert_score import score
from transformers import AutoTokenizer
from langchain_ollama import ChatOllama

from retriever import MyRetriever
from dataset import QuestionAnswering
from prompt_utils import run_simple_prompt, run_rag_chain


bert_tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")
rouge = Rouge()
smooth_fn = SmoothingFunction().method1


def truncate_and_decode(text, tokenizer, max_len=512):
    enc = tokenizer(text, truncation=True, max_length=max_len, return_tensors="pt", add_special_tokens=False)
    return tokenizer.decode(enc["input_ids"][0], skip_special_tokens=True)

def evaluate(qna_df, retriever_types, slm_model, llm_model, dataset_name, k=10, chunk_size=512, is_rerank=False):
    results = []
    total_samples = len(qna_df)

    for retriever_type in retriever_types:
        print(f"\n{retriever_type.upper()} :")
        
        if retriever_type not in ["llm_wo_knowledge", "slm_wo_knowledge"]:
            retriever_name = retriever_type.split("_")[-1]
            retriever_path = "random" if retriever_name == "random" else dataset_name.lower()
            retriever = MyRetriever(
                raw_data_path=f"data/text/{retriever_path}",
                retriever_type="contriever" if retriever_name == "random" else retriever_name,
                chunk_size=chunk_size
            )

        if "options" in qna_df.columns and qna_df["options"].apply(lambda x: isinstance(x, list) and len(x) > 0).all():
            evaluation_mode = "multiple_choice"
            print("Running evaluation for multiple choice mode...")
        else:
            evaluation_mode = "sentence"
            print("Running evaluation for sentence mode...")

        correct_answers = 0
        scores_list = []
        start_time = time.time()

        for _, sample in tqdm(qna_df.iterrows(), total=len(qna_df)):
            question = sample["question"].lower()
            options = list(map(str.lower, sample["options"])) if evaluation_mode == "multiple_choice" else None
            gt_answer = sample["answer"].lower()

            if retriever_type == "llm_wo_knowledge":
                pred_answer = run_simple_prompt(llm_model, question, options).lower()
            elif retriever_type == "slm_wo_knowledge":
                pred_answer = run_simple_prompt(slm_model, question, options).lower()
            else:
                pred_answer = run_rag_chain(slm_model, retriever, question, options, k, is_rerank).lower()
            
            if evaluation_mode == "multiple_choice":
                if pred_answer not in ['a', 'b', 'c', 'd']:
                    print("Invalid prediction:", pred_answer)
                correct_answers += int(pred_answer == gt_answer)

            else:
                bleu = sentence_bleu([gt_answer.split()], pred_answer.split(), smoothing_function=smooth_fn)
                rouge_l = rouge.get_scores(pred_answer, gt_answer)[0]['rouge-l']['f']
                truncated_pred = truncate_and_decode(pred_answer, bert_tokenizer)
                truncated_gt = truncate_and_decode(gt_answer, bert_tokenizer)
                try:
                    _, _, bert_f1 = score([truncated_pred], [truncated_gt], lang='en', model_type='allenai/scibert_scivocab_uncased')
                    bert_score_value = bert_f1.item()
                except RuntimeError:
                    bert_score_value = 0.0

                scores_list.append({
                    "BLEU": bleu,
                    "ROUGE-L": rouge_l,
                    "BERTScore": bert_score_value
                })

        end_time = time.time()
        processing_time = end_time - start_time

        if evaluation_mode == "multiple_choice":
            acc = correct_answers / total_samples * 100
            results.append({
                "Retriever": retriever_type.upper(),
                "K / Chunk Size": f"{k} / {chunk_size}",
                "Accuracy (%)": f"{acc:.2f}",
                "Processing Time (s)": f"{processing_time:.2f}",
            })
            print(f"{retriever_type}, {k} / {chunk_size}, {acc:.2f}, {processing_time:.2f}")
        else:
            mean_bleu = sum(s["BLEU"] for s in scores_list) / total_samples
            mean_rouge = sum(s["ROUGE-L"] for s in scores_list) / total_samples
            mean_bert = sum(s["BERTScore"] for s in scores_list) / total_samples
            results.append({
                "Retriever": retriever_type.upper(),
                "Rerank": is_rerank,
                "K / Chunk Size": f"{k} / {chunk_size}",
                "BLEU": f"{mean_bleu:.4f}",
                "ROUGE-L": f"{mean_rouge:.4f}",
                "BERTScore": f"{mean_bert:.4f}",
                "Processing Time (s)": f"{processing_time:.2f}",
            })
            print(f"{retriever_type}, BLEU: {mean_bleu:.4f}, ROUGE-L: {mean_rouge:.4f}, BERTScore: {mean_bert:.4f}, Time: {processing_time:.2f}")

    return pd.DataFrame(results)

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"]="5"
    
    parser = argparse.ArgumentParser(description="Evaluating the model.")
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--k", type=int, default=2)
    parser.add_argument("--chunk_size", type=int, default=256)
    parser.add_argument("--rerank", type=bool, default=False)
    args = parser.parse_args()

    slm_model = ChatOllama(model="llama3.2:1b", temperature=0)
    llm_model = ChatOllama(model="gemma:7b", temperature=0) # $ ollama pull gemma:7b to run a new model
    qna_df = QuestionAnswering(args.dataset_name).get_question_answering_dataframe()
    
#  ["llm_wo_knowledge", "slm_wo_knowledge", "slm_random", "slm_contriever", "slm_specter", "slm_longformer", "slm_medcpt", "slm_bm25", "slm_rrf2"]
    retriever_types = ["slm_specter"]
    results_df = evaluate(
        qna_df,
        retriever_types,
        slm_model,
        llm_model,
        args.dataset_name,
        args.k,
        args.chunk_size,
    )
    print(f"\nEvaluation Result:")
    print(results_df.to_markdown(index=False))
