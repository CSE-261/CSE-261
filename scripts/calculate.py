import json
import re
import string
from collections import Counter

def normalize_answer(s: str) -> str:
    """Normalize text: convert to lowercase, remove punctuation, articles, and extra whitespace."""
    if not isinstance(s, str):
        return ""
    
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def exact_match_score(prediction: str, ground_truth: str) -> float:
    return 1.0 if normalize_answer(prediction) == normalize_answer(ground_truth) else 0.0

def contains_match_score(prediction: str, ground_truth: str) -> float:
    pred_norm = normalize_answer(prediction)
    gt_norm = normalize_answer(ground_truth)
    if not gt_norm:
        return 0.0
    return 1.0 if gt_norm in pred_norm else 0.0

def f1_score(prediction: str, ground_truth: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    gt_tokens = normalize_answer(ground_truth).split()
    
    common = Counter(pred_tokens) & Counter(gt_tokens)
    num_same = sum(common.values())
    
    if num_same == 0:
        return 0.0
        
    precision = 1.0 * num_same / len(pred_tokens)
    recall = 1.0 * num_same / len(gt_tokens)
    return (2 * precision * recall) / (precision + recall)

def process_and_save_metrics(input_file: str, pred_key: str, ref_key: str, output_item_file: str, output_avg_file: str):
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    item_results = []
    em_scores = []
    cm_scores = []
    f1_scores = []
    
    # Store existing advanced metrics if they are available in the data
    faithfulness_scores = []
    context_recall_scores = []
    answer_similarity_scores = []

    for item in data:
        prediction = item.get(pred_key, '')
        reference = item.get(ref_key, '')
        
        em = exact_match_score(prediction, reference)
        cm = contains_match_score(prediction, reference)
        f1 = f1_score(prediction, reference)
        
        em_scores.append(em)
        cm_scores.append(cm)
        f1_scores.append(f1)
        
        # Add the computed metrics to the individual item record
        new_item = item.copy()
        new_item['exact_match'] = em
        new_item['contain_match'] = cm
        new_item['calculated_f1_score'] = f1 # Named explicitly to avoid collision with existing f1_score if present
        
        item_results.append(new_item)
        
        # Collect existing advanced metrics to average later
        if item.get('faithfulness') is not None:
            faithfulness_scores.append(item['faithfulness'])
        if item.get('context_recall') is not None:
            context_recall_scores.append(item['context_recall'])
        if item.get('answer_similarity') is not None:
            answer_similarity_scores.append(item['answer_similarity'])

    total_samples = len(data)
    avg_em = sum(em_scores) / total_samples if total_samples > 0 else 0.0
    avg_cm = sum(cm_scores) / total_samples if total_samples > 0 else 0.0
    avg_f1 = sum(f1_scores) / total_samples if total_samples > 0 else 0.0

    averages = {
        "total_samples": total_samples,
        "exact_match": avg_em,
        "contain_match": avg_cm,
        "f1_score": avg_f1
    }
    
    # Add optional RAGAS averages if we collected any
    if faithfulness_scores:
        averages["faithfulness"] = sum(faithfulness_scores) / len(faithfulness_scores)
    if context_recall_scores:
        averages["context_recall"] = sum(context_recall_scores) / len(context_recall_scores)
    if answer_similarity_scores:
        averages["answer_similarity"] = sum(answer_similarity_scores) / len(answer_similarity_scores)

    # Save Item-Level Metrics
    with open(output_item_file, 'w', encoding='utf-8') as f:
        json.dump(item_results, f, indent=4, ensure_ascii=False)
        
    # Save Average Metrics
    with open(output_avg_file, 'w', encoding='utf-8') as f:
        json.dump(averages, f, indent=4, ensure_ascii=False)
        
    print(f"Processed {input_file}:")
    print(f"  -> Saved item metrics to {output_item_file}")
    print(f"  -> Saved averages to {output_avg_file}")

# Process both files
process_and_save_metrics(
    input_file="baseline_strict_middle_results.json",
    pred_key="model_prediction",
    ref_key="answer",
    output_item_file="baseline_strict_middle_item_metrics.json",
    output_avg_file="baseline_strict_middle_average_metrics.json"
)

process_and_save_metrics(
    input_file="eval_details.json",
    pred_key="response",
    ref_key="reference",
    output_item_file="eval_details_item_metrics.json",
    output_avg_file="eval_details_average_metrics.json"
)