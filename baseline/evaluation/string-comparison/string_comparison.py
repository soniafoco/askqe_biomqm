import json
import nltk
from utils import compare_answers
import os 

nltk.download("punkt")

predicted_file = "QA/vanilla_bt_qwen-3b.jsonl" 
reference_file = "QA/vanilla_src_qwen-3b.jsonl"

results_list = []
try:
    with open(predicted_file, "r", encoding="utf-8") as pred_file, open(reference_file, "r", encoding="utf-8") as ref_file:
        for pred_line, ref_line in zip(pred_file, ref_file):
            try:
                pred_data = json.loads(pred_line)
                ref_data = json.loads(ref_line)

                predicted_answers = pred_data.get("answers_bt", [])
                reference_answers = ref_data.get("answers", [])
                pred_data["answers_src"] = reference_answers
                pred_data["answers_bt"] = predicted_answers

                if isinstance(predicted_answers, str):
                    try:
                        predicted_answers = json.loads(predicted_answers)
                    except json.JSONDecodeError:
                        continue

                if isinstance(reference_answers, str):
                    try:
                        reference_answers = json.loads(reference_answers)
                    except json.JSONDecodeError:
                        continue

                if not isinstance(predicted_answers, list) or not isinstance(reference_answers, list):
                    continue
                if not predicted_answers or not reference_answers or len(predicted_answers) != len(reference_answers):
                    continue

                row_scores = []
                for pred, ref in zip(predicted_answers, reference_answers):
                    print(pred)
                    print(ref)
                    f1, EM, chrf, bleu = compare_answers(str(pred), str(ref))
                    row_scores.append({
                        "f1": f1,
                        "em": EM,
                        "chrf": chrf,
                        "bleu": bleu
                    })

                # Save per-row result
                pred_data["scores"] = row_scores

                f1_scores = [x["f1"]   for x in row_scores]
                em_scores = [1 if x["em"] is True else 0 for x in row_scores]
                chrf_scores = [x["chrf"] for x in row_scores]
                bleu_scores = [x["bleu"] for x in row_scores]

                pred_data["avg_f1"] = sum(f1_scores) / len(f1_scores)
                pred_data["avg_em"] = sum(em_scores) / len(em_scores)
                pred_data["avg_chrf"] = sum(chrf_scores) / len(chrf_scores)
                pred_data["avg_bleu"] = sum(bleu_scores) / len(bleu_scores)

                results_list.append(pred_data)

            except json.JSONDecodeError as e:
                print(f"Skipping a corrupted line due to JSONDecodeError: {e}")
                continue

except FileNotFoundError as e:
    print(f"File not found: {e}")

os.makedirs("evaluation/string-comparison", exist_ok=True)

jsonl_output_file = "evaluation/string-comparison/biomqm_f1.jsonl"
with open(jsonl_output_file, "w", encoding="utf-8") as jsonl_file:
    for row in results_list:
        jsonl_file.write(json.dumps(row, ensure_ascii=False) + "\n")