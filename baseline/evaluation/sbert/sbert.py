import json
import nltk
import torch
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

nltk.download("punkt")

tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')

output_file = "evaluation/sbert/biomqm_sbert.jsonl"

with open(output_file, mode="a", encoding="utf-8") as out_f:

    predicted_file = "QA/vanilla_bt_qwen-3b.jsonl" 
    reference_file = "QA/vanilla_src_qwen-3b.jsonl"

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

                    scores = []
                    cosine_sim_list = []

                    for pred, ref in zip(predicted_answers, reference_answers):
                        """
                        if not isinstance(pred, str) or not isinstance(ref, str):
                            continue
                        if pred.strip() == "" or ref.strip() == "":
                            continue
                        """

                        encoded_pred = tokenizer(str(pred), padding=True, truncation=True, return_tensors='pt')
                        encoded_ref = tokenizer(str(ref), padding=True, truncation=True, return_tensors='pt')

                        with torch.no_grad():
                            pred_output = model(**encoded_pred)
                            ref_output = model(**encoded_ref)

                        pred_embed = mean_pooling(pred_output, encoded_pred['attention_mask'])
                        pred_embeds = F.normalize(pred_embed, p=2, dim=1)

                        ref_embed = mean_pooling(ref_output, encoded_ref['attention_mask'])
                        ref_embeds = F.normalize(ref_embed, p=2, dim=1)

                        cos_sim = F.cosine_similarity(pred_embeds, ref_embeds, dim=1).mean().item()
                        cosine_sim_list.append(cos_sim)
                        scores.append({"cos_sim": cos_sim})

                    
                    pred_data["scores"] = scores
                    if len(cosine_sim_list) == 0:
                        pred_data["avg_cos_similarity"] = None
                    else:
                        pred_data["avg_cos_similarity"] = sum(cosine_sim_list) / len(cosine_sim_list)
                
                    out_f.write(json.dumps(pred_data, ensure_ascii=False) + "\n")

                except json.JSONDecodeError as e:
                    print(f"Skipping a corrupted line due to JSONDecodeError: {e}")
                    continue

    except FileNotFoundError as e:
        print(f"File not found: {e}")
