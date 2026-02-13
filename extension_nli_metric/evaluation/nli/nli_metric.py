import json
import os
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F

ENTAILMENT_SCORE = 1
CONTRADICTION_SCORE = 0
NEUTRAL_SCORE = 0.5

def parse_args():
    parser = argparse.ArgumentParser(
        description="NLI-based scoring of answers using entailment"
    )
    parser.add_argument("--input_f1", type=str, required=True,
                        help="Input JSONL file with questions, answers (SRC and BT) and score")
    parser.add_argument("--input_sbert", type=str, required=True,
                        help="Input JSONL file with questions, answers (SRC and BT) and sbert score")
    parser.add_argument("--output", type=str, required=True,
                        help="Output JSONL file with entailed answers")
    return parser.parse_args()


def get_nli_label(model, tokenizer, device, question, pred, ref):

    premise = question + ' ' + ref + '.'
    hypothesis = question + ' ' + pred + '.'

    inputs = tokenizer(
        premise,
        hypothesis,
        truncation=True,
        max_length=512,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        logits = model(**inputs).logits

    probs = F.softmax(logits, dim=-1)

    entailment_score = probs[0][2].item()  # entailment
    contradiction_score = probs[0][0].item()  # contradiction
    neutral_score = probs[0][1].item()  # neutral

    if entailment_score > contradiction_score and entailment_score > neutral_score:
        label = 'entailment'
    elif contradiction_score > entailment_score and contradiction_score > neutral_score:
        label = 'contradiction'
    else:
        label = 'neutral'

    print("PREMISE:", premise)
    print("HYPOTHESIS:", hypothesis)
    print("label:", label)

    return label


def main():
    args = parse_args()

    nli_model_name = "roberta-large-mnli"
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(nli_model_name)
    model = AutoModelForSequenceClassification.from_pretrained(nli_model_name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    print("NLI model loaded on", device)

    try:
        with open(args.input_f1, "r", encoding="utf-8") as f_in, open(args.input_sbert, "r", encoding="utf-8") as f_sbert, open(args.output, "w", encoding="utf-8") as out_f:
            for line, line_sbert in zip(f_in, f_sbert):
                try:
                    data = json.loads(line)
                    data_sbert = json.loads(line_sbert)

                    predicted_answers = data.get("answers_bt", [])
                    reference_answers = data.get("answers_src", [])
                    questions = data.get("questions", [])
                    scores = data.get("scores", [])
                    scores_sbert = data_sbert.get("scores", [])

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

                    if isinstance(questions, str):
                        try:
                            questions = json.loads(questions)
                        except json.JSONDecodeError:
                            continue

                    if not isinstance(predicted_answers, list) or not isinstance(reference_answers, list) or not isinstance(questions, list):
                        continue
                    if not predicted_answers or not reference_answers or len(predicted_answers) != len(reference_answers) != len(questions):
                        continue

                    nli_scores = []
                    scores_new = []

                    for pred, ref, question, score, score_sbert in zip(predicted_answers, reference_answers, questions, scores, scores_sbert):
                        """
                        if not isinstance(pred, str) or not isinstance(ref, str):
                            continue
                        if pred.strip() == "" or ref.strip() == "":
                            continue
                        """
                        f1_score = score.get("f1", None)
                        if f1_score is None:
                            continue
                    
                        nli_score = f1_score

                        # If there is no answer - can't run NLI, keep the score 0
                        if str(pred).lower() != "no answer" and str(ref).lower() == "no answer":
                            nli_score = 0

                        # If the score is 1, there is a full overlap between the
                        # candidate and the predicted answer, so the score is 1
                        elif 0 <= f1_score < 1:
                            nli_label = get_nli_label(model, tokenizer, device, str(question), str(pred), str(ref))

                            if nli_label == 'entailment':  # If entails, the score is 1
                                nli_score = ENTAILMENT_SCORE
                            elif nli_label == 'contradiction':  # If contradicts, the score is 0
                                nli_score = CONTRADICTION_SCORE
                            elif nli_label == 'neutral':
                                nli_score = f1_score
                                
                        score["nli"] = nli_score
                        score["cos_sim"] = score_sbert["cos_sim"]
                        scores_new.append(score)
                        nli_scores.append(nli_score)
                        
                    data["scores"] = scores_new
                    if len(nli_scores) == 0:
                        data["avg_nli"] = None
                    else:
                        data["avg_nli"] = sum(nli_scores) / len(nli_scores)
                    data["avg_cos_similarity"] = data_sbert["avg_cos_similarity"]
                
                    out_f.write(json.dumps(data, ensure_ascii=False) + "\n")

                except json.JSONDecodeError as e:
                    print(f"Skipping a corrupted line due to JSONDecodeError: {e}")
                    continue

    except FileNotFoundError as e:
        print(f"File not found: {e}")


if __name__ == "__main__":
    main()