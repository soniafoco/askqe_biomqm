from allennlp.predictors.predictor import Predictor
import allennlp_models.pair_classification
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
    parser.add_argument("--input", type=str, required=True,
                        help="Input JSONL file with questions, answers (SRC and BT) and sbert score")
    parser.add_argument("--output", type=str, required=True,
                        help="Output JSONL file with entailed answers")
    return parser.parse_args()


def get_nli_label(predictor, question, pred, ref):

    premise = question + ' ' + ref + '.'
    hypothesis = question + ' ' + pred + '.'

    res = predictor.predict(
        premise=premise,
        hypothesis=hypothesis
    )

    print("PREMISE:", premise)
    print("HYPOTHESIS:", hypothesis)
    print("label:", res['label'])

    return res['label']


def main():
    args = parse_args()

    predictor = Predictor.from_path("https://storage.googleapis.com/allennlp-public-models/snli_roberta-2020.06.09.tar.gz",
    predictor_name="textual_entailment")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    try:
        with open(args.input, "r", encoding="utf-8") as f_in, open(args.output, "w", encoding="utf-8") as out_f:
            for line in f_in:
                try:
                    data = json.loads(line)

                    predicted_answers = data.get("answers_bt", [])
                    reference_answers = data.get("answers_src", [])
                    questions = data.get("questions", [])
                    scores = data.get("scores", [])

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

                    for pred, ref, question, score in zip(predicted_answers, reference_answers, questions, scores):
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
                            nli_label = get_nli_label(predictor, str(question), str(pred), str(ref))

                            if nli_label == 'entailment':  # If entails, the score is 1
                                nli_score = ENTAILMENT_SCORE
                            elif nli_label == 'contradiction':  # If contradicts, the score is 0
                                nli_score = CONTRADICTION_SCORE
                            elif nli_label == 'neutral':
                                nli_score = f1_score
                                
                        nli_scores.append(nli_score)
                            

                    data["nli_scores"] = nli_scores
                    if len(nli_scores) == 0:
                        data["avg_nli"] = None
                    else:
                        data["avg_nli"] = sum(nli_scores) / len(nli_scores)
                
                    out_f.write(json.dumps(data, ensure_ascii=False) + "\n")

                except json.JSONDecodeError as e:
                    print(f"Skipping a corrupted line due to JSONDecodeError: {e}")
                    continue

    except FileNotFoundError as e:
        print(f"File not found: {e}")


if __name__ == "__main__":
    main()