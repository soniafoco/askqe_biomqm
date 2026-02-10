import json
import os
import argparse
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description="NLI-based filtering of answers using entailment"
    )
    parser.add_argument("--sentence_type", type=str, required=True,
                        help="Either SRC or BT")
    parser.add_argument("--input", type=str, required=True,
                        help="Input JSONL file with answers")
    parser.add_argument("--output", type=str, required=True,
                        help="Output JSONL file with entailed answers")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Entailment probability threshold")
    return parser.parse_args()

def is_entailed_sentence(source, answer, tokenizer, model, device, threshold):
    inputs = tokenizer(
        source,
        answer,
        truncation=True,
        max_length=512,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        logits = model(**inputs).logits

    probs = F.softmax(logits, dim=-1)
    entailment_score = probs[0][2].item()  # entailment

    return entailment_score >= threshold


def wrap_answer(ans):
    return f"The text mentions {ans}."


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

    with open(args.input, "r", encoding="utf-8") as fin, \
         open(args.output, "w", encoding="utf-8") as fout:

        for line_num, line in enumerate(tqdm(fin), start=1):
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"JSON error on line {line_num}: {e}")
                continue

            if args.sentence_type == "src":
                answers = obj.get("answers", [])
            elif args.sentence_type == "bt_tgt":
                answers = obj.get("answers_bt", [])
            source = obj.get(args.sentence_type, []) #src or bt_tgt
            questions = obj.get("questions", [])

            if not source:
                continue

            # Ensure answers is a list
            if isinstance(answers, str):
                try:
                    answers = json.loads(answers)
                except Exception:
                    answers = []

            if not isinstance(answers, list):
                continue

            # Ensure questions is a list
            if isinstance(questions, str):
                try:
                    questions = json.loads(questions)
                except Exception:
                    questions = []

            if not isinstance(questions, list):
                continue

            entailed = []

            for i in range(len(answers)):
                try:
                    wrapped_answer = wrap_answer(answers[i])
                    if is_entailed_sentence(source,wrapped_answer,tokenizer,model,device,args.threshold):
                        entailed.append(answers[i])
                    else:
                        print("\n\n\nNOT ENTAILED: ")
                        print("ANSWER:", str(answers[i]))
                        print("SOURCE", source)
                        questions.pop(i)
                except Exception as e:
                    print("NLI error: ", str(answers[i])[:50], e)

            obj["questions"] = questions
            obj["answers"] = entailed

            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
            fout.flush()

            print(len(entailed), "/", len(answers), "entailed")

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

if __name__ == "__main__":
    main()