import json
import os
import argparse
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description="NLI-based filtering of answers using sliding window entailment"
    )
    parser.add_argument("--sentence_type", type=str, required=True,
                        help="Either SRC or BT")
    parser.add_argument("--input", type=str, required=True,
                        help="Input JSONL file with answers")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Entailment probability threshold")
    parser.add_argument("--window_tokens", type=int, default=350,
                        help="Sliding window size in tokens")
    parser.add_argument("--stride", type=int, default=200,
                        help="Stride size in tokens")
    return parser.parse_args()


def is_entailed_sliding(source,answer,tokenizer,model,device,threshold,window_tokens,stride):
    """
    Returns True if answer is entailed by ANY chunk of source.
    """
    source_tokens = tokenizer(
        source,
        truncation=False,
        return_tensors=None
    )["input_ids"]

    max_start = max(len(source_tokens) - window_tokens + 1, 1)

    for start in range(0, max_start, stride):
        chunk_tokens = source_tokens[start:start + window_tokens]

        if len(chunk_tokens) < 50:
            continue

        chunk_text = tokenizer.decode(chunk_tokens, skip_special_tokens=True)

        inputs = tokenizer(
            chunk_text,
            answer,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            logits = model(**inputs).logits

        probs = F.softmax(logits, dim=-1)
        entailment_score = probs[0][2].item()  # index 2 = entailment

        if entailment_score >= threshold:
            return True

    return False


def main():
    args = parse_args()

    nli_model_name = "roberta-large-mnli"

    tokenizer = AutoTokenizer.from_pretrained(nli_model_name)
    model = AutoModelForSequenceClassification.from_pretrained(nli_model_name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    print("NLI model loaded on", device)

    with open(args.input, "r", encoding="utf-8") as fin, \
         open(f"{args.input}_entailed", "w", encoding="utf-8") as fout:

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

            entailed = []

            for i in range(len(answers)):
                try:
                    if is_entailed_sliding(source,answers[i],tokenizer,model,device,args.threshold,args.window_tokens,args.stride):
                        entailed.append(answers[i])
                    else:
                        questions.pop(i)
                except Exception as e:
                    print("NLI error on", answers[i][:50], e)

            obj["questions"] = questions
            obj["answers"] = entailed

            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
            fout.flush()

            print(len(entailed), "/", len(answers), "entailed")

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

if __name__ == "__main__":
    main()