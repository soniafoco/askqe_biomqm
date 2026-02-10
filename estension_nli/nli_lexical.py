import json
import os
import argparse
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm
import re


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

def normalize(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def lexical_grounded(answer, source, min_overlap=1):
    """
    Returns True if answer shares at least `min_overlap`
    content tokens with the source.
    """
    ans_tokens = set(normalize(answer).split())
    src_tokens = set(normalize(source).split())

    stopwords = {"the", "a", "an", "of", "to", "and", "is", "are"}
    ans_tokens = ans_tokens - stopwords

    overlap = ans_tokens & src_tokens
    return len(overlap) >= min_overlap


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


def supported_in_text(source, answer, tokenizer, model, device, threshold=0.5):
    """
    Returns True ONLY if the answer introduces supported content.
    """

    # STEP 1: lexical grounding
    if lexical_grounded(answer, source):
        return True  # NON nonsense

    # STEP 2: NLI (solo se lessicalmente non grounded)
    wrapped_answer = f"The text mentions {answer}."
    entailed = is_entailed_sentence(
        source, wrapped_answer, tokenizer, model, device, threshold
    )

    return entailed



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
                    continue

            if not isinstance(answers, list):
                continue

            # Ensure questions is a list
            if isinstance(questions, str):
                try:
                    questions = json.loads(questions)
                except Exception:
                    continue

            if not isinstance(questions, list):
                continue

            entailed_answers = []
            entailed_questions = []

            for ans, q in zip(answers, questions):
                try:
                    if supported_in_text(ans, source, tokenizer, model, device, args.threshold):
                        entailed_answers.append(ans)
                        entailed_questions.append(q)
                    else:
                        print("\nNOT SUPPORTED:")
                        print("ANSWER:", ans)
                        print("SOURCE:", source)

                except Exception as e:
                    print("Error", str(ans)[:50], e)

            obj["questions"] = entailed_questions
            obj["answers"] = entailed_answers

            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
            fout.flush()

            print(len(entailed_answers), "/", len(answers), "entailed")

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

if __name__ == "__main__":
    main()