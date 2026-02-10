import torch
import json
import os
import argparse
from prompt import qa_prompt
from transformers import AutoTokenizer, AutoModelForCausalLM


def main():
   
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str) #
    
    parser.add_argument("--prompt", type=str) #atomic/vanilla

    args = parser.parse_args()

    # =========================================== Load Dataset ===========================================
    with open(f"../QG/{args.prompt}_qwen-3B.jsonl", 'r') as f_qg, open(f"askqe_atomic_facts_bt.jsonl", 'r') as f_bt, open(f"{args.output_path}-{args.prompt}.jsonl", 'a') as f_out:
        for qg_line, bt_line in zip(f_qg, f_bt):
            qg_data = json.loads(qg_line)
            data = json.loads(bt_line)

            


if __name__ == "__main__":
    main()