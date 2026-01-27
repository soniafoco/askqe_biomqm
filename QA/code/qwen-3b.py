import torch
import json
import os
import argparse
from prompt import qa_prompt
from transformers import AutoTokenizer, AutoModelForCausalLM


model_id = "Qwen/Qwen2.5-3B-Instruct"

def main():
    # =========================================== LLM Setup ===========================================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
             cache_dir="",
            device_map="auto",
        )
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", type=str) 
    parser.add_argument("--sentence_type", type=str) # "src" or "bt_tgt"
    parser.add_argument("--prompt", type=str) #atomic/vanilla

    args = parser.parse_args()

    # =========================================== Load Dataset ===========================================
    with open(f"../QG/{args.prompt}_qwen-3B.jsonl", 'r') as f_qg, open(f"askqe_atomic_facts_bt.jsonl", 'r') as f_bt, open(f"{args.output_path}-{args.prompt}.jsonl", 'a') as f_out:
        for qg_line, bt_line in zip(f_qg, f_bt):
            qg_data = json.loads(qg_line)
            data = json.loads(bt_line)

            questions = qg_data.get("questions")
            sentence = data.get(args.sentence_type, None)

            if sentence and questions:
                prompt_template = qa_prompt
                prompt = prompt_template.replace("{{sentence}}", sentence).replace("{{questions}}", questions)

                messages = [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt},
                ]

                input_ids = tokenizer.apply_chat_template(
                        messages,
                        add_generation_prompt=True,
                        return_tensors="pt",
                    ).to(device)
                
                with torch.no_grad():
                    outputs = model.generate(
                        input_ids,
                        max_new_tokens=1024,
                        eos_token_id=tokenizer.eos_token_id,
                    )
                response = outputs[0][input_ids.shape[-1]:]
                generated_answers = tokenizer.decode(response, skip_special_tokens=True)

                if generated_answers:
                    generated_answers = generated_answers.strip('"\'')
                
                print(f"> {questions}")
                print(f"> {generated_answers}")
                print("\n======================================================\n\n")

                data['answers'] = generated_answers
                f_out.write(json.dumps(data, ensure_ascii=False) + '\n')
            else:
                pass


if __name__ == "__main__":
    main()