import torch
import json
from prompt import atomic_fact_prompt
import os
from transformers import AutoTokenizer, AutoModelForCausalLM

model_id = "Qwen/Qwen2.5-3B-Instruct"   

input_file = f"dev_with_backtranslation.jsonl"
output_file = f"askqe_atomic_facts.jsonl"

def main():
    # =========================================== LLM Setup ===========================================
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )

    # =========================================== Load Dataset ===========================================
    with open(input_file, "r", encoding="utf-8") as file, open(output_file, "w", encoding="utf-8") as out_file:
        for line in file:
            data = json.loads(line)
            if "src" in data:
                sentence = data["src"]
                prompt = atomic_fact_prompt.replace("{{sentence}}", sentence)

                messages = [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt},
                ]

                input_ids = tokenizer.apply_chat_template(
                        messages,
                        add_generation_prompt=True,
                        return_tensors="pt",
                    )
                
                with torch.no_grad():
                    outputs = model.generate(
                        input_ids,
                        max_new_tokens=1024,
                        eos_token_id=tokenizer.eos_token_id,
                    )

                response_ids = outputs[0][input_ids.shape[-1]:]
                response = tokenizer.decode(response_ids, skip_special_tokens=True)
                response = response.strip()
                if response.lower().startswith("atomic facts"):
                    response = response.split(":", 1)[1].strip()

                print("> ", response)
                print("=" * 80)
                
                data[f"atomic_facts"] = response
                out_file.write(json.dumps(data, ensure_ascii=False) + "\n")

            else:
                pass

if __name__ == "__main__":
    main()