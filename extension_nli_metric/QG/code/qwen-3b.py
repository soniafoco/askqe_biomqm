import torch
import json
import argparse
from prompt import prompts
import os
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
    parser.add_argument("--prompt", type=str)
    args = parser.parse_args()

    # =========================================== Load Dataset ===========================================

    with open("dev_with_backtranslation.jsonl", 'r') as f_in, open(args.output_path, 'w') as f_out:
        for line in f_in:
            data = json.loads(line)
            sentence = data.get('src', None)
            print(sentence)
            if sentence:
                prompt_template = prompts[args.prompt]

                # Default to 'vanilla' prompt format if atomic_facts are missing/empty
                if args.prompt == "semantic":
                    semantic = data.get('semantic_roles', None)
                    if semantic:
                        prompt = prompt_template.replace("{{sentence}}", sentence).replace("{{semantic_roles}}", semantic)
                    else:
                        prompt = prompt_template.replace("{{sentence}}", sentence)

                elif args.prompt == "atomic":
                    atomics = data.get('atomic_facts', None)
                    if atomics:
                        prompt = prompt_template.replace("{{sentence}}", sentence).replace("{{atomic_facts}}", str(atomics))
                    else:
                        prompt = prompt_template.replace("{{sentence}}", sentence)

                else:  # Default case (vanilla)
                    prompt = prompt_template.replace("{{sentence}}", sentence)
                
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
                generated_questions = tokenizer.decode(response, skip_special_tokens=True)

                if generated_questions:
                    generated_questions = generated_questions.strip('"\'')
                
                print(f"> {generated_questions}")
                print("\n======================================================\n")

                data['questions'] = generated_questions
                f_out.write(json.dumps(data, ensure_ascii=False) + '\n')
            else:
                pass

if __name__ == "__main__":
    main()