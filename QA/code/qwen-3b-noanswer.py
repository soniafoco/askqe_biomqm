import torch
import json
import argparse
from prompt import qa_prompt_no_answer
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
    parser.add_argument("--input_path", type=str) 
    parser.add_argument("--output_path", type=str) 
    parser.add_argument("--sentence_type", type=str) #src/bt_tgt

    args = parser.parse_args()

    # =========================================== Load Dataset ===========================================
    with open(args.input_path, 'r') as f_in, open(args.output_path, 'a') as f_out:
        
        for line in f_in:
            data = json.loads(line)

            sentence = data.get(args.sentence_type, None)
            questions = data.get("questions", None)

            if sentence and questions:
                prompt_template = qa_prompt_no_answer
                prompt = prompt_template.replace("{{sentence}}", sentence).replace("{{questions}}", questions)

                messages = [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt},
                ]

                input_ids = tokenizer.apply_chat_template(
                        messages,
                        add_generation_prompt=True,
                        return_tensors="pt",
                    )
                
                if not isinstance(input_ids, torch.Tensor):
                    input_ids = torch.as_tensor(input_ids["input_ids"])


                input_ids = input_ids.to(model.device)
                
                with torch.no_grad():
                    outputs = model.generate(
                        input_ids=input_ids,
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
                
                data[f'answers'] = generated_answers
                f_out.write(json.dumps(data, ensure_ascii=False) + '\n')
                
            else:
                data["answers"] = []
                f_out.write(json.dumps(data, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()