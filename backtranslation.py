import json
from deep_translator import GoogleTranslator
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--input_path", type=str) 
parser.add_argument("--output_path", type=str) 

args = parser.parse_args()


updated_jsonl = []
with open(args.input_path, 'r', encoding='utf-8') as f:
    for line in f:
        data = json.loads(line.strip())
        tgt_lang = data["lang_tgt"]
        translator = GoogleTranslator(source=tgt_lang, target='en')
        if "tgt" in data:
            print("Translation: ", data["tgt"])
            try:
                translated_text = translator.translate(data["tgt"])
                print("Backtranslation: ", translated_text)
                data["bt_tgt_gt"] = translated_text
            except Exception as e:
                print(f"Translation failed for: {data['tgt']} with error: {e}")
                data["bt_tgt_gt"] = ""
        updated_jsonl.append(data)

with open(args.output_path, 'w', encoding='utf-8') as f:
    for entry in updated_jsonl:
        f.write(json.dumps(entry, ensure_ascii=False) + '\n')