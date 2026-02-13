from typing import dataclass_transform
import json
import pandas as pd
import numpy as np
import argparse

def compare(input_file, output_grouped, output_global): 
    results = []

    with open(input_file, "r", encoding="utf-8") as f_nli:
        for line in f_nli:
            try:
                data = json.loads(line)

                results.append({"Language": f"{data['lang_src']}-{data['lang_tgt']}",
                        "Severity": data["severity"],
                        "F1": data["avg_f1"],
                        "EM": data["avg_em"],
                        "CHRF": data["avg_chrf"],
                        "BLEU": data["avg_bleu"],
                        "SBERT": data["avg_cos_similarity"],
                        "NLI": data["avg_nli"]})
                
            except json.JSONDecodeError as e:
                print(f"Skipping invalid JSON line: {line.strip()}")
                continue

    df = pd.DataFrame(results)

    # Summary per langguage and error type
    df['Severity'] = df['Severity'].apply(lambda x: 'Major+Critical' if x in ['Major', 'Critical'] else x)
    summary = df.groupby(["Language", "Severity"]).mean()
    summary.to_csv(output_grouped, index=True, encoding="utf-8")
    print("GROUPED (lang-error) summary:")
    print(summary)

    # Global summary
    numeric_columns = ["F1", "EM", "CHRF", "BLEU", "SBERT"]
    global_summary = df[numeric_columns].mean()
    global_summary.to_csv(output_global, index=True, encoding="utf-8")
    print("GLOBAL summary:")
    print(global_summary)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--output_grouped', type=str, required=True)
    parser.add_argument('--output_global', type=str, required=True)
    
    args = parser.parse_args()
    
    compare(args.input, args.output_grouped, args.output_global)
