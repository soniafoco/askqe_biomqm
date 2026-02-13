from typing import dataclass_transform
import json
import pandas as pd
import numpy as np
import argparse

def compare(input_file, output_file): 
    results = []

    with open(input_file, "r", encoding="utf-8") as f_nli:
        for line in input_file:
            try:
                data = json.loads(line)

                results.append({"Language": f"{data['lang_src']}-{data['lang_tgt']}",
                        "Severity": data["severity"],
                        "F1": data["avg_f1"],
                        "EM": data["avg_em"],
                        "CHRF": data["avg_chrf"],
                        "BLEU": data["avg_bleu"]}
                        )
                
            except json.JSONDecodeError as e:
                print(f"Skipping invalid JSON line: {line.strip()}")
                continue

    df = pd.DataFrame(results)

    df['Severity'] = df['Severity'].apply(lambda x: 'Major+Critical' if x in ['Major', 'Critical'] else x)

    summary = df.groupby(["Language", "Severity"]).mean()
    summary.to_csv(output_file, index=True, encoding="utf-8")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    
    args = parser.parse_args()
    
    compare(args.input, args.output)
