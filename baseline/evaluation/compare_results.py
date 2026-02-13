from typing import dataclass_transform
import json
import pandas as pd
import numpy as np
import argparse

def compare(input_sbert, input_f1, output_grouped, output_global): 
    results = []

    with open(input_sbert, "r", encoding="utf-8") as f_sbert, open(input_f1, "r", encoding="utf-8") as f_f1:
        for line_sbert, line_f1 in zip(f_sbert, f_f1):
            try:
                sbert_data = json.loads(line_sbert)
                f1_data = json.loads(line_f1)

                results.append({"Language": f"{f1_data['lang_src']}-{f1_data['lang_tgt']}",
                        "Severity": f1_data["severity"],
                        "F1": f1_data["avg_f1"],
                        "EM": f1_data["avg_em"],
                        "CHRF": f1_data["avg_chrf"],
                        "BLEU": f1_data["avg_bleu"],
                        "SBERT": sbert_data["avg_cos_similarity"]}
                        )
                
            except json.JSONDecodeError as e:
                print(f"Skipping invalid JSON line: ", e)
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
    parser.add_argument('--input_sbert', type=str, required=True)
    parser.add_argument('--input_f1', type=str, required=True)
    parser.add_argument('--output_grouped', type=str, required=True)
    parser.add_argument('--output_global', type=str, required=True)
    
    args = parser.parse_args()
    
    compare(args.input_sbert, args.input_f1, args.output_grouped, args.output_global)
