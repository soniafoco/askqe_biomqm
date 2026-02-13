import json
import argparse

#This script reads a JSONL file containing human-annotated translation data and assigns a 
#binary decision label (“accept” or “reject”) to each entry based on the severity of target-side 
#errors. If any error in target_errors has severity marked as major or critical, the translation 
#is labeled as “reject”; otherwise, it is labeled as “accept”, and the updated records are written 
#to a new JSONL output file.

def annotate(input_file, output_file):
    with open(input_file, "r", encoding="utf-8") as infile, open(output_file, "w", encoding="utf-8") as outfile:
        for line in infile:
            data = json.loads(line.strip())
            reject_decision = False
            if "errors_tgt" in data:
                for error in data["errors_tgt"]:
                    if error.get("severity", "").lower() in ["critical", "major"]:
                        reject_decision = True
                        break
            
            if reject_decision:
                data["decision"] = "reject"
            else:
                data["decision"] = "accept"
        
            json.dump(data, outfile, ensure_ascii=False)
            outfile.write("\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    
    args = parser.parse_args()

    annotate(args.input, args.output)