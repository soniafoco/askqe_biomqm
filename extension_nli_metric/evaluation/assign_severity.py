import json
import argparse

# Define severity levels
severity_levels = {"critical": 4, "major": 3, "minor": 2, "neutral": 1, "no error": 0}

def get_highest_severity(xcomet_annotations):
    """Retrieve the highest severity from error spans across all xcomet_annotation entries."""
    max_severity = "no error"

    if not xcomet_annotations:
        return max_severity

    for annotation in xcomet_annotations:
        error_severity = annotation.get("severity", "no error")
        if severity_levels[error_severity.lower()] > severity_levels[max_severity.lower()]:
            max_severity = error_severity  # Update max severity

    return max_severity

def process_jsonl(input_file, output_file):
    """Process JSONL file and assign highest severity."""
    with open(input_file, "r", encoding="utf-8") as infile, open(output_file, "w", encoding="utf-8") as outfile:
        for line in infile:
            data = json.loads(line)
            data["severity"] = get_highest_severity(data.get("errors_tgt", []))  # Ensure it’s a list
            outfile.write(json.dumps(data, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True,)
    parser.add_argument("--output", type=str, required=True,)

    args = parser.parse_args()

    process_jsonl(args.input, args.output)

   