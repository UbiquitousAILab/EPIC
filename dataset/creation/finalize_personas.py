import json
import sys

in_file = sys.argv[1]
out_file = in_file.replace(".json", "_finalized.json")

with open(sys.argv[1], "r", encoding="utf-8") as f:
    data = json.load(f)

processed_data = [{
    "persona_index": i,
    "preference_blocks": [
        {
            "preference": p["preference"],
            "queries": [
                {
                    "question": p["question"],
                }
            ]
        }
        for p in persona["preferences"]
    ]
} for i, persona in enumerate(data)]

with open(out_file, "w", encoding="utf-8") as f:
    json.dump(processed_data, f, indent=2, ensure_ascii=False)