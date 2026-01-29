from datasets import load_dataset

from collections import defaultdict

import json

ds = load_dataset("lmsys/lmsys-chat-1m", split="train")

best_by_conv = {}

for ex in ds:
    cid = ex["conversation_id"]
    turn = ex["turn"]

    if (cid not in best_by_conv) or (turn > best_by_conv[cid]["turn"]):
        best_by_conv[cid] = ex

print("unique conversations:", len(best_by_conv))
def conversation_to_text(conv):
    return "\n".join(f"{m['role']}: {m['content']}" for m in conv)

output_path = "lmsys_chat1m_conversations_text.jsonl"

with open(output_path, "w", encoding="utf-8") as f:
    for cid, ex in best_by_conv.items():
        text = conversation_to_text(ex["conversation"])
        record = {
            "conversation_id": cid,
            "model": ex["model"],
            "language": ex["language"],
            "text": text,
        }
        json.dump(record, f, ensure_ascii=False)
        f.write("\n")

print("saved to", output_path)
output_path = "lmsys_chat1m_conversations_structured.jsonl"

with open(output_path, "w", encoding="utf-8") as f:
    for cid, ex in best_by_conv.items():
        record = {
            "conversation_id": cid,
            "model": ex["model"],
            "language": ex["language"],
            "conversation": ex["conversation"],
        }
        json.dump(record, f, ensure_ascii=False)
        f.write("\n")
