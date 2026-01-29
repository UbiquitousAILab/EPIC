import json

def count_words(text: str) -> int:
    return len(text.split())

def conversation_to_chunks(conv,
                           max_words=180,   
                           max_turns=8,     
                           overlap_turns=2  
                           ):
    """
    conv: [{"role": "user"/"assistant", "content": "..."},
           ...]  (하나의 conversation 전체)
    return: [chunk_text1, chunk_text2, ...]
    """
    chunks = []
    cur_msgs = []
    cur_words = 0

    for msg in conv:
        msg_text = f"{msg['role']}: {msg['content']}"
        msg_words = count_words(msg_text)

        if cur_msgs and (
            cur_words + msg_words > max_words or
            len(cur_msgs) >= max_turns
        ):
            chunks.append("\n".join(cur_msgs))

            if overlap_turns > 0:
                cur_msgs = cur_msgs[-overlap_turns:]
                cur_words = sum(count_words(m) for m in cur_msgs)
            else:
                cur_msgs = []
                cur_words = 0

        cur_msgs.append(msg_text)
        cur_words += msg_words

    if cur_msgs:
        chunks.append("\n".join(cur_msgs))

    return chunks

input_path = "lmsys_chat1m_conversations_structured.jsonl"
output_path_chunks = "lmsys_chat1m_conv_chunks_text.jsonl"

with open(input_path, "r", encoding="utf-8") as fin, \
     open(output_path_chunks, "w", encoding="utf-8") as fout:

    for line in fin:
        ex = json.loads(line)
        cid = ex["conversation_id"]
        conv = ex["conversation"]
        chunks = conversation_to_chunks(conv)

        for idx, chunk_text in enumerate(chunks):
            record = {
                "conversation_id": cid,
                "chunk_id": idx,
                "model": ex["model"],
                "language": ex["language"],
                "text": chunk_text,
            }
            json.dump(record, fout, ensure_ascii=False)
            fout.write("\n")

print("saved chunked conversations to", output_path_chunks)
