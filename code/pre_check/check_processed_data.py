#!/usr/bin/env python
# dump_first10_processed.py
# ------------------------------------------------------------
# 将 build_dataset 处理后的前 10 条样本保存成人可读格式
# ------------------------------------------------------------
import json
from pathlib import Path
from transformers import AutoTokenizer

# === 根据你的项目路径修改 ===
MODEL_PATH = "/datanfs4/xinzheyu/project/models/Qwen/Qwen2.5-3B-Instruct"
DATA_PATH  = "datas/our_datas/math_train_datas_verification.json"
OUT_FILE   = "processed_preview_first10.txt"
MAX_LEN    = 16384      # 与训练保持一致

# === 你已有的工具 ===
from utlis import read_json_file, dict_list_to_hf_dataset
from train_deepspeed import build_dataset   # 或者直接复制函数进来

def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    raw_datas = read_json_file(DATA_PATH)
    ds = build_dataset(tokenizer, raw_datas, MAX_LEN)

    n = min(10, len(ds))
    with Path(OUT_FILE).open("w", encoding="utf-8") as fout:
        for idx in range(n):
            sample = ds[idx]
            inp_txt = tokenizer.decode(sample["input_ids"], skip_special_tokens=False)

            # 取 labels 中 ≠ -100 的 token 作为目标输出
            target_ids = [tid for tid, lbl in zip(sample["input_ids"], sample["labels"]) if lbl != -100]
            # 可去掉末尾 eos
            if target_ids and target_ids[-1] == tokenizer.eos_token_id:
                target_ids = target_ids[:-1]
            out_txt = tokenizer.decode(target_ids, skip_special_tokens=True)

            fout.write(f"=== Sample {idx+1} ===\n")
            fout.write("## Instruction:  " + inp_txt + "\n\n")
            fout.write("## Label: " + out_txt + "\n\n\n")

    print(f"[✓] 已保存至 {OUT_FILE}")

if __name__ == "__main__":
    main()
