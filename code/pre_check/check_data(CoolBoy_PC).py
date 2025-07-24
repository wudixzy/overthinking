import argparse
import os
from transformers import AutoTokenizer
from tqdm import tqdm
from utlis import read_json_file, dict_list_to_hf_dataset  # 保持原训练脚本路径

# ----------------------- 引入训练用的数据处理函数 ------------------------ #
def build_dataset(tokenizer, raw_datas, max_len):
    processed = []
    for d in tqdm(raw_datas, desc="Tokenizing"):
        prompt = (
            "You will be given a Problem, a Solution with a thinking process, and a Final Answer.\n"
            "Your task is to annotate each round of verification within <think></think> tags using <vv></vv> and <iv></iv> labels.\n\n"
            "Annotation Rules:\n"
            "a) Valid verification (<vv></vv>):\n"
            "- Corrects previous reasoning errors.\n"
            "- Represents the first verification of a reasoning step.\n"
            "b) Invalid verification (<iv></iv>):\n"
            "- Repeats already verified reasoning step without changing the answer.\n"
            "- Only confirms existing results.\n"
            "c) First-round verification: The initial verification must be labeled as <vv></vv>.\n"
            "d) Nesting requirement: Ensure tags are properly closed without altering the original text structure and content.\n"
            "e) Each full verification round must be completely wrapped in its tags. Never leave any verification content outside the tags.\n"
            "f) A verification round includes all thoughts in that round, from the verification trigger to its conclusion.\n"
            "Example:\n"
            "Problem: What is 1 + 1?\n"
            "Solution: <think> Okay, I need to calculate 1 + 1. The result is 2. Let me double-check again, 2 - 1 = 1, so 1 + 1 = 2. "
            "Yes, it is 2. Let me confirm again. Yes, it is 2. </think> I think the answer is 2.\n"
            "Final Answer: 2.\n"
            "Output: <think> Okay, I need to calculate 1 + 1. The result is 2. <vv> Let me double-check again, 2 - 1 = 1, so 1 + 1 = 2. "
            "Yes, it is 2. </vv> <iv> Let me confirm again. Yes, it is 2. </iv> </think> I think the answer is 2.\n\n"
            f"Problem: {d['question']}\n"
            f"Solution: {d['model_solution']}\n"
            f"Final Answer: {d['answer']}\n"
            "Output: "
        )

        messages = [
            {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
        prompt_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True)
        answer_ids = tokenizer(d["verification"], add_special_tokens=False)["input_ids"]

        input_ids = prompt_ids + answer_ids + [tokenizer.eos_token_id]
        labels = [-100] * len(prompt_ids) + answer_ids + [tokenizer.eos_token_id]

        input_ids = input_ids[:max_len]
        labels = labels[:max_len]

        processed.append({"input_ids": input_ids, "labels": labels})

    return dict_list_to_hf_dataset(processed)


# ------------------------ 引入你的 sanity check 函数 ----------------------- #
def inspect_hf_dataset(ds, tokenizer, max_seq_len=None, show_bad_samples=3):
    import numpy as np

    length_mismatch, no_eos, over_len = [], [], []
    low_cover, bad_align, empty_label = [], [], []

    cov_rates, inp_lens = [], []

    for idx, sample in enumerate(ds):
        inp, lbl = sample["input_ids"], sample["labels"]

        if len(inp) != len(lbl):
            length_mismatch.append(idx)
        if inp[-1] != tokenizer.eos_token_id or lbl[-1] != tokenizer.eos_token_id:
            no_eos.append(idx)
        if max_seq_len and len(inp) > max_seq_len:
            over_len.append(idx)

        valid_cnt = sum(l != -100 for l in lbl)
        cov = valid_cnt / len(lbl)
        cov_rates.append(cov)
        inp_lens.append(len(inp))

        if valid_cnt == 0:
            empty_label.append(idx)
        elif cov < 0.02:
            low_cover.append(idx)

        try:
            first_valid = next(i for i, l in enumerate(lbl) if l != -100)
            if any(lbl[i] != -100 for i in range(first_valid)):
                bad_align.append(idx)
        except StopIteration:
            pass

    def _echo(name, arr):
        if arr:
            print(f"❌ {name}: {len(arr)} 个 → index 示例 {arr[:5]}")
    print(f"\n📊 样本总数: {len(ds)}")
    _echo("input/label 长度不一致", length_mismatch)
    _echo("末尾缺少 <eos>", no_eos)
    _echo("超出 max_seq_len", over_len)
    _echo("labels 全被 -100 覆盖", empty_label)
    _echo("labels 覆盖率 < 2%", low_cover)
    _echo("labels 与 prompt 错位", bad_align)

    print(f"\n✅ 平均 label 覆盖率: {np.mean(cov_rates):.2%}")
    print(f"✅ 输入长度 P50 / P95: "
          f"{np.percentile(inp_lens,50):.0f} / {np.percentile(inp_lens,95):.0f}\n")

    if show_bad_samples and (length_mismatch or bad_align or low_cover):
        bad_pool = (length_mismatch + bad_align + low_cover)[:show_bad_samples]
        print("🔍 抽样查看问题样本（前 120 tokens）：")
        for idx in bad_pool:
            text = tokenizer.decode(ds[idx]["input_ids"][:], skip_special_tokens=False)
            length = len(ds[idx]["input_ids"])
            # text_labels = tokenizer.decode(ds[idx]["labels"][:], skip_special_tokens=False)
            print(f"\n--- sample #{idx}--\nlenght #{length}\n ---\n{text}\n")


# ------------------------ 命令行入口 ------------------------ #
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default='/datanfs4/xinzheyu/project/overthinking_Dr.Dai/code/datas/our_datas/math_train_datas_verification.json',
                        help="已处理数据 (json/arrow/parquet) 或 HF dataset repo 本地路径")
    parser.add_argument("--tokenizer", default='/datanfs4/xinzheyu/project/models/Qwen/Qwen2.5-3B-Instruct',
                        help="Tokenizer 路径 (必须与训练一致)")
    parser.add_argument("--max_seq_len", type=int, default=16384)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer,
        use_fast=False,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    raw_data = read_json_file(args.data)
    processed_ds = build_dataset(tokenizer, raw_data, args.max_seq_len)

    inspect_hf_dataset(processed_ds, tokenizer, max_seq_len=args.max_seq_len)