#!/usr/bin/env python
# baseline_stream_vllm_save.py
# ------------------------------------------------------------
# 取前 10 条样本 → 贪心生成（≤1024 token）→ 实时打印 + 保存 json
# ------------------------------------------------------------
import argparse, json, os
from itertools import islice
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

# ---------- 保留你原来的 prompt 构造 ----------
def build_inference_prompt(data_point):
    """
    根据单个数据点构建用于推理的 Prompt。
    这个函数必须严格复制你训练脚本中的 Prompt 格式。
    (此函数无需改动)
    """
    prompt_str = (
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
        f"Problem: {data_point['question']}\n"
        f"Solution: {data_point['model_solution']}\n"
        f"Final Answer: {data_point['answer']}\n"
        "Output: "
    )
    # vLLM 不需要 role messages, 它直接处理格式化好的 prompt 字符串
    return prompt_str


def load_first_n(path, n=10):
    with open(path, "r", encoding="utf-8") as f:
        return list(islice(json.load(f), n))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--data_path",  required=True)
    ap.add_argument("--out_file",   required=True,
                    help="保存结果的 json 路径")
    ap.add_argument("--tensor_parallel_size", type=int, default=8)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--max_new_tokens", type=int, default=2048)
    args = ap.parse_args()

    # ---- vLLM 模型 ----
    llm = LLM(
        model=args.model_path,
        trust_remote_code=True,
        tensor_parallel_size=args.tensor_parallel_size,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    raw = load_first_n(args.data_path, 10)
    prompts, meta = [], []        # meta 用来回写答案
    for d in raw:
        messages = [
            {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
            {"role": "user",   "content": build_inference_prompt(d)},
        ]
        p = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        prompts.append(p)
        meta.append({"question": d["question"],
                     "verification_gt": d["verification"]})

    prompts.sort(key=len)         # 提前排序省显存

    params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=args.max_new_tokens,
        stop_token_ids=[tokenizer.eos_token_id],
    )

    results = []   # ↓ 收集输出文本
    print(f"\n>>> Total prompts: {len(prompts)}; batch_size={args.batch_size}\n")

    try:   # vLLM ≥ 0.3 (支持流式)
        streams = llm.generate(prompts, params, stream=True)
        for i, stream in enumerate(streams):
            print(f"\n===== Sample {i+1} =====")
            chunks = []
            for chunk in stream:
                print(chunk.text, end="", flush=True)
                chunks.append(chunk.text)
            full_text = "".join(chunks).strip()
            results.append(full_text)
            print("\n")
    except TypeError:  # 旧版 vLLM
        outputs = llm.generate(prompts, params)
        for i, out in enumerate(outputs):
            full_text = out.outputs[0].text.strip()
            print(f"\n===== Sample {i+1} =====\n{full_text[:2000]}\n")
            results.append(full_text)

    # ---- 写文件 ----
    for m, pred in zip(meta, results):
        m["model_output"] = pred      # 回填模型输出

    with open(args.out_file, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"\n[✓] 已保存 10 条结果 → {args.out_file}")

if __name__ == "__main__":
    main()
