import os
import json
import argparse
from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
import evaluate


# 全局变量定义 (与原脚本保持一致)
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


def main():
    parser = argparse.ArgumentParser()
    # **重要**: 这里的 model_path 指向你第一步中合并好的模型路径
    parser.add_argument("--model_path", type=str,
                        default="/datanfs4/xinzheyu/project/models/Qwen2.5-3B-Instruct-merged-for-vllm",
                        help="合并 LoRA 后的模型路径")
    parser.add_argument("--test_file", type=str,
                        default="/datanfs4/xinzheyu/project/overthinking_Dr.Dai/code/datas/our_datas/math_test_datas_verification.json",
                        help="评估用的 JSON 数据文件路径")
    parser.add_argument("--output_file", type=str,
                        default="/datanfs4/xinzheyu/project/overthinking_Dr.Dai/code/results/qwen2.5/eval_results_vllm.json",
                        help="保存生成结果的文件路径")
    parser.add_argument("--max_new_tokens", type=int, default=7168, help="模型生成的最大 token 数")
    parser.add_argument("--tensor_parallel_size", type=int, default=4, help="用于多 GPU 推理的张量并行数")

    args = parser.parse_args()

    # 1. 使用 vLLM 加载模型
    print(f"从 {args.model_path} 使用 vLLM 加载模型...")
    llm = LLM(
        model=args.model_path,
        trust_remote_code=True,
        tensor_parallel_size=args.tensor_parallel_size,
    )
    print("模型已成功使用 vLLM 加载。")

    # 2. 单独加载 Tokenizer 用于构建 Chat Template
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 3. 加载评估数据并构建 Prompts
    with open(args.test_file, 'r', encoding='utf-8') as f:
        test_data = json.load(f)

    # --- 修改开始 ---
    # 创建一个列表来存放打包好的数据（原始索引, prompt, 标准答案）
    # 这样可以在排序后依然保持数据对齐
    indexed_data = []
    references = []  # 标准答案列表，保持原始顺序用于最终评估
    for i, item in enumerate(test_data):
        messages = [
            {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
            {"role": "user", "content": build_inference_prompt(item)}
        ]
        prompt_templated = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

        # 打包原始索引和生成的prompt
        indexed_data.append((i, prompt_templated))
        references.append(item["verification"])

    # 根据 prompt 的长度对打包好的数据进行排序
    print("为实现高效批处理，正在按 prompt 长度进行排序...")
    indexed_data.sort(key=lambda x: len(x[1]))

    # 解包排序后的数据，得到排序后的索引列表和 prompts 列表
    sorted_indices, sorted_prompts = zip(*indexed_data)
    # --- 修改结束 ---

    # 4. 设置采样参数
    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.8,
        top_k=20,
        min_p=0,
        max_tokens=args.max_new_tokens,
        stop_token_ids=[tokenizer.eos_token_id],  # 设置停止 token
    )

    # 5. 使用 vLLM 对【排序后】的 prompts 进行批量推理
    print(f"开始对 {len(sorted_prompts)} 个已排序的样本进行推理...")
    # 将排序后的 prompts 传递给模型
    sorted_outputs = llm.generate(sorted_prompts, sampling_params)
    print("推理完成。")

    # 从 vLLM 的输出中提取文本结果，此时结果是排序后的
    sorted_predictions = [output.outputs[0].text.strip() for output in sorted_outputs]

    # --- 修改开始 ---
    # 使用之前保存的索引，将预测结果恢复到原始顺序
    print("正在将预测结果恢复至原始顺序以便评估...")
    predictions = [None] * len(sorted_predictions)
    for original_index, prediction in zip(sorted_indices, sorted_predictions):
        predictions[original_index] = prediction
    # --- 修改结束 ---

    # --- 推理完成后立即保存预测结果（不含评估） ---
    print("正在保存推理结果（未评估）...")
    raw_output_path = args.output_file.replace(".json", "_predict_only.json")
    with open(raw_output_path, 'w', encoding='utf-8') as f:
        for i, item in enumerate(test_data):
            item['model_prediction'] = predictions[i]
        json.dump(test_data, f, ensure_ascii=False, indent=4)
    print(f"[✓] 推理结果已保存至 {raw_output_path}")

    # --- 计算评估指标 ---
    print("开始评估...")
    exact_match = sum(pred.strip() == ref.strip() for pred, ref in zip(predictions, references))
    em_score = exact_match / len(predictions) * 100

    rouge = evaluate.load('rouge')
    rouge_results = rouge.compute(predictions=predictions, references=references)

    print("\n--- 评估结果 (vLLM 排序推理) ---")
    print(f"精确匹配率 (EM): {em_score:.2f}%")
    print(f"ROUGE-L F1 分数: {rouge_results['rougeL'] * 100:.2f}")
    print("----------------------------------\n")

    # --- 将评估后的版本再保存一次 ---
    print("正在保存评估后的完整结果...")
    for i, item in enumerate(test_data):
        item['model_prediction'] = predictions[i]
        item['em'] = int(predictions[i].strip() == references[i].strip())  # 每条打上 EM 标签
    evaluated_path = args.output_file.replace(".json", "_evaluated.json")
    with open(evaluated_path, 'w', encoding='utf-8') as f:
        json.dump(test_data, f, ensure_ascii=False, indent=4)
    print(f"[✓] 完整评估结果已保存至 {evaluated_path}")


if __name__ == "__main__":
    main()