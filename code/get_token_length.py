import json
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoTokenizer
from tqdm import tqdm

# ─── 1. 配置 (Configuration) ───
# ⚠️ 请将这里替换为你的JSON数据文件路径
JSON_FILE_PATH = "overthinking_Dr.Dai/code/datas/our_datas/math_test_datas_verification.json"
MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"

# ─── 2. 定义与你训练脚本完全一致的 Prompt 模板 ───
# To ensure the token count is accurate, we use the exact same prompt structure.
PROMPT_TEMPLATE = (
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
    "Problem: {question}\n"
    "Solution: {model_solution}\n"
    "Final Answer: {answer}\n"
    "Output: "
)


def analyze_token_distribution():
    """
    Loads data, tokenizes it according to the training script logic,
    and reports the token length distribution.
    """
    print(f"✅ Loading tokenizer: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False, trust_remote_code=True)

    print(f"✅ Loading data from: {JSON_FILE_PATH}")
    try:
        with open(JSON_FILE_PATH, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
    except FileNotFoundError:
        print(f"❌ Error: The file '{JSON_FILE_PATH}' was not found.")
        print("Please update the 'JSON_FILE_PATH' variable in the script.")
        return
    except json.JSONDecodeError:
        print(f"❌ Error: The file '{JSON_FILE_PATH}' is not a valid JSON file.")
        return

    token_lengths = []
    answer_token_lengths = []
    print("⏳ Tokenizing all data points to calculate lengths...")

    for item in tqdm(raw_data, desc="Processing data"):
        # 1. 构建 Prompt
        prompt_text = PROMPT_TEMPLATE.format(
            question=item['question'],
            model_solution=item['model_solution'],
            answer=item['answer']
        )
        messages = [
            {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
            {"role": "user", "content": prompt_text}
        ]

        # 2. Tokenize Prompt 部分 (与训练脚本逻辑一致)
        prompt_ids = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True
        )

        # 3. Tokenize Answer 部分 (与训练脚本逻辑一致)
        answer_ids = tokenizer(item["verification"], add_special_tokens=False)["input_ids"]

        # 4. 计算总长度
        # 总长度 = prompt长度 + answer长度 + eos_token_id (1个)
        total_length = len(prompt_ids) + len(answer_ids) + 1
        answer_token_lengths.append(len(answer_ids))
        token_lengths.append(total_length)

    if not token_lengths:
        print("❌ No data was processed. Please check your JSON file.")
        return

    # --- 3. 统计和报告 (Statistics and Reporting) ---
    lengths_np = np.array(token_lengths)
    lengths_answer_np = np.array(answer_token_lengths)
    
    
    print("\n" + "=" * 50)
    print("📊 Token Length Distribution Report")
    print("=" * 50)
    print(f"Total samples analyzed: {len(lengths_answer_np)}")
    print(f"Max token length:   {lengths_answer_np.max()}")
    print(f"Min token length:   {lengths_answer_np.min()}")
    print(f"Mean token length:  {lengths_answer_np.mean():.2f}")
    print(f"Median (50th percentile): {np.percentile(lengths_answer_np, 50):.0f}")
    print(f"90th percentile:    {np.percentile(lengths_answer_np, 90):.0f}")
    print(f"95th percentile:    {np.percentile(lengths_answer_np, 95):.0f}")
    print(f"99th percentile:    {np.percentile(lengths_answer_np, 99):.0f}")
    print("=" * 50 + "\n")
    
    
    # --- 4. 绘制直方图 (Plotting Histogram) ---
    plt.figure(figsize=(12, 6))
    plt.hist(lengths_answer_np, bins=50, alpha=0.75, color='skyblue', edgecolor='black')
    plt.title('Answer Token Length Distribution', fontsize=16)
    plt.xlabel('Token Count', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Add vertical lines for mean and percentiles for better visualization
    plt.axvline(lengths_answer_np.mean(), color='red', linestyle='dashed', linewidth=2, label=f'Mean: {lengths_answer_np.mean():.0f}')
    plt.axvline(np.percentile(lengths_answer_np, 95), color='orange', linestyle='dashed', linewidth=2,
                label=f'95th Percentile: {np.percentile(lengths_answer_np, 95):.0f}')

    plt.legend()
    plt.tight_layout()
    print("📈 Displaying histogram plot. Close the plot window to exit the script.")
    plt.show()



    print("\n" + "=" * 50)
    print("📊 Token Length Distribution Report")
    print("=" * 50)
    print(f"Total samples analyzed: {len(lengths_np)}")
    print(f"Max token length:   {lengths_np.max()}")
    print(f"Min token length:   {lengths_np.min()}")
    print(f"Mean token length:  {lengths_np.mean():.2f}")
    print(f"Median (50th percentile): {np.percentile(lengths_np, 50):.0f}")
    print(f"90th percentile:    {np.percentile(lengths_np, 90):.0f}")
    print(f"95th percentile:    {np.percentile(lengths_np, 95):.0f}")
    print(f"99th percentile:    {np.percentile(lengths_np, 99):.0f}")
    print("=" * 50 + "\n")

    # --- 4. 绘制直方图 (Plotting Histogram) ---
    plt.figure(figsize=(12, 6))
    plt.hist(lengths_np, bins=50, alpha=0.75, color='skyblue', edgecolor='black')
    plt.title('Token Length Distribution', fontsize=16)
    plt.xlabel('Token Count', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Add vertical lines for mean and percentiles for better visualization
    plt.axvline(lengths_np.mean(), color='red', linestyle='dashed', linewidth=2, label=f'Mean: {lengths_np.mean():.0f}')
    plt.axvline(np.percentile(lengths_np, 95), color='orange', linestyle='dashed', linewidth=2,
                label=f'95th Percentile: {np.percentile(lengths_np, 95):.0f}')

    plt.legend()
    plt.tight_layout()
    print("📈 Displaying histogram plot. Close the plot window to exit the script.")
    plt.show()


if __name__ == "__main__":
    analyze_token_distribution()