import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# ------------------- 配置你的路径 -------------------
# 基础模型的路径 (例如 Qwen2.5-3B-Instruct)
BASE_MODEL_PATH = "/datanfs4/xinzheyu/project/models/Qwen/Qwen3-4B"

# 你的 LoRA 适配器 checkpoint 路径
ADAPTER_PATH = "/datanfs4/xinzheyu/project/overthinking_Dr.Dai/code/checkpoints/ckpt-qwen3-4b-verify-16k/checkpoint-381"

# 合并后模型的保存路径 (起个新名字)
MERGED_MODEL_SAVE_PATH = "/datanfs4/xinzheyu/project/models/Qwen3-4B-merged-for-vllm"
# ----------------------------------------------------

def merge_and_save_model(base_model_path, adapter_path, save_path):
    """加载基础模型和LoRA适配器，合并后保存到新目录"""
    print(f"Loading base model from {base_model_path}...")
    # 以 float16 或 bfloat16 加载以节省内存
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        device_map="cpu",  # 在 CPU 上合并，避免 OOM
        trust_remote_code=True,
    )

    print(f"Loading LoRA adapter from {adapter_path}...")
    # 加载 PeftModel
    model = PeftModel.from_pretrained(base_model, adapter_path)

    print("Merging LoRA adapter into the base model...")
    # 合并权重
    model = model.merge_and_unload()
    print("Merge successful.")

    # 加载与基础模型匹配的 Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)

    print(f"Saving merged model and tokenizer to {save_path}...")
    # 保存合并后的模型和 Tokenizer
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print("Model and tokenizer saved successfully!")

if __name__ == "__main__":
    # 确保保存路径存在
    os.makedirs(MERGED_MODEL_SAVE_PATH, exist_ok=True)
    merge_and_save_model(BASE_MODEL_PATH, ADAPTER_PATH, MERGED_MODEL_SAVE_PATH)