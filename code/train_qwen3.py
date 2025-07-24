#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
train_qwen25_verify.py
------------------------------------------------
• Qwen2.5-3B-Instruct + 全量 LoRA
• 长文本验证标签标注任务
"""

import os, random, json
import numpy as np
from tqdm import tqdm
import torch
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    TrainingArguments, Trainer,
    DataCollatorForLanguageModeling, GenerationConfig,
    TrainerCallback, TrainerState, TrainerControl,
    DataCollatorWithPadding
)
from peft import LoraConfig, TaskType, get_peft_model
from flash_attn.losses.cross_entropy import CrossEntropyLoss

# 你的工具函数（保持和原项目一致）
from utlis import read_json_file, dict_list_to_hf_dataset


# --------------------------- 0. 通用 --------------------------- #
def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything(42)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

SYS_MSG = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."

flash_ce = CrossEntropyLoss(ignore_index=-100, reduction="mean", inplace_backward=True)


class MyTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs["labels"]                          # (B, L)

        logits = model(
            input_ids      = inputs["input_ids"],
            attention_mask = inputs.get("attention_mask")
        ).logits                                            # (B, L, V)

        shift_logits = logits[:, :-1, :].contiguous()       # (B, L-1, V)
        shift_labels = labels[:, 1:].contiguous()           # (B, L-1)

        flat_logits = shift_logits.view(-1, shift_logits.size(-1))
        flat_labels = shift_labels.view(-1)

        # 防护断言（开发期留着，OK 后可删）
        assert flat_logits.shape[0] == flat_labels.shape[0], \
            f"{flat_logits.shape} vs {flat_labels.shape}"

        loss = flash_ce(flat_logits, flat_labels)

        return (loss, shift_logits) if return_outputs else loss


# -------------------- 1. 数据构建 -------------------- #
def build_dataset(tokenizer, raw_datas, max_len):
    processed, n_drop = [], 0
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
            {"role": "system", "content": SYS_MSG},
            {"role": "user",   "content": prompt},
        ]
        prompt_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, enable_thinking=False)
        answer_ids = tokenizer(d["verification"], add_special_tokens=False)["input_ids"]
        total_len = len(prompt_ids) + len(answer_ids) + 1  # +1 eos

        if total_len > max_len:            # —— 超窗直接丢弃
            n_drop += 1
            continue

        input_ids = prompt_ids + answer_ids + [tokenizer.eos_token_id]
        labels    = [-100] * len(prompt_ids) + answer_ids + [tokenizer.eos_token_id]
        processed.append({
            "input_ids": input_ids,
            "labels": labels,
            "prompt_len": len(prompt_ids)  # ← 新增这一行
        })

    print(f"[build_dataset] 丢弃超长样本 {n_drop}/{len(raw_datas)}  ({n_drop/len(raw_datas):.2%})")
    return dict_list_to_hf_dataset(processed)


# -------------------- 2. 贪心自测回调 -------------------- #
class GreedyEvalCallback(TrainerCallback):
    def __init__(self, tokenizer, sample_inputs, interval=40):
        self.tok = tokenizer
        self.sample = sample_inputs
        self.interval = interval
        self.gen_cfg = GenerationConfig(
            max_new_tokens=2048, temperature=0.0, top_p=1.0, top_k=-1,
            eos_token_id=tokenizer.eos_token_id,
        )

    # ---------- callback 中 ----------
    def on_step_end(self, args, state, control, **kwargs):
        # 只在 world_process_zero 执行
        if not state.is_world_process_zero:
            return control
        if state.global_step % self.interval == 0 and state.global_step > 0:
            model = kwargs["model"].eval()
            with torch.no_grad():
                out = model.generate(
                    input_ids=self.sample["input_ids"].to(model.device),
                    attention_mask=self.sample["attention_mask"].to(model.device),
                    generation_config=self.gen_cfg,
                )
            # 改成
            new_tokens = out[0][self.sample["input_ids"].shape[1]:]  # 只取模型新生成部分
            txt = self.tok.decode(new_tokens, skip_special_tokens=True)[:2048]
            print(f"\n[Greedy Eval @ step {state.global_step}] >>> {txt} ...\n")
            model.train()
        return control


# -------------------- 3. 主函数 -------------------- #
def main():
    model_name = "/datanfs4/xinzheyu/project/models/Qwen/Qwen3-4B"
    max_seq_len = 16384

    raw_train = read_json_file("datas/our_datas/math_train_datas_verification.json")
    # 调试代码使用 便于快速查看
    # raw_train = raw_train[:500]

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        trust_remote_code=True,
    )

    # —— LoRA 全量模块 —— #
    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ],
        r=8, lora_alpha=32, lora_dropout=0.1, inference_mode=False,
    )
    model = get_peft_model(base_model, lora_cfg)
    model.print_trainable_parameters()
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()

    train_ds = build_dataset(tokenizer, raw_train, max_seq_len)

    # —— Collator 修正 —— #
    # data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer,
        # pad_to_multiple_of=8,  # 可选，便于 tensor 并行
        return_tensors="pt"
    )

    args = TrainingArguments(
        output_dir="/datanfs4/xinzheyu/project/overthinking_Dr.Dai/code/checkpoints/ckpt-qwen3-4b-verify-16k-2",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,      # ==> 实际等效总 batch 2
        num_train_epochs=4,
        learning_rate=5e-5,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        max_grad_norm=1,
        logging_steps=10,
        save_steps=100,
        bf16=True,
        report_to="none",
        label_names=["labels"],
        gradient_checkpointing=True,
        deepspeed="/datanfs4/xinzheyu/project/overthinking_Dr.Dai/ds_config.json",
        eval_strategy="no",
        # eval_steps=40,
    )

    first = train_ds[0]
    prompt_ids = first["input_ids"][: first["prompt_len"]]  # 只保留 prompt
    sample_inputs = {
        "input_ids": torch.tensor(prompt_ids).unsqueeze(0),
        "attention_mask": torch.ones(1, len(prompt_ids), dtype=torch.long)
    }

    trainer = MyTrainer(
        model=model,
        args=args,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        data_collator=data_collator,
        callbacks=[GreedyEvalCallback(tokenizer, sample_inputs, interval=100)],
    )

    trainer.train()


if __name__ == "__main__":
    main()
