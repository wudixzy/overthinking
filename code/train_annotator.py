import os

# 分布式LoRA和DDP的冲突的解决方案：
# 1:
"""
    加入ddp_find_unused_parameters=False 参数
        args = TrainingArguments(
            output_dir="./checkpoints/qwen25-3b-lora-verification",
            per_device_train_batch_size=1,
            gradient_accumulation_steps=4,
            num_train_epochs=3,
            learning_rate=1e-4,
            logging_steps=10,
            save_steps=100,
            bf16=True,
            report_to="none",
            ddp_find_unused_parameters=False,
        )
"""
# 2:
"""
    设置参数 "use_reentrant": False 注意通过gradient_checkpointing_kwargs传入gradient_checkpointing_enable函数中
        base_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

"""

# 错误的根本原因：DDP找没用到的参数时标记的Hook，而在PyTorch的默认Activation Checkpointing实现的backward中，
# 会重新完整进行一次forward导致再次进行参数的标记，两次标记的行为是被禁止的，因此出现问题。 但是这个解释只能解释通方案2，
# 方案1的解决原理应该与此不同，但是暂时还没有找到合理的解释，方案2涉及到的底层函数我也没有搞清楚。

os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"
os.environ["TORCH_DISABLE_DISTRIBUTED_DTENSOR"] = "1"
os.environ["TORCH_COMPILE"] = "0"
os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"
os.environ["ACCELERATE_FIND_UNUSED_PARAMETERS"] = "false"

import torch
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    TrainingArguments, Trainer, DataCollatorForSeq2Seq
)
from utlis import read_json_file, dict_list_to_hf_dataset
from tqdm import tqdm
from peft import LoraConfig, TaskType, get_peft_model

class TrainAnnotator:
    def __init__(self, model_name='/datanfs4/xinzheyu/project/models/Qwen/Qwen2.5-3B-Instruct'):
        self.train_datas = read_json_file('datas/our_datas/math_train_datas_verification.json')
        self.test_datas = read_json_file('datas/our_datas/math_test_datas_verification.json')

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, use_fast=False, trust_remote_code=True
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            inference_mode=False,
            r=8,
            lora_alpha=32,
            lora_dropout=0.1
        )

        base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
        )
        base_model.enable_input_require_grads()
        base_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

        self.model = get_peft_model(base_model, self.config)

        self.train_ids = self.process_function(self.train_datas, max_length=4096)
        # self.test_ids = self.process_function(self.test_datas)

    def process_function(self, datas, max_length=20000):
        processed_datas = []
        for data in tqdm(datas, desc='Processing'):
            question = data['question']
            model_solution = data['model_solution']
            answer = data['answer']

            input_prompt = (
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
                f"Problem: {question}\nSolution: {model_solution}\nFinal Answer: {answer}\nOutput: "
            )

            instruction = self.tokenizer(
                f"<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n"
                f"<|im_start|>user\n{input_prompt}<|im_end|>\n"
                f"<|im_start|>assistant\n",
                add_special_tokens=False
            )

            response = self.tokenizer(data["verification"], add_special_tokens=False)

            input_ids = instruction["input_ids"] + response["input_ids"] + [self.tokenizer.pad_token_id]
            attention_mask = instruction["attention_mask"] + response["attention_mask"] + [1]
            labels = [-100] * len(instruction["input_ids"]) + response["input_ids"] + [self.tokenizer.pad_token_id]

            if len(input_ids) > max_length:
                input_ids = input_ids[:max_length]
                attention_mask = attention_mask[:max_length]
                labels = labels[:max_length]

            processed_datas.append({
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels
            })

        return dict_list_to_hf_dataset(processed_datas)

    def train(self):
        args = TrainingArguments(
            output_dir="./checkpoints/qwen3-4b-lora-verification",
            per_device_train_batch_size=1,
            gradient_accumulation_steps=4,
            num_train_epochs=3,
            learning_rate=1e-4,
            logging_steps=10,
            save_steps=100,
            bf16=True,
            report_to="none",
            # ddp_find_unused_parameters=False,
        )

        trainer = Trainer(
            model=self.model,
            args=args,
            tokenizer=self.tokenizer,
            train_dataset=self.train_ids,
            data_collator=DataCollatorForSeq2Seq(self.tokenizer, padding=True),
        )

        trainer.train()


def main():
    trainer = TrainAnnotator(model_name="/datanfs4/xinzheyu/project/models/Qwen/Qwen3-4B")
    trainer.train()


if __name__ == "__main__":
    main()
