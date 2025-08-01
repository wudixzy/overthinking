# overthinking

## 训练脚本优化全程回顾

按时间顺序记录：从最初“纯 DDP + 标准交叉熵”脚手架，到最终在 **8 × 24 GB 4090** 上跑 **16 K token** 长序列微调。每一步先说 **为什么要做**，再点 **背后技术原理**，最后列 **直接收益**。

---

### 1. 基础框架

* **做了什么**  
  使用 `AutoModelForCausalLM` + `Trainer` + DDP 搭最小闭环。
* **结果**  
  纯 BF16 DDP 最长只能跑 **8 K** token。

---

### 2. 打开 TF32

* **为什么** Ampere+ GPU 用 TensorCore 跑 FP32 GEMM，速度≈↑25 %。  
  * **怎么做**  
    ```python
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32  = True
    ```

* **收益** 推理 & 训练全部加速，无成本。

---

### 3. 引入 Flash-Attention 2

* **原理** Q/K/V 融合单核，访存最少 → 显存 ∝ L。
* **代码**

  ```python
    AutoModelForCausalLM.from_pretrained(..., attn_implementation="flash_attention_2")
  ```
* **收益** 注意力显存≈官方 SDPA 的 60 %，速度 ×2-3。

---

### 4. Gradient Checkpointing

* **原理** 反向时重算，释放激活显存 ≈ 50 %。
* **细节**

  ```python
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()
  ```
* **收益** 将序列长度上限推到 **16 K**。

---

### 5. DeepSpeed ZeRO-3

* **动机** 全参权重 7 GB 仍挤占显存。
* **做法** `zero_optimization.stage = 3`。
* **收益** 权重 + 优化器显存摊到 8 卡，单卡 < 1 GB。

---

### 6. Flash-Attn Memory-Efficient CE

* **问题** 16 K × vocab logits ≈ 10 GB。
* **解决**

  ```python
    from flash_attn.losses.cross_entropy import CrossEntropyLoss
    flash_ce = CrossEntropyLoss(ignore_index=-100, inplace_backward=True)
  ```
* **收益** logits 显存压到 ≈ 3 GB，16 K token 稳跑。

---

### 7. 自定义 `Trainer.compute_loss`

* **为什么** 替换默认 CE，兼容 HF≥4.40 `num_items_in_batch`。
* **实现** 手动 shift labels，签名加 `**kwargs`。
* **收益** 向前向后兼容。

---

### 8. 动态 Padding Collator

* **改动** `DataCollatorWithPadding(tokenizer, return_tensors="pt")`。
* **收益** 计算 token 数下降≈10 %。

---

### 9. 数据预处理改进

* **措施** 超长样本直接丢弃；额外记录 `prompt_len` 便于调试。
* **收益** 避免 run-time 截断引发 label 错位。

---

### 10. GreedyEval Callback

* **功能** 训练中每 N 步 greedy 生成，快速检查标签格式。
* **收益** 第一时间发现梯度爆炸或模式崩溃。

---

### 11. TrainingArguments 调优

* `bf16=True`
* `gradient_accumulation_steps=8`
* `lr_scheduler_type="cosine"`, `warmup_ratio=0.1`
* `max_grad_norm=1`
* **收益** 平稳收敛，显存可控。

---

### 12. Label-Shift & Flash-CE 对齐

* **问题** 未右移导致 `assert labels.shape == (n_rows,)`。
* **改动** `shift_logits = logits[:, :-1]`, `shift_labels = labels[:, 1:]`。
* **收益** loss 首轮 4-6，正常下降。

---

### 13. Padding Bug & 自定义 Collator

* **问题** HF≤4.38 `pad_to_multiple_of=8` 仅补 `input_ids` 不补 `labels`。
* **解决** 升级 4.41 或自写 `collate_fn` 补 `-100`。
* **收益** 消除长度差断言。

---

### 14. 梯度累积 × 学习率线性缩放

* **操作** `GA 2→8`，`lr 1e-5→5e-5`，warm-up ≥ 300 step。
* **结果** 有效 batch=8，loss 曲线更稳，训练 \~2 epoch 收敛到 0.35±。

---

### 15. 离线 W\&B 监控

* **做法**

  ```python
    os.environ["WANDB_MODE"] = "offline"
    args = TrainingArguments(..., report_to="wandb", run_name="qwen3-4b-verify-16k")
  ```
* **同步** `wandb sync ./wandb/offline-run-*`。
* **收益** 完整曲线/grad\_norm/GreedyEval 可视化，无需在线。

---

## 最终效果

| 指标          | 初版     | 最终                 |
| ----------- | ------ | ------------------ |
| **最长序列**    | 8 K    | **16 K**（理论可 24 K） |
| **单卡峰值显存**  | 30 GB+ | **≈ 18 GB**        |
| **可训练参数**   | 3.8 B  | **15 M**           |
| **训练 loss** | 无法收敛   | **0.35 ± 0.04**    |

---

## 核心脚本片段（精简版）

```python
# 环境 & 离线 W&B
import os, torch
os.environ["WANDB_MODE"] = "offline"
os.environ["WANDB_DIR"]  = "./wandb"
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32  = True

# 自定义 collator 解决 labels padding
def collate_fn(feats):
    batch = tokenizer.pad(feats, padding="longest",
                          pad_to_multiple_of=8, return_tensors="pt")
    L = batch["input_ids"].shape[1]
    batch["labels"] = torch.tensor(
        [f["labels"] + [-100]*(L-len(f["labels"])) for f in feats], dtype=torch.long
    )
    return batch

# Flash-Attn CE & label shift
from flash_attn.losses.cross_entropy import CrossEntropyLoss
flash_ce = CrossEntropyLoss(ignore_index=-100, inplace_backward=True)

class MyTrainer(Trainer):
    def compute_loss(self, model, inputs, **kw):
        labels = inputs["labels"]
        logits = model(input_ids=inputs["input_ids"],
                       attention_mask=inputs["attention_mask"]).logits
        loss = flash_ce(
            logits[:, :-1, :].contiguous().view(-1, logits.size(-1)),
            labels[:, 1:].contiguous().view(-1)
        )
        return loss
```

---

## 复现与监控指南

1. **安装依赖**
  Qwen3 需要比较新的transformers环境，建议torch、transformers、 flash-attn都安装较新版本。
  推理截断使用vLLM对torch之类的依赖要求比较严格，建议重新创建新环境进行配置。
  

2. **启动训练**
   训练 Qwen3-4B 的示例代码
   ```bash
   nohup bash -c 'CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 deepspeed train_qwen3.py' > /datanfs4/xinzheyu/project/overthinking_Dr.Dai/code/log/qwen3_4b/token16k2.log 2>&1 &
   ```
   
   训练 Qwen2.5-3B-Instruct 的示例代码
   ```bash
   nohup bash -c 'CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 deepspeed train_deepspeed.py' > /datanfs4/xinzheyu/project/overthinking_Dr.Dai/code/log/qwen25_3b_instruct/token16k2.log 2>&1 &
   ```
   
    Qwen2.5-3B-Instruct 的训练大概需要一个多小时， Qwen3-4B 的训练大概需要三个小时。 差异很大，可能是模型结构改动比较大？
    
    Todo:
        看Qwen2.5 和 Qwen3 的文档，比较一下差异点。
