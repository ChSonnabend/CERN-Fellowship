#!/usr/bin/env python3
import argparse
import os
from dataclasses import dataclass
from typing import Dict, List

import torch
from torch.utils.data import Dataset

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)

from peft import LoraConfig, get_peft_model, TaskType

# Example usage:
# export HF_HOME=/scratch/$USER/hf_cache
# python3.9 -m torch.distributed.run --nproc_per_node=8 /scratch/csonnabe/cern-fellowship/LLM_finetune/train_lora_text2text.py --model_name /scratch/$USER/models/DeepSeek-R1-Distill-Llama-70B --data_jsonl o2_sft_last100/dataset.jsonl --out_dir lora_o2physics_70b --max_length 4096 --batch_size 1 --grad_accum 16 --epochs 1 --lr 2e-5 --gradient_checkpointing

# or with accelerate:
# export PATH=$HOME/.local/bin:$PATH
# accelerate config
# accelerate launch --config_file /scratch/csonnabe/hf_cache/accelerate/default_config.yaml --num_processes 8 /scratch/csonnabe/cern-fellowship/LLM_finetune/train_lora_text2text.py --model_name /scratch/$USER/models/DeepSeek-R1-Distill-Llama-70B --data_jsonl o2_sft_last100/dataset.jsonl --out_dir lora_o2physics_70b_ds --max_length 2048 --batch_size 1 --grad_accum 16 --epochs 1 --lr 2e-5 --gradient_checkpointing


# -------------------------
# Dataset: 2 files, line-aligned
# -------------------------
class PairedTextDataset(Dataset):
    def __init__(self, inputs_path: str, outputs_path: str):
        with open(inputs_path, "r", encoding="utf-8") as f:
            self.inputs = [line.rstrip("\n") for line in f]
        with open(outputs_path, "r", encoding="utf-8") as f:
            self.outputs = [line.rstrip("\n") for line in f]

        if len(self.inputs) != len(self.outputs):
            raise ValueError(
                f"inputs lines ({len(self.inputs)}) != outputs lines ({len(self.outputs)})"
            )

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx: int):
        return {"prompt": self.inputs[idx], "answer": self.outputs[idx]}

class JsonlPRDataset(Dataset):
    def __init__(self, jsonl_path: str):
        self.rows = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)
                instr = (obj.get("instruction") or "").strip()
                ctx = (obj.get("context") or "").strip()
                diff = (obj.get("completion") or "").strip()

                prompt = (
                    f"{instr}\n\n"
                    f"Context:\n{ctx}\n"
                )
                self.rows.append({"prompt": prompt, "answer": diff})

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx: int):
        return self.rows[idx]


# -------------------------
# Collator: build prompt+answer, mask prompt tokens in labels
# -------------------------
@dataclass
class SFTDataCollator:
    tokenizer: AutoTokenizer
    max_length: int = 4096
    add_eos: bool = True

    def __call__(self, batch: List[Dict[str, str]]) -> Dict[str, torch.Tensor]:
        # Simple "instruction -> completion" format.
        # You can swap this for a chat template later if you want.
        prompts = [ex["prompt"] for ex in batch]
        answers = [ex["answer"] for ex in batch]

        # Build full texts and also compute prompt token lengths to mask them.
        full_texts = []
        prompt_texts = []
        for p, a in zip(prompts, answers):
            # Keep it explicit to reduce ambiguity for code tasks.
            prompt = f"### Instruction:\n{p}\n\n### Response:\n"
            full = prompt + a
            if self.add_eos and (not full.endswith(self.tokenizer.eos_token or "")):
                full += self.tokenizer.eos_token or ""
            full_texts.append(full)
            prompt_texts.append(prompt)

        tok_full = self.tokenizer(
            full_texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        tok_prompt = self.tokenizer(
            prompt_texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        input_ids = tok_full["input_ids"]
        attention_mask = tok_full["attention_mask"]

        labels = input_ids.clone()
        # mask prompt tokens so loss is only on the answer part
        for i in range(labels.size(0)):
            prompt_len = int(tok_prompt["attention_mask"][i].sum().item())
            labels[i, :prompt_len] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def build_lora(model, r: int, alpha: int, dropout: float):
    # LLaMA-style projection module names are common in 70B distills.
    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ]
    cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias="none",
        target_modules=target_modules,
    )
    return get_peft_model(model, cfg)


@torch.inference_mode()
def run_inference(model, tokenizer, in_path: str, out_path: str, max_new_tokens: int, temperature: float):
    model.eval()
    with open(in_path, "r", encoding="utf-8") as f:
        inputs = [line.rstrip("\n") for line in f]

    preds = []
    for p in inputs:
        prompt = f"### Instruction:\n{p}\n\n### Response:\n"
        tok = tokenizer(prompt, return_tensors="pt").to(model.device)
        gen = model.generate(
            **tok,
            max_new_tokens=max_new_tokens,
            do_sample=(temperature > 0),
            temperature=max(temperature, 1e-6),
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        text = tokenizer.decode(gen[0], skip_special_tokens=True)
        # keep only response part
        if "### Response:" in text:
            text = text.split("### Response:", 1)[1].lstrip()
        preds.append(text)

    with open(out_path, "w", encoding="utf-8") as f:
        for t in preds:
            f.write(t.replace("\n", "\\n") + "\n")  # keep 1-line-per-sample


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", default="deepseek-ai/DeepSeek-R1-Distill-Llama-70B")
    ap.add_argument("--inputs", help="inputs.txt (one prompt per line)")
    ap.add_argument("--outputs", help="outputs.txt (one target per line)")
    ap.add_argument("--data_jsonl", required=True, help="dataset.jsonl from PR pipeline")
    ap.add_argument("--out_dir", required=True, help="where to save adapter/tokenizer")
    ap.add_argument("--max_length", type=int, default=4096)

    # LoRA
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--lora_dropout", type=float, default=0.05)

    # Training
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--grad_accum", type=int, default=16)
    ap.add_argument("--save_steps", type=int, default=200)
    ap.add_argument("--log_steps", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)

    # Memory helpers
    ap.add_argument("--use_4bit", action="store_true", help="QLoRA (needs bitsandbytes + supported GPUs)")
    ap.add_argument("--gradient_checkpointing", action="store_true")

    # Optional inference
    ap.add_argument("--predict_inputs", default=None, help="inputs file to run inference on after training")
    ap.add_argument("--predict_out", default=None, help="output file for predictions")
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.0)

    args = ap.parse_args()
    set_seed(args.seed)

    os.makedirs(args.out_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model (bf16 recommended on A100/H100; fp16 otherwise)
    quantization_config = None
    model_kwargs = dict(
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map=None,          # let DDP handle device placement
        local_files_only=True,    # load strictly from local path
    )

    if args.use_4bit:
        from transformers import BitsAndBytesConfig
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model_kwargs["quantization_config"] = quantization_config
        # device_map="auto" is typical for single-node inference; for DDP training keep device_map None.

    model = AutoModelForCausalLM.from_pretrained(args.model_name, **model_kwargs)

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    model = build_lora(model, r=args.lora_r, alpha=args.lora_alpha, dropout=args.lora_dropout)
    model.print_trainable_parameters()

    # train_ds = PairedTextDataset(args.inputs, args.outputs)
    train_ds = JsonlPRDataset(args.data_jsonl)
    collator = SFTDataCollator(tokenizer=tokenizer, max_length=args.max_length)

    training_args = TrainingArguments(
        output_dir=args.out_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        logging_steps=args.log_steps,
        save_steps=args.save_steps,
        save_total_limit=3,
        bf16=torch.cuda.is_available(),  # uses bf16 if supported by GPU/driver
        fp16=False,
        optim="adamw_torch",
        report_to="none",
        ddp_find_unused_parameters=False,
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        data_collator=collator,
        tokenizer=tokenizer,
    )

    trainer.train()

    # Save adapter + tokenizer
    model.save_pretrained(args.out_dir)
    tokenizer.save_pretrained(args.out_dir)

    # Optional inference
    if args.predict_inputs and args.predict_out:
        run_inference(
            model=model,
            tokenizer=tokenizer,
            in_path=args.predict_inputs,
            out_path=args.predict_out,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )


if __name__ == "__main__":
    main()