#!/usr/bin/env python3
from __future__ import annotations

import argparse

from datasets import load_dataset
from transformers import TrainingArguments
from trl import DPOTrainer
from unsloth import FastLanguageModel


def main() -> None:
    parser = argparse.ArgumentParser(description="Safety-focused DPO for Qwen 2.5 3B QLoRA adapters.")
    parser.add_argument("--dataset", default="data/fine_tuning/dpo_preferences_seed.jsonl")
    parser.add_argument("--output-dir", default="artifacts/qwen25_3b_dpo_lora")
    parser.add_argument("--model", default="unsloth/Qwen2.5-3B-Instruct-bnb-4bit")
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--steps", type=int, default=100)
    args = parser.parse_args()

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model,
        max_seq_length=args.max_seq_length,
        dtype=None,
        load_in_4bit=True,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_alpha=32,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )

    dataset = load_dataset("json", data_files=args.dataset, split="train")

    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        tokenizer=tokenizer,
        train_dataset=dataset,
        beta=0.1,
        max_length=args.max_seq_length,
        max_prompt_length=768,
        args=TrainingArguments(
            output_dir=args.output_dir,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=8,
            learning_rate=1e-5,
            max_steps=args.steps,
            warmup_steps=10,
            logging_steps=5,
            save_steps=args.steps,
            optim="adamw_8bit",
            report_to="none",
        ),
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
