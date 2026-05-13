# Stage 5 Fine-Tuning and Alignment Materials

This folder contains reproducible starter workflows for the planned Qwen 2.5 3B alignment path.

## Datasets

- `data/fine_tuning/tool_calling_sft_seed.jsonl`: small ChatML-style examples for tool-calling supervised fine-tuning.
- `data/fine_tuning/dpo_preferences_seed.jsonl`: starter preference pairs for safety-focused DPO.

The seed files are intentionally small. They document the format and support demo review; real training should expand them with human-reviewed examples.

## Scripts

- `training/sft/train_qwen25_qlora_sft.py`: QLoRA SFT template for tool-calling and appliance-domain examples.
- `training/dpo/train_qwen25_qlora_dpo.py`: DPO template for chosen/rejected safety preference pairs.

Both scripts are designed for a Colab T4 or similar GPU runtime with `unsloth`, `trl`, `datasets`, and `transformers` installed.

## Recommended Order

1. Run SFT on a larger tool-calling set.
2. Continue SFT on appliance troubleshooting conversations.
3. Run DPO with safety preference pairs.
4. Export LoRA adapters and configure the backend SLM wrapper to serve the aligned model through LM Studio.
