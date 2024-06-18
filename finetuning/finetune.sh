mkdir -p ../data/finetuned_models
CUDA_VISIBLE_DEVICES=0 python finetune_mistral.py \
--train_dataset_path "../data/whatsapp/consolidated_wa_finetune.jsonl" \
--output_dir "../data/finetuned_models/my_whatsapp_clone" \
--base_model_id "alpindale/Mistral-7B-v0.2-hf" \
--max_length 5500 \
--lora_r 32 \
--lora_alpha 64 \
--lora_dropout 0.05 \
--batch_size 1 \
--gradient_accumulation_steps 8 \
--epochs 5
