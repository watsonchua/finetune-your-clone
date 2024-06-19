base_model_id="alpindale/Mistral-7B-v0.2-hf" lora_weights="../data/finetuned_models/my_whatsapp_clone" CUDA_VISIBLE_DEVICES=0 uvicorn generate_mistral:app --port 9092 --host 0.0.0.0 --reload
