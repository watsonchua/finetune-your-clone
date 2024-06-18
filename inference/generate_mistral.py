from transformers import AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer
import torch
from peft import PeftModel
from fire import Fire



from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
from typing import Dict, List
import os


# load model
base_model_id = os.getenv("base_model_id")
lora_weights = os.getenv("lora_weights")


bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)


base_model = AutoModelForCausalLM.from_pretrained(
    pretrained_model_name_or_path=base_model_id,
    quantization_config=bnb_config,  
    trust_remote_code=True,
    token=True
)

eval_tokenizer = AutoTokenizer.from_pretrained(base_model_id, add_bos_token=True, trust_remote_code=True, use_fast=True)
# unk token was used as pad token during finetuning, must set the same here
eval_tokenizer.pad_token = eval_tokenizer.unk_token
ft_model = PeftModel.from_pretrained(base_model, lora_weights)


device = torch.device("cuda")
print(device)
ft_model.to(device)
ft_model.eval()

print(ft_model)


# end load model




def generate_with_model(eval_prompt, temperature, repetition_penalty, custom_stop_tokens, max_new_tokens):        
    model_input = eval_tokenizer(eval_prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        if custom_stop_tokens is None:
            model_output = ft_model.generate(**model_input, max_new_tokens=max_new_tokens, repetition_penalty=repetition_penalty, temperature=temperature)[0]
        else:
            model_output = ft_model.generate(**model_input, max_new_tokens=max_new_tokens, repetition_penalty=repetition_penalty, stop_strings=custom_stop_tokens.split(","), tokenizer=eval_tokenizer, temperature=temperature)[0]

        text_output = eval_tokenizer.decode(model_output, skip_special_tokens=True)
        print(text_output)
        return text_output




# start the fast api app

app = FastAPI()


class CompletionQuery(BaseModel):
    prompt: str = "Hello how are you?"
    temperature: float = 0.5
    max_new_tokens: int = 100
    repetition_penalty: float = 1.15
    custom_stop_tokens: str = "<|eot_id|>"



@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post("/generate/")
async def generate(query: CompletionQuery):    
    response = generate_with_model(eval_prompt=query.prompt, temperature=query.temperature, repetition_penalty=query.repetition_penalty, custom_stop_tokens=query.custom_stop_tokens, max_new_tokens=query.max_new_tokens)
    return {"generated_text": response}
