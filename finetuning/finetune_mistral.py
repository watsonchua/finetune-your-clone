# reference: https://github.com/brevdev/notebooks/blob/main/mistral-finetune-own-data.ipynb


from datasets import load_dataset
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
from accelerate import FullyShardedDataParallelPlugin, Accelerator
from torch.distributed.fsdp.fully_sharded_data_parallel import FullOptimStateDictConfig, FullStateDictConfig
# import wandb
import os
import transformers
from datetime import datetime
from fire import Fire

def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )




def main(
        train_dataset_path,
        output_dir,
        eval_dataset_path=None,
        model_path=None,
        base_model_id="mistralai/Mistral-7B-v0.1",
        # project="im8-clauses-qa-finetune",
        # base_model_name="mistral",
        max_length=512,
        lora_r=32,
        lora_alpha=64,
        lora_dropout=0.05,
        batch_size=32,
        epochs=10,
        gradient_accumulation_steps=1
):
    
    # run_name = base_model_name + "-" + project

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    # wandb.login()

    # wandb_project = "im-finetune"
    # if len(wandb_project) > 0:
    #     os.environ["WANDB_PROJECT"] = wandb_project

    
    
    train_dataset = load_dataset('json', data_files=train_dataset_path, split='train')
    eval_dataset = load_dataset('json', data_files=eval_dataset_path, split='train') if eval_dataset_path is not None else None
    
    if model_path is not None:        
        model = AutoModelForCausalLM.from_pretrained(model_path, quantization_config=bnb_config)
    else:
        model = AutoModelForCausalLM.from_pretrained(base_model_id, quantization_config=bnb_config)

    tokenizer = AutoTokenizer.from_pretrained(
        base_model_id,
        padding_side="left",
        add_eos_token=True,
        add_bos_token=True,
    )
    # tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token = tokenizer.unk_token



    def formatting_func(example):
    # text = f"### Question: {example['input']}\n ### Answer: {example['output']}"
    # return text
            # return tokenizer.bos_token + example['input'] + tokenizer.eos_token
        return example['input'] # bos_token and eos_token are already added by tokenizer


    def generate_and_tokenize_prompt(prompt):
            result = tokenizer(
                formatting_func(prompt),
                truncation=True,
                max_length=max_length,
                padding="max_length",
            )
            result["labels"] = result["input_ids"].copy()
            return result


    tokenized_train_dataset = train_dataset.map(generate_and_tokenize_prompt)
    tokenized_val_dataset = eval_dataset.map(generate_and_tokenize_prompt) if eval_dataset is not None else None



    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)
    print(model)


    config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
            "lm_head",
        ],
        bias="none",
        lora_dropout=lora_dropout,  # Conventional
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, config)
    print_trainable_parameters(model)


    fsdp_plugin = FullyShardedDataParallelPlugin(
        state_dict_config=FullStateDictConfig(offload_to_cpu=True, rank0_only=False),
        optim_state_dict_config=FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=False),
    )

    accelerator = Accelerator(fsdp_plugin=fsdp_plugin)
    model = accelerator.prepare_model(model)


    if torch.cuda.device_count() > 1: # If more than 1 GPU
        model.is_parallelizable = True
        model.model_parallel = True


    trainer = transformers.Trainer(
        model=model,
        train_dataset=tokenized_train_dataset,
        # eval_dataset=tokenized_val_dataset,
        args=transformers.TrainingArguments(
            output_dir=output_dir,
            warmup_steps=1,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            gradient_checkpointing=True,
            # max_steps=1000,
            num_train_epochs=epochs,
            learning_rate=2.5e-5, # Want a small lr for finetuning
            bf16=True,
            optim="paged_adamw_8bit",
            logging_steps=100,              # When to start reporting loss
            logging_dir="./logs",        # Directory for storing logs
            save_strategy="epoch",       # Save the model checkpoint every logging step
            # save_steps=100,                # Save checkpoints every 50 steps
            # evaluation_strategy="steps" if tokenized_val_dataset is not None else None, # Evaluate the model every logging step
            # eval_steps=100 if tokenized_val_dataset is not None else None,               # Evaluate and save checkpoints every 50 steps
            # do_eval=True if tokenized_val_dataset is not None else False,                # Perform evaluation at the end of training
            # report_to="wandb",           # Comment this out if you don't want to use weights & baises
            report_to="none",
            # run_name=f"{run_name}-{datetime.now().strftime('%Y-%m-%d-%H-%M')}"          # Name of the W&B run (optional)
        ),
        data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )

    model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
    trainer.train()
    trainer.save_model()



if __name__ == "__main__":
    Fire(main)