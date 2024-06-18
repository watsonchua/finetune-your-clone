# Reference: https://github.com/brevdev/notebooks/blob/main/mistral-finetune-own-data.ipynb


from datasets import load_dataset
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
from accelerate import FullyShardedDataParallelPlugin, Accelerator
from torch.distributed.fsdp.fully_sharded_data_parallel import FullOptimStateDictConfig, FullStateDictConfig
import transformers
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
        base_model_id="alpindale/Mistral-7B-v0.2-hf",
        max_length=5500,
        lora_r=32,
        lora_alpha=64,
        lora_dropout=0.05,
        batch_size=1,
        epochs=5,
        gradient_accumulation_steps=8
):
    # Load model for 4-bit quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    train_dataset = load_dataset('json', data_files=train_dataset_path, split='train')
    eval_dataset = load_dataset('json', data_files=eval_dataset_path, split='train') if eval_dataset_path is not None else None
    

    model = AutoModelForCausalLM.from_pretrained(base_model_id, quantization_config=bnb_config)

    # The tokenizer has been configured to add the BOS and EOS tokens. We do not have to add them in our inputs anymore.
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_id,
        padding_side="left",
        add_eos_token=True,
        add_bos_token=True,
    )

    # Use the unknown token for padding
    tokenizer.pad_token = tokenizer.unk_token

    def generate_and_tokenize_prompt(example):
            result = tokenizer(
                example['input'], # data is just the text in the input field of the json line
                truncation=True,
                max_length=max_length,
                padding="max_length",
            )
            result["labels"] = result["input_ids"].copy()
            return result


    tokenized_train_dataset = train_dataset.map(generate_and_tokenize_prompt)
    # we are not using a validation dataset for now
    tokenized_val_dataset = eval_dataset.map(generate_and_tokenize_prompt) if eval_dataset is not None else None



    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)
    print(model)


    config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        # target modules for lora specific to mistral
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
        lora_dropout=lora_dropout,
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

    if torch.cuda.device_count() > 1: 
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
            num_train_epochs=epochs,
            learning_rate=2.5e-5,
            bf16=True,
            optim="paged_adamw_8bit",
            logging_steps=100,              
            logging_dir="./logs",       
            save_strategy="epoch", 
            report_to="none",
        ),
        data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )

    model.config.use_cache = False 
    trainer.train()
    trainer.save_model()



if __name__ == "__main__":
    Fire(main)