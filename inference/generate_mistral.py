from transformers import AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer
from transformers import StoppingCriteria, StoppingCriteriaList
import torch
from peft import PeftModel
from fire import Fire



prompt_template = "<human>: {instruction} \n\n<assistant>:"

def main(
        base_model_id="mistralai/Mistral-7B-v0.1",
        model_path=None,
        lora_weights=None,
        max_new_tokens=100,
        repetition_penalty=1.15,
        temperature=0.01,
        gradio=False,
        gradio_port=8503,
        format_prompt=True, # format prompt in <human> <assistant> format
        truncate_prompt=True,
        fastapi=False,
        quantize=True,
        custom_stop_tokens=None
):

    if quantize:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
    else:
        bnb_config = None

    base_model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path =base_model_id if model_path is None else model_path,  # Mistral, same as before
        quantization_config=bnb_config,  # Same quantization config as before
        # device_map="auto", # causes ValueError for Llama2-70B
        trust_remote_code=True,
        token=True
    )

    eval_tokenizer = AutoTokenizer.from_pretrained(base_model_id, add_bos_token=True, trust_remote_code=True, use_fast=True)
    # unk token was used as pad token during finetuning
    eval_tokenizer.pad_token = eval_tokenizer.unk_token
    if lora_weights is None:
        ft_model = base_model
    else:
        ft_model = PeftModel.from_pretrained(base_model, lora_weights)
    
    
    # device = ft_model.device
    device = torch.device("cuda")
    print(device)
    ft_model.to(device)
    ft_model.eval()

    print(ft_model)


    # define stopping criteria

    # class StoppingCriteriaSub(StoppingCriteria):
    #     def __init__(self, stops = [], encounters=1):
    #         super().__init__()
    #         self.stops = [stop.to("cuda") for stop in stops]

    #     def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
    #         for stop in self.stops:
    #             if torch.all((stop == input_ids[0][-len(stop):])).item():
    #                 return True

    #         return False

    # stop_words = ["<human>:", "<bot>:", "<assistant>:", "</s>", "\n--"]
    # stop_words_ids = [eval_tokenizer(stop_word, return_tensors='pt')['input_ids'].squeeze() for stop_word in stop_words]
    # stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])
    
    
    
    def generate(eval_prompt, temperature, max_new_tokens):
        
        if format_prompt:
            eval_prompt = prompt_template.format(instruction=eval_prompt)
        

        # model_input = eval_tokenizer(eval_prompt, return_tensors="pt").to("cuda")
        model_input = eval_tokenizer(eval_prompt, return_tensors="pt").to(device)



        with torch.no_grad():
            if custom_stop_tokens is None:
                model_output = ft_model.generate(**model_input, max_new_tokens=max_new_tokens, repetition_penalty=repetition_penalty, temperature=temperature)[0]
            else:
                model_output = ft_model.generate(**model_input, max_new_tokens=max_new_tokens, repetition_penalty=repetition_penalty, stop_strings=custom_stop_tokens.split(","), tokenizer=eval_tokenizer, temperature=temperature)[0]

            text_output = eval_tokenizer.decode(model_output, skip_special_tokens=True)
            print(text_output)
            if format_prompt and truncate_prompt:
                return text_output.split("<human>:")[1].split("<assistant>:")[1]
            else:
                return text_output


    
    if not gradio:
        while True:
            eval_prompt = input("Enter prompt:")
            if not eval_prompt.strip():
                break

            response = generate(eval_prompt, temperature, max_new_tokens)
            print(response)
    
        
    else:
        # use gradio interface
        import gradio as gr
        
        demo = gr.Interface(
            fn=generate,
            inputs=[
                    gr.Textbox(label="Question", placeholder="What do I do in the event of a security incident?"),
                    gr.Slider(0.01, 1.0, value=0.01, label="Temperature"),
                    gr.Slider(100, max_new_tokens if max_new_tokens>100 else 400, value=100, label="Max Output Tokens"),
                    # gr.Slider(50,400,value=100,label="Repetition Penalty"),                
            ],

            outputs=[gr.Textbox(label="Answer")],
            allow_flagging="never",
        )

        demo.launch(server_name="0.0.0.0", server_port=gradio_port)

        
      


if __name__ == "__main__":
    Fire(main)