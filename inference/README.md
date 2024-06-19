# Inference

This README file is specifically for hosting the model for inference.

## Setting up the environment

1. Create a new virtual environment for preprocessing using the command `python3 -m venv path_to_your_env`. E.g. `python3 -m venv ~/environments/finetune-inference`. You can use the previous environment set up for preprocessing.
2. Activate the environment. `source path_to_your_env/bin/activate`
3. Change directory to the finetuning folder using `cd inference`. You should be in the folder `finetune-your-clone/inference` now.
4. Install the necessary packages: `pip install -r requirements.txt`

## Inference

1. If you are doing inference on a different machine from the one used for finetuning, copy the adapter files generated previously and put it in `finetune-your-clone/data/finetuned_models/<your_finetuned_model_name>`, e.g. `finetune-your-clone/data/finetuned_models/my_whatsapp_clone` . These three files must be present:
   - `adapter_config.json`
   - `adapter_model.safetensors`
   - `training_args.bin`
2. Change the parameter `lora_weights` in `generate_mistral_wa.sh` to point to the path where you stored your adapter files. You can also change the port number and CUDA device number.
4. Run the bash script `bash generate_mistral_wa.sh`. This will start a FastAPI service and your model will be available at `http://localhost:<your_selected_port>/generate`.
