# Finetuning

This README file is specifically for finetuning the model.

## Setting up the enviroment

1. Create a new virtual environment for preprocessing using the command `python3 -m venv path_to_your_env`. E.g. `python3 -m venv ~/environments/finetune-training`. You can use the previous environment set up for preprocessing.
2. Activate the environment. `source path_to_your_env/bin/activate`
3. Change directory to the finetuning folder using `cd finetuning`. You should be in the folder `finetune-your-clone/finetuning` now.
4. Install the necessary packages: `pip install -r requirements.txt`

## Finetuning

1. If you are doing the finetuning on a different machine from the preprocessing one, copy the jsonl file generated previously and put it in `finetune-your-clone/data/whatsapp/`.
2. Run the bash script `bash finetune.sh`. The model will start finetuning and the model will be saved in `finetune-your-clone/data/finetuned_models/my_whatsapp_clone`. You can change the output model name and the training parameters.
