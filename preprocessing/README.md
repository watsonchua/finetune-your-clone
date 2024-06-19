# Preprocessing
This README file is specifically for preprocessing the data.

## Setting up the enviroment

1. Create a new virtual environment for preprocessing using the command `python3 -m venv path_to_your_env`. E.g. `python3 -m venv ~/environments/finetune-preprocessing`
2. Activate the environment. `source path_to_your_env/bin/activate`
3. Change directory to the preprocessing folder using `cd preprocessing`. You should be in the folder `finetune-your-clone/preprocessing` now.
4. Install the necessary packages: `pip install requirements.txt`


## Exporting and preprocessing your WhatsApp Data

1. Export individual chats in your WhatsApp app. You will get a zip file for each chat. More details [here](https://faq.whatsapp.com/1180414079177245/?helpref=uf_share)
2. Create a folder named `data/whatsapp` in the root of this project using `mkdir -p ../data/whatsapp`. You should see the new folder `finetune-your-clone/data/whatsapp`
3. Put the zip files into this the folder.
4. Change directory to the whatsapp folder using `cd whatsapp`. You should be in the folder `finetune-your-clone/preprocessing/whatsapp` now.
5. Run the preprocessing bash file `bash preprocess.sh`. A jsonl file named `consolidated_wa_finetune.jsonl` will be generated from your whatsapp files. This will be your training file. You can edit the output filename in the bash file. 

 
