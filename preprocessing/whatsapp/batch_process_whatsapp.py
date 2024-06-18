import zipfile
from pathlib import Path
from load_whatsapp import preprocess_convo
from fire import Fire


def main(input_path, output_path):
    folder_path = Path(input_path)

    # extract zip files
    for zip_path in folder_path.glob('*.zip'):
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(folder_path)

    # convert all convos into jsonl
    for fp in folder_path.glob('WhatsApp Chat*.txt'):
        file_output_path = folder_path / Path(fp.stem.replace('WhatsApp Chat with ', 'wa_finetune_').replace(' ', '_').lower() + '.jsonl')
        print(file_output_path)
        preprocess_convo(fp, file_output_path, "user")


    # merge all jsonls into one consolidated file
    all_content = ""
    for fp in folder_path.glob('wa_finetune_*.jsonl'):
        print(fp)
        all_content += fp.read_text()


    with open(output_path, 'w') as f:
        f.write(all_content)


if __name__ == "__main__":
    Fire(main)