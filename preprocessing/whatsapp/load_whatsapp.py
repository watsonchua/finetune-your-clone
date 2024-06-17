# Reference: https://www.linkedin.com/pulse/building-chatbot-fine-tune-llms-whatsapp-data-daniel-pleus/

import pandas as pd
from whatstk import WhatsAppChat
from transformers import AutoTokenizer
from fire import Fire
from tqdm.auto import tqdm
import json

base_model_id = "alpindale/Mistral-7B-v0.2-hf"
encoder = AutoTokenizer.from_pretrained(base_model_id, add_bos_token=False, add_special_tokens=False, trust_remote_code=True, use_fast=True, force_download=False)

def collapse_messages(df):
    if len(df) == 0:
        return df
    
    new_data = []
    
    df_temp = df.copy()
    current_row = df_temp.iloc[0]
    current_role = current_row["chat_message"][0]

    for _, row in tqdm(df_temp[1:].iterrows(), total=len(df_temp)-1):
        row_role = row["chat_message"][0]
        row_message = row["chat_message"][1]

        if row_role == current_role and row["time_delta"] < 600:
            current_row["chat_message"] = (current_row["chat_message"][0], current_row["chat_message"][1] + "\n" + row_message)
        else:
            new_data.append(current_row.to_dict())
            current_row = row
            current_role = row_role
    
    # add last row
    new_data.append(current_row.to_dict())

    return pd.DataFrame(new_data)



def preprocess_convo(input_path, output_path, role="user", chat_owner="Watson"):
    chat = WhatsAppChat.from_source(filepath=input_path)
    df = chat.df

    # Calculate time passed since previous message
    df["date_previous"] = df["date"].shift(periods=1)
    df["time_delta"] = (df["date"]-df["date_previous"]).dt.total_seconds()
    df["chat_message"] = df.apply(lambda x: ("system" if x["username"] == chat_owner else role, x["message"]), axis=1)
    df = collapse_messages(df)

    query = []
    conversation = []
    token_len = 0


    for _, row in tqdm(df.iterrows(), total=len(df)):
        row_role = row["chat_message"][0]
        row_message = row["chat_message"][1]

        if row_message == "<Media omitted>":
            continue

        chat_message_formatted = "<start_header_id>{role}<end_header_id>{message}".format(role=row_role, message=row_message)
        chat_message_formatted_len = len(encoder.encode(chat_message_formatted))
      
        # This defines how close messages should be to be in the same conversation, and what is the maximum length of each conversation
        if row["time_delta"]<3600 and token_len + chat_message_formatted_len<5000: 
            conversation.append(chat_message_formatted)
        
        # if conversation is more than one hour later or length is too long
        # add previous conversation to query and create new conversation
        else:
            query.append(conversation)
            # reset
            conversation = [chat_message_formatted]
            token_len = chat_message_formatted_len

    # write out the last conversation
    query.append(conversation)


    df_model = pd.DataFrame({"query": query})
    df_model['query_str'] = df_model['query'].apply(lambda x: "<|eot_id|>".join(x))
    df_model['query_len'] = df_model['query_str'].apply(lambda x: len(encoder.encode(x)))
    
    # remove short conversations
    df_model_filtered = df_model[df_model['query_len'] > 100]


    # write output as json lines
    with open(output_path, 'w') as f:
        for index, row in df_model_filtered.iterrows():
            f.write(json.dumps({'input': row['query_str']}) + '\n')        



if __name__ == "__main__":
    input_path = "./data/WhatsApp Chat with Wan Ching/WhatsApp Chat with Wan Ching.txt" 
    output_path = "./data/WhatsApp Chat with Wan Ching/wa_finetune_2.jsonl"
    role = "wife"
    preprocess_convo(input_path, output_path, role)
    # fire(main)