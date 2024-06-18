# Reference: https://www.linkedin.com/pulse/building-chatbot-fine-tune-llms-whatsapp-data-daniel-pleus/

import pandas as pd
import json
from whatstk import WhatsAppChat
from transformers import AutoTokenizer
from tqdm.auto import tqdm


SAME_CONVO_THRESHOLD_SECONDS = 3600
SAME_USER_THRESHOLD_SECONDS = 600
HISTORY_MAX_TOKENS = 5000
CONVO_MIN_TOKENS = 100


# create the tokenizer to measure the length of the text
base_model_id = "alpindale/Mistral-7B-v0.2-hf"
encoder = AutoTokenizer.from_pretrained(base_model_id, add_bos_token=False, add_special_tokens=False, trust_remote_code=True, use_fast=True, force_download=False)

# combine messages from the same sender within 5 mins into a single new-line separated message
def collapse_messages(df, delta_threshold=SAME_USER_THRESHOLD_SECONDS):
    if len(df) == 0:
        return df
    
    new_data = []
    
    df_temp = df.copy()
    current_row = df_temp.iloc[0]
    current_role = current_row["chat_message"][0]

    for _, row in tqdm(df_temp[1:].iterrows(), total=len(df_temp)-1):
        row_role = row["chat_message"][0]
        row_message = row["chat_message"][1]

        if row_role == current_role and row["time_delta"] < delta_threshold:
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

        # Ignore media
        if row_message == "<Media omitted>":
            continue

        chat_message_formatted = "<start_header_id>{role}<end_header_id>{message}".format(role=row_role, message=row_message)
        chat_message_formatted_len = len(encoder.encode(chat_message_formatted))
      
        # Add message to conversation if it's within one hour from the previous message, and the history is less than 5000 tokens
        if row["time_delta"]<SAME_CONVO_THRESHOLD_SECONDS and token_len + chat_message_formatted_len<HISTORY_MAX_TOKENS: 
            conversation.append(chat_message_formatted)
        
        # If message is more than one hour from the previous one or history length is too long, create a new conversation to hold the current message
        else:
            # Write the current conversation to the final query
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
    df_model_filtered = df_model[df_model['query_len'] > CONVO_MIN_TOKENS]


    # write output as json lines in the format {'input': formatted_message} so that we can use it for finetuning
    with open(output_path, 'w') as f:
        for _, row in df_model_filtered.iterrows():
            f.write(json.dumps({'input': row['query_str']}) + '\n')        

