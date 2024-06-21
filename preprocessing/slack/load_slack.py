from langchain_community.chat_loaders.slack import SlackChatLoader
from langchain_community.chat_loaders.utils import (
    map_ai_messages,
    merge_chat_runs,
)
from langchain_core.chat_sessions import ChatSession
from fire import Fire
from typing import List
from preprocessing.whatsapp.load_whatsapp import encoder, HISTORY_MAX_TOKENS, CONVO_MIN_TOKENS
from tqdm.auto import tqdm
from langchain_core.messages.human import HumanMessage
from langchain_core.messages.ai import AIMessage
import pandas as pd
import json

def preprocess_convo(input_path, output_path):
    loader = SlackChatLoader(
        path=input_path
    )

    raw_messages = loader.lazy_load()
    # Merge consecutive messages from the same sender into a single message
    merged_messages = merge_chat_runs(raw_messages)
    # Convert messages from "UMCH0973Q" to AI messages
    messages: List[ChatSession] = list(
        map_ai_messages(merged_messages, sender="UMCH0973Q")
    )

    query = []


    for convo in tqdm(messages):
        conversation = []
        token_len = 0
        for row in convo["messages"]:
            if type(row) == HumanMessage:
                row_role = "user"
            else:
                row_role = "system"
        
            row_message = row.content

            chat_message_formatted = "<start_header_id>{role}<end_header_id>{message}".format(role=row_role, message=row_message)
            chat_message_formatted_len = len(encoder.encode(chat_message_formatted))
      
            # Add message to conversation if it's within one hour from the previous message, and the history is less than 5000 tokens
            if  token_len + chat_message_formatted_len<HISTORY_MAX_TOKENS: 
                conversation.append(chat_message_formatted)
        
            # If  history length is too long, create a new conversation to hold the current message
            else:
                # Write the current conversation to the final query
                query.append(conversation)
                # reset
                conversation = [chat_message_formatted]
                token_len = chat_message_formatted_len
        
        # write out last conversation
        query.append(conversation)


    df_model = pd.DataFrame({"query": query})
    df_model['query_str'] = df_model['query'].apply(lambda x: "<|eot_id|>".join(x))
    df_model['query_len'] = df_model['query_str'].apply(lambda x: len(encoder.encode(x)))
    print(df_model['query_len'].describe())
    
    # remove short conversations
    df_model_filtered = df_model[df_model['query_len'] > 0]
    # df_model_filtered = df_model[df_model['query_len'] > CONVO_MIN_TOKENS]


    # write output as json lines in the format {'input': formatted_message} so that we can use it for finetuning
    with open(output_path, 'w') as f:
        for _, row in df_model_filtered.iterrows():
            f.write(json.dumps({'input': row['query_str']}) + '\n')      






if __name__ == "__main__":
    # Fire(preprocess_convo)
    input_path = "/mnt/c/Users/watso/Downloads/slackdump_Windows_x86_64/amelia/DM08UFH0A.zip"
    output_path = "/mnt/c/Users/watso/Downloads/slackdump_Windows_x86_64/amelia/DM08UFH0A.jsonl"
    preprocess_convo(input_path, output_path)