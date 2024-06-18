import logging
import os
from telegram import Update
from telegram.ext import filters, ApplicationBuilder, ContextTypes, CommandHandler, MessageHandler
from transformers import AutoTokenizer
# from gradio_client import Client
from dotenv import load_dotenv
import requests

load_dotenv()

TELEGRAM_TOKEN = os.getenv('TELEGRAM_TOKEN')

generate_url = "https://capdev.govtext.gov.sg:9092/generate/"
# client = Client("https://capdev.govtext.gov.sg:9092/")

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

user_sessions = {}


base_model_id = "alpindale/Mistral-7B-v0.2-hf"
eval_tokenizer = AutoTokenizer.from_pretrained(base_model_id, add_bos_token=True, add_special_tokens=False, trust_remote_code=True, use_fast=True)
prompt_template = "<start_header_id>{role}<end_header_id>{message}<|eot_id|>"
input_max_length = 5500

def get_encoded_length(text):
    encoded = eval_tokenizer.encode(text)
    return len(encoded)


def format_query(history, message):
    formatted_messages_list = []
    new_msg_formatted = prompt_template.format(role="user", message=message)
    new_msg_formatted_length = get_encoded_length(new_msg_formatted)
    total_content_length = new_msg_formatted_length

    formatted_messages_list = [new_msg_formatted]
    for user_msg, system_msg in reversed(history):
        system_msg_formatted = prompt_template.format(role="system", message=system_msg)
        system_msg_formatted_length = get_encoded_length(system_msg_formatted)
        if total_content_length + system_msg_formatted_length < input_max_length:
            formatted_messages_list.insert(0,system_msg_formatted)
        else:
            break

        user_msg_formatted = prompt_template.format(role="user", message=user_msg)
        user_msg_formatted_length = get_encoded_length(user_msg_formatted)
        if total_content_length + user_msg_formatted_length < input_max_length:
            formatted_messages_list.insert(0, user_msg_formatted)

        else:
            break


    # print(formatted_messages_list)

    return "".join(formatted_messages_list) + "<start_header_id>system<end_header_id>"


def get_reply_from_chatbot(message, history):

    query_formatted = format_query(history, message)
    # print(query_formatted)

    # using gradio client
    # result = client.predict(
	# 	eval_prompt=query_formatted,
	# 	temperature=0.7,
	# 	max_new_tokens=100,
	# 	api_name="/predict"
    # )


    # using fastapi
    headers = {
        'accept': 'application/json',
        'Content-Type': 'application/json',
    }

    json_data = {
        'prompt': query_formatted,
        'temperature': 0.7,
        'max_new_tokens': 100,
        'repetition_penalty': 1.15,
        'custom_stop_tokens': '<|eot_id|>',
    }

    response = requests.post(generate_url, headers=headers, json=json_data)
    if response.status_code == 200:
        result = response.json()['generated_text']
    else:
        return response.json()
    
    # find the last generated message
    response_str = result.split("<start_header_id>system<end_header_id>")[-1]
    response_str = response_str.split("<|eot_id|>")[0].strip()

    return response_str

welcome_message = """
I am Watson. Yes, I think I am. Talk to me!

Use the following commands if needed:
/start: Start a new conversation
/clear: Clear the chat history and continue the conversation
/stop: End the conversation 
"""

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_chat.id
    user_sessions[user_id] = []
    await context.bot.send_message(chat_id=update.effective_chat.id, text=welcome_message)

async def clear(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_sessions[update.effective_chat.id] = []
    await context.bot.send_message(chat_id=update.effective_chat.id, text="Restarting a new chat session!")

async def stop(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await context.bot.send_message(chat_id=update.effective_chat.id, text="Goodbye!")

async def echo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_chat.id
    print(context)
    received =  'User: ' + update.message.text
    response = 'Bot: ' + update.message.text
    user_sessions[user_id] = user_sessions[user_id] + [received + '\n' + response]
    await context.bot.send_message(chat_id=update.effective_chat.id, text='\n'.join(user_sessions[user_id]))


async def talk_to_llm(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_chat.id
    history = user_sessions[user_id]
    user_msg  =  update.message.text
    system_msg = get_reply_from_chatbot(user_msg, history)
    user_sessions[user_id] = history + [(user_msg, system_msg)]
    await context.bot.send_message(chat_id=update.effective_chat.id, text=system_msg)

async def unknown(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await context.bot.send_message(chat_id=update.effective_chat.id, text="Sorry, I didn't understand that command.")

if __name__ == '__main__':
    application = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
    
    start_handler = CommandHandler('start', start)
    clear_handler = CommandHandler('clear', clear)
    stop_handler = CommandHandler('stop', stop)
    # echo_handler = MessageHandler(filters.TEXT & (~filters.COMMAND), echo)
    llm_handler = MessageHandler(filters.TEXT & (~filters.COMMAND), talk_to_llm)
    unknown_handler = MessageHandler(filters.COMMAND, unknown)


    application.add_handler(start_handler)
    application.add_handler(clear_handler)
    application.add_handler(stop_handler)
    application.add_handler(llm_handler)
    application.add_handler(unknown_handler)

    
    application.run_polling()